#include <fstream>
#include <iostream>
#include <memory>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferRuntime.h>
#include <opencv2/opencv.hpp>
#include "ilogger.h"
#include "cuda_tools.h"
#include "preprocess_kernel.cuh"


struct AffineMatrix {
	float src2dst[6];
	float dst2src[6];

	/*
	Affine Matrix: shape 2x3
	[[scale, 0, -scale * from.width * 0.5 + to.width * 0.5],
	 [0, scale, -scale * from.height * 0.5 + to.height * 0.5]]
	*/
	void compute_affine_matrix(const cv::Size& from, const cv::Size& to) {
		float scale_x = to.width / (float)from.width;
		float scale_y = to.height / (float)from.height;
		float scale = std::min(scale_x, scale_y);
		src2dst[0] = scale;
		src2dst[1] = 0;
		src2dst[2] = -scale * from.width * 0.5 + to.width * 0.5 + scale * 0.5 - 0.5;
		src2dst[3] = 0;
		src2dst[4] = scale;
		src2dst[5] = -scale * from.height * 0.5 + to.height * 0.5 + scale * 0.5 - 0.5;
		
		cv::Mat m2x3_src2dst(2, 3, CV_32F, src2dst);
		cv::Mat m2x3_dst2src(2, 3, CV_32F, dst2src);
		cv::invertAffineTransform(m2x3_src2dst, m2x3_dst2src);
	}

	cv::Mat src2dst_mat() {
		return cv::Mat(2, 3, CV_32F, src2dst);
	}
};

std::vector<unsigned char> loadEngineFile(const std::string& file) {
	std::ifstream ins(file, std::ios::in | std::ios::binary);
	if (!ins.is_open()) {
		return {};
	}
	ins.seekg(0, std::ios::end);
	size_t length = ins.tellg();
	std::vector<uint8_t> data;
	if (length > 0) {
		ins.seekg(0, std::ios::beg);
		data.resize(length);
		ins.read((char*)&data[0], length);
	}
	ins.close();
	return data;
}

template<typename _T>
std::shared_ptr<_T> makeNvShared(_T* ptr) {
	return std::shared_ptr<_T>(ptr, [](_T* p) {p->destroy(); });
}

void inference() {
	ILogger::TrtLogger logger;
	initLibNvInferPlugins(&logger, "");
	auto engine_data = loadEngineFile("modnet_512x512_commodity_int8.engine");
	auto runtime = makeNvShared(nvinfer1::createInferRuntime(logger));
	auto engine = makeNvShared(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
	if (engine == nullptr) {
		printf("Deserialize cuda engine failed.\n");
		runtime->destroy();
		return;
	}

	cudaStream_t stream = nullptr;
	checkCudaRuntime(cudaStreamCreate(&stream));
	auto execution_context = makeNvShared(engine->createExecutionContext());
	// set input tensor size
	int input_batch = 1;
	int input_channel = 3;
	int input_height = 512;
	int input_width = 512;
	int input_size = input_batch * input_channel * input_height * input_width;
	// set host and device memory for input tensor
	float* input_data_host = nullptr;
	float* input_data_device = nullptr;
	checkCudaRuntime(cudaMallocHost(&input_data_host, input_size * sizeof(float)));
	checkCudaRuntime(cudaMalloc(&input_data_device, input_size * sizeof(float)));

	float mean[] = { 0.485, 0.456, 0.406 };
	float std[] = { 0.229, 0.224, 0.225 };

	float* affine_matrix_host = nullptr;
	float* affine_matrix_device = nullptr;

	checkCudaRuntime(cudaMallocHost(&affine_matrix_host, 6 * sizeof(float)));
	checkCudaRuntime(cudaMalloc(&affine_matrix_device, 6 * sizeof(float)));

	AffineMatrix affine_matrix;
	CUDAKernel::Norm normalize = CUDAKernel::Norm::mean_std(mean, std, 1/255.0f, CUDAKernel::ChannelType::invert);

	auto image = cv::imread("demo.png");
	int image_size = image.cols * image.rows * 3;
	affine_matrix.compute_affine_matrix(image.size(), cv::Size(512, 512));

	uint8_t* image_device = nullptr;
	checkCudaRuntime(cudaMalloc(&image_device, image_size * sizeof(uint8_t)));
	memcpy(affine_matrix_host, affine_matrix.dst2src, sizeof(affine_matrix.dst2src));
	checkCudaRuntime(cudaMemcpyAsync(image_device, image.data, image_size, cudaMemcpyHostToDevice, stream));
	checkCudaRuntime(cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(affine_matrix.dst2src), cudaMemcpyHostToDevice, stream));

	CUDAKernel::warp_affine_bilinear_and_normalize_plane(image_device, image.cols * 3, image.cols, image.rows, input_data_device, input_width, input_height, affine_matrix_device, 0, normalize, stream);

	int output_batch = 1;
	int output_channel = 1;
	int output_height = 512;
	int output_width = 512;
	int output_size = output_batch * output_channel * output_height * output_width;
	float* output_data_host = nullptr;
	float* output_data_device = nullptr;
	checkCudaRuntime(cudaMallocHost(&output_data_host, output_size * sizeof(float)));
	checkCudaRuntime(cudaMalloc(&output_data_device, output_size * sizeof(float)));

	auto input_dims = execution_context->getBindingDimensions(0);
	input_dims.d[0] = input_batch;
	execution_context->setBindingDimensions(0, input_dims);
	float* bindings[] = { input_data_device, output_data_device };
	bool success = execution_context->enqueueV2((void**)bindings, stream, nullptr);
	cv::Mat output_result(output_height, output_width, CV_32F);
	checkCudaRuntime(cudaMemcpyAsync(output_result.data, output_data_device, output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
	checkCudaRuntime(cudaStreamSynchronize(stream));
	checkCudaRuntime(cudaPeekAtLastError());

	checkCudaRuntime(cudaStreamDestroy(stream));
	checkCudaRuntime(cudaFreeHost(input_data_host));
	checkCudaRuntime(cudaFreeHost(output_data_host));
	checkCudaRuntime(cudaFree(input_data_device));
	checkCudaRuntime(cudaFree(output_data_device));
	checkCudaRuntime(cudaFree(affine_matrix_device));
	checkCudaRuntime(cudaFree(image_device));
}

int main() {
	CUDATools::device_description(0);
	inference();
	return 0;
}
