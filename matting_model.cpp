#include <mutex>
#include <queue>
#include <atomic>
#include <condition_variable>

#include "ilogger.h"
#include "trt_infer.h"
#include "cuda_tools.h"
#include "matting_model.h"
#include "infer_controller.h"
#include "monopoly_allocator.h"
#include "preprocess_kernel.cuh"


namespace MattingModel {
	/*
	scale, 0, -scale * srcSize.width * 0.5 + dstSize.width * 0.5
	0, scale, -scale * srcSize.height * 0.5 + dstSize.height * 0.5
	*/
	struct AffineMatrix
	{
		float src2dst[6];
		float dst2src[6];

		void compute(const cv::Size& srcSize, const cv::Size& dstSize) {
			float scale_x = dstSize.width / (float)srcSize.width;
			float scale_y = dstSize.height / (float)srcSize.height;

			float scale = std::min(scale_x, scale_y);
			// scale * 0.5 - 0.5 set image to center
			src2dst[0] = scale;
			src2dst[1] = 0;
			src2dst[2] = -scale * srcSize.width * 0.5 + dstSize.width * 0.5 + scale * 0.5 - 0.5;
			src2dst[3] = 0;
			src2dst[4] = scale;
			src2dst[5] = -scale * srcSize.height * 0.5 + dstSize.height * 0.5 + scale * 0.5 - 0.5;

			cv::Mat m2x3_src2dst(2, 3, CV_32F, src2dst);
			cv::Mat m2x3_dst2src(2, 3, CV_32F, dst2src);
			cv::invertAffineTransform(m2x3_src2dst, m2x3_dst2src);
		}

		cv::Mat src2dst_mat() {
			return cv::Mat(2, 3, CV_32F, src2dst);
		}
	};

	using ControllerImpl = InferController<cv::Mat, cv::Mat, std::tuple<std::string, int>, AffineMatrix>;
	class InferImpl : public Infer, public ControllerImpl {
	public:
		virtual ~InferImpl() {
			stop();
		}

		virtual bool startUp(const std::string& file, int gpuId, bool use_multi_preprocess_stream) {
			float mean[] = { 0.485, 0.456, 0.406 };
			float std[] = { 0.229, 0.224, 0.225 };
			normalize_ = CUDAKernel::Norm::mean_std(mean, std, 1/255.0f, CUDAKernel::ChannelType::invert);
			use_multi_preprocess_stream_ = use_multi_preprocess_stream;
			return ControllerImpl::startUp(std::make_tuple(file, gpuId));
		}

		virtual void worker(std::promise<bool>& result) override {
			std::string file = std::get<0>(start_param_);
			int gpuId = std::get<1>(start_param_);
		}

	private:
		int input_width_ = 0;
		int input_height_ = 0;
		int gpu_ = 0;
		CUStream stream_ = nullptr;
		bool use_multi_preprocess_stream_ = false;
		CUDAKernel::Norm normalize_;
	};

}
