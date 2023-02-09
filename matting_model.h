#pragma once
#include <memory>
#include <string>
#include <future>
#include <vector>
#include <opencv2/opencv.hpp>
#include "trt_tensor.h"


namespace MattingModel {
	void imageToTensor(const cv::Mat& image, std::shared_ptr<TRT::Tensor>& tensor, int ibatch);

	class Infer
	{
	public:
		virtual std::shared_future<TRT::Tensor> preprocess_input(const cv::Mat& image) = 0;
		virtual std::vector<std::shared_future<TRT::Tensor>> preprocess_inputs(const std::vector<cv::Mat>& images) = 0;
	
		std::shared_ptr<Infer> createInfer(const std::string& engineFile, int gpuId);
	};
}
