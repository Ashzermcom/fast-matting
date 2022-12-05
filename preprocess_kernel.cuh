#ifndef PREPROCESS_KERNEL_CUH
#define PREPROCESS_KERNEL_CUH
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>


namespace CUDAKernel {
	enum class NormType : int {
		none = 0,
		meanStd = 1,
		alphaBeta = 2
	};

	enum class ChannelType : int {
		none = 0,
		invert = 1
	};

	struct Norm {
		float mean[3];
		float std[3];
		float alpha, beta;
		NormType norm_type = NormType::none;
		ChannelType channel_type = ChannelType::none;
		static Norm mean_std(const float mean[3], const float std[3], float alpha = 1 / 255.0f, ChannelType channel_type = ChannelType::none);
		static Norm alpha_beta(float alpha, float beta=0, ChannelType channel_type = ChannelType::none);
		static Norm none();
	};

	void warp_affine_bilinear_and_normalize_plane(
		uint8_t* src, int src_line_size, int src_width, int src_height, float* dst, int dst_width, int dst_height, float* matrix_2_3, 
		uint8_t const_value, const Norm& norm, cudaStream_t stream
	);
}

#endif // !PREPROCESS_KERNEL_CUH
