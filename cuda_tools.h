#ifndef CUDA_TOOLS_HPP
#define CUDA_TOOLS_HPP

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_THREADS_PER_BLOCK 512

#define checkCudaRuntime(call) CUDATools::check_cuda_runtime(call, #call, __LINE__, __FILE__)

#define checkCudaKernel(...) \
	__VA_ARGS__; \
	do { \
		cudaError_t cudaStatus = cudaPeekAtLastError(); \
		if (cudaStatus != cudaSuccess) { \
		} \
	} while(0);


namespace CUDATools {
    bool check_cuda_runtime(cudaError_t e, const char* call, int iLine, const char* szFile);
	bool check_device_id(int device_id);
	int current_device_id();
	void device_description(int device_id);

	/*
	Auto change to current device and go back when realesed.
	*/
	class AutoDevice {
	public:
		AutoDevice(int device_id = 0);
		virtual ~AutoDevice();
	private:
		int old_ = -1;
	};
}

#endif // !CUDA_TOOLS_HPP

