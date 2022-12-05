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
	void device_description(int device_id);
}

#endif // !CUDA_TOOLS_HPP

