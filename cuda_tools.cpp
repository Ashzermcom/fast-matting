#include "cuda_tools.h"

namespace CUDATools {
    bool check_cuda_runtime(cudaError_t code, const char* call, int iLine, const char* szFile) {
        if (code != cudaSuccess) {
            const char* err_name = cudaGetErrorName(code);
            const char* err_message = cudaGetErrorString(code);
            printf("[RUNTIME ERROR] %s:%d %s failed.\n code = %s, message = %s\n", szFile, iLine, call, err_name, err_message);
            return false;
        }
        return true;
    }

    bool check_device_id(int device_id) {
        int device_count = -1;
        checkCudaRuntime(cudaGetDeviceCount(&device_count));
        if (device_id < 0 || device_id >= device_count) {
            printf("Invalid device id: %d, count = %d", device_id, device_count);
            return false;
        }
        return true;
    }

    int current_device_id() {
        int device_id = 0;
        checkCudaRuntime(cudaGetDevice(&device_id));
        return device_id;
    }

    void device_description(int device_id) {
        int driver_version, runtime_version;
        cudaDeviceProp device_property;
        checkCudaRuntime(cudaGetDevice(&device_id));
        checkCudaRuntime(cudaGetDeviceProperties(&device_property, device_id));
        printf("Device %d: %s\n", device_id, device_property.name);
        cudaDriverGetVersion(&driver_version);
        cudaRuntimeGetVersion(&runtime_version);
        printf("CUDA driver version / Runtime Version %d.%d / %d.%d\n", driver_version / 1000, (driver_version % 100)/10, runtime_version / 1000, (runtime_version % 100)/10);
        printf("Total amount of global memory: %.0f MBytes\n", (float)device_property.totalGlobalMem / 1048576.0f);
        printf("(%2d) Multiprocessors\n", device_property.multiProcessorCount);
        printf("GPU max clock rate: %.0f MHz (%0.2f GHz)\n", device_property.clockRate * 1e-3f, device_property.clockRate * 1e-6f);
        // Gpu memory info
        printf("Memory clock rate: %.0f Mhz\n", device_property.memoryClockRate * 1e-3f);
        if (device_property.l2CacheSize) {
            printf("L2 cache size: %d bytes\n", device_property.l2CacheSize);
        }
        printf("Total amount of constant memory: %lu bytes\n", device_property.totalConstMem);
        printf("Total amount of shared memory per block: %lu bytes\n", device_property.sharedMemPerBlock);
        printf("Total number of registers available per block: %d\n", device_property.regsPerBlock);
        // Thread info
        printf("Maximum number of threads per multiprocessor: %d\n", device_property.maxThreadsPerMultiProcessor);
        printf("Maximum number of threads per block: %d\n", device_property.maxThreadsPerBlock);
        printf("Max dimension size of a thread grid (x,y,z): (%d, %d, %d)\n", device_property.maxGridSize[0], device_property.maxGridSize[1], device_property.maxGridSize[2]);
        printf("Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n", device_property.maxThreadsDim[0], device_property.maxThreadsDim[1], device_property.maxThreadsDim[2]);
    }

    AutoDevice::AutoDevice(int device_id) {
        checkCudaRuntime(cudaGetDevice(&old_));
        checkCudaRuntime(cudaSetDevice(device_id));
    }

    AutoDevice::~AutoDevice() {
        checkCudaRuntime(cudaSetDevice(old_));
    }
}
