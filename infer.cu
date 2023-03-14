#include <stdarg.h>
#include <fstream>
#include <sstream>
#include <numeric>
#include <unordered_map>
#include <NvInfer.h>
#include <cuda_runatime.h>

#include "infer.hpp"


namespace trt
{
#define checkRuntime(call) \
do { \
    auto __call_ret_code__ = (call); \
    if (__call_ret_code__ != cudaSuccess) { \
        printf("CUDA runtime error %s # %s, code = %s [%d]", #call, cudaGetErrorString(__call_ret_code__), \
        cudaGetErrorName(__call_ret_code__),__call_ret_code__); \
        abort(); \
    } \
} while (0)

#define checkKernel(...) \
do { \
    {(__VA_ARGS__);} \
    checkRuntime(cudaPeekAtLastError()); \
} while(0)

#define Assert(op) \
do { \
    bool cond = !(!(op)); \
    if (!cond) { \ 
        printf("Assert failed, " #op " : " __VA_ARGS__); \
        abort(); \
    } \
} while(0)

static std::string file_name(const std::string& path, bool include_suffix) {
    if (path.empty()) { return ""; }
    int p = path.rfind('/');
    int e = path.rfind('\\');
    p = std::max(p, e);
    p += 1;
    if (include_suffix) {
        return path.substr(p);
    }
      int u = path.rfind('.');
    if (u == -1) return { path.substr(p); }
    if (u <= p) u = path.size();
    return path.substr(p, u - p);
}

void __log_func(const char *file, int line, const char *fmt, ...) {
    va_list vl;
    va_start(vl, fmt);
    char buffer[2048];
    string filename = file_name(file, true);
    int n = snprintf(buffer, sizeof(buffer), "[%s:%d]: ", filename.c_str(), line);
    vsnprintf(buffer + n, sizeof(buffer) - n, fmt, vl);
    fprintf(stdout, "%s\n", buffer);
}

static std::string format_shape(const Dims &shape) {
  stringstream output;
  char buf[64];
  const char *fmts[] = {"%d", "x%d"};
  for (int i = 0; i < shape.nbDims; ++i) {
    snprintf(buf, sizeof(buf), fmts[i != 0], shape.d[i]);
    output << buf;
  }
  return output.str();
}

Timer::Timer() {
    checkRuntime(cudaEventCreate((cudaEvent_t*)& start_));
    checkRuntime(cudaEventCreate((cudaEvent_t*)& stop_));
}

Timer::~Timer() {
    checkRuntime(cudaEventDestroy((cudaEvent_t) start_));
    checkRuntime(cudaEventDestroy((cudaEvent_t) stop_));
}

void Timer::start(void* stream) {
    stream_ = stream;
    checkRuntime(cudaEventRecord((cudaEvent_t) start_, (cudaEvent_t) stream_));
}

float Timer::stop(const char* prefix, bool print) {
    checkRuntime(cudaEventRecord((cudaEvent_t) stop_, (cudaEvent_t) stream_));
    checkRuntime(cudaEventSynchronize((cudaEvent_t) stop_));

    float latency = 0;
    checkRuntime(cudaEventElapsedTime(&latency, (cudaEvent_t) start_, (cudaEvent_t) stop_));

    if (print) {
        printf("[%s]: %.5f ms\n", prefix, latency);
    }
    return latency;
}

BaseMemory::BaseMemory(void* cpu, size_t cpu_bytes, void* gpu, size_t gpu_bytes) {
    reference(cpu, cpu_bytes, gpu, gpu_bytes);
}

} // namespace trt