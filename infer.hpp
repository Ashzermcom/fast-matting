#ifdef __INFER_HPP__
#define __INFER_HPP__

#include <memory>
#include <string>
#include <vector>


namespace FMInfer {

#define INFO(...) FMInfer::__log_func(__FILE__, __LINE__, __VA_ARGS__)
void __log_func(const char* file, int line, const char* fmt, ...);


enum class DataType : int {
    FLOAT = 0,
    HALF = 1,
    INT8 = 2,
    INT32 = 3,
    BOOL = 4,
    UINT8 = 5
};

class Timer {
public:
    Timer();
    virtual ~Timer();
    void start(void* stream = nullptr);
    float stop(const char* prefix = "Timer", bool print=true);
}

class MemoryBase {
public:
    MemoryBase() = default;
    MemoryBase(void* cpu, size_t cpu_bytes, void* gpu, size_t gpu_bytes);
    virtual ~MemoryBase();
    virtual void* gpu(size_t bytes);
    virtual void* cpu(size_t bytes);
    
}

}

#endif // __INFER_HPP__
