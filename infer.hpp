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
    void release_gpu();
    void release_cpu();
    void release();
    inline bool owner_gpu() const { return owner_gpu_; }
    inline bool owner_cpu() const { return owner_cpu_; }
    inline size_t cpu_bytes() cosnt { return cpu_bytes_; }

protected:
    void* cpu_;
    size_t cpu_bytes_ = 0;
    size_t cpu_capacity_ = 0;
    bool owner_cpu_ = true;
    void* gpu_;
    size_t gpu_bytes_ = 0;
    size_t gpu_capacity_ = 0;
    bool owner_gpu_ = true;
}

template <typename _T>
class Memory : public MemoryBase {
public:
    Memory() = default;
    Memory(const Memory& other) = delete;
    Memory& operator=(const Memory& other) = delete;
    
    virtual _T* gpu(size_t size) override {
        MemoryBase::gpu(size * sizeof(_T));
    }
    virtual _T* cpu(size_t size) override {
        MemoryBase::cpu(size * sizeof(_T));
    }

    inline size_t cpu_size() const { return cpu_bytes_ / sizeof(_T); }
    inline size_t gpu_size() const { return gpu_bytes_ / sizeof(_T); }

    virtual inline _T* gpu() const override { return (_T*) gpu_; }
    virtual inline _T* cpu() const override { return (_T*) cpu_; }
}

class Infer {
public:
    virtual bool forward(const std::vector<void*> &bindings, void* stream=nullptr, void* input_consum_event=nullptr) = 0;
    virtual int index(const std::string& name) = 0;

    virtual std::vector<int> run_dims(const std::string& name) = 0;
    virtual std::vector<int> run_dims(int ibinding) = 0;

    virtual std::vector<int> static_dims(const std::string& name) = 0;
    virtual std::vector<int> static_dims(int ibinding) = 0;

    virtual int num_bindings() = 0;
    virtual bool is_input(int ibinding) = 0;

    virtual bool set_run_dims(const std::string& name, const std::vector<int>& dims) = 0;
    virtual bool set_run_dims(int ibinding, cosnt std::vector<int>& dims);

    virtual DataType dtype(cosnt std::string& name) = 0;
    virtual DataType dtype(int ibinding) = 0;

    virtual bool has_dynamic_dim() = 0;
    virtual void print() = 0;
}

}

#endif // __INFER_HPP__
