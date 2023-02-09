#include <algorithm>
#include <fstream>
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime.h>
#include "ilogger.h"
#include "cuda_tools.h"
#include "trt_infer.h"


class Logger : public nvinfer1::ILogger {
public:
	virtual void log(Severity severity, const char* msg) noexcept override {

		if (severity == Severity::kINTERNAL_ERROR) {
			printf("NVInfer INTERNAL_ERROR: %s", msg);
			abort();
		}
		else if (severity == Severity::kERROR) {
			printf("NVInfer: %s", msg);
		}
		else  if (severity == Severity::kWARNING) {
			printf("NVInfer: %s", msg);
		}
		else  if (severity == Severity::kINFO) {
			printf("NVInfer: %s", msg);
		}
		else {
			printf("%s", msg);
		}
	}
};
static Logger gLogger;

namespace TRT {
	template<typename _T>
	std::shared_ptr<_T> makeNvShared(_T* ptr) {
		return std::shared_ptr<_T>(ptr, [](_T* p) {p->destroy(); });
	}

	template<typename _T>
	static void destory_nvidia_pointer(_T* ptr) {
		if (ptr) {
			ptr->destroy();
		}
	}

	class EngineContext {
	public:
		cudaStream_t stream_ = nullptr;
		bool owner_stream_ = false;
		std::shared_ptr<nvinfer1::IExecutionContext> context_;
		std::shared_ptr<nvinfer1::ICudaEngine> engine_;
		std::shared_ptr<nvinfer1::IRuntime> runtime_ = nullptr;
		virtual ~EngineContext() { destroy(); }

		void set_stream(CUStream stream) {
			if (owner_stream_) {
				if (stream_) {
					cudaStreamDestroy(stream_);
				}
				owner_stream_ = false;
			}
			stream_ = stream;
		}

		bool build_model(const void* pdata, size_t size) {
			destroy();
			if (pdata == nullptr || size == 0) {
				return false;
			}
			owner_stream_ = true;
			checkCudaRuntime(cudaStreamCreate(&stream_));
			if (stream_ == nullptr) {
				return false;
			}
			runtime_ = makeNvShared(nvinfer1::createInferRuntime(gLogger));
			if (runtime_ == nullptr) {
				return false;
			}
			engine_ = makeNvShared(runtime_->deserializeCudaEngine(pdata, size, nullptr));
			if (engine_ == nullptr) {
				return false;
			}
			context_ = makeNvShared(engine_->createExecutionContext());
			if (context_ == nullptr) {
				return false;
			}
			return true;
		}

	private:
		void destroy() {
			context_.reset();
			engine_.reset();
			runtime_.reset();
			if (owner_stream_) {
				if (stream_) {
					cudaStreamDestroy(stream_);
				}
			}
			stream_ = nullptr;
		}
	};

	class InferImpl :public Infer {
	public:
		virtual ~InferImpl();
		virtual bool load(const std::string& file);
		virtual void destory();
		virtual void forward(bool sync) override;
		virtual int get_max_batch_size() override;
		virtual CUStream get_stream() override;
		virtual void set_stream(CUStream stream) override;
		virtual void synchronize() override;
		virtual size_t get_device_memory_size() override;
		virtual std::shared_ptr<MixMemory> get_workspace() override;
		virtual std::shared_ptr<Tensor> input(int index = 0) override;
		virtual std::string get_input_name(int index = 0) override;
		virtual std::shared_ptr<Tensor> output(int index = 0) override;
		virtual std::string get_output_name(int index = 0) override;
		virtual std::shared_ptr<Tensor> tensor(const std::string& name) override;
		virtual bool is_output_name(const std::string& name) override;
		virtual bool is_input_name(const std::string& name) override;
		virtual void set_input(int index, std::shared_ptr<Tensor> tensor) override;
		virtual void set_output(int index, std::shared_ptr<Tensor> tensor) override;
		virtual std::shared_ptr<std::vector<uint8_t>> serial_engine() override;

		virtual void print() override;

		virtual int num_output();
		virtual int num_input();
		virtual int device() override;

	private:
		void build_engine_input_and_output_mapper();
		std::vector<std::shared_ptr<Tensor>> inputs_;
		std::vector<std::shared_ptr<Tensor>> outputs_;
		std::vector<int> inputs_map_to_ordered_index_;
		std::vector<int> outputs_map_to_ordered_index_;
		
		std::vector<std::string> inputs_name_;
		std::vector<std::string> outputs_name_;
		std::vector<std::shared_ptr<Tensor>> orderedBlobs_;
		std::map<std::string, int> blobsNameMapper_;

		std::shared_ptr<EngineContext> context_;
		std::vector<void*> bindingsPtr_;
		std::shared_ptr<MixMemory> workspace_;
		int device_ = 0;
	};

	InferImpl::~InferImpl() {
		destory();
	}

	void InferImpl::destory() {
		int old_device = 0;
		checkCudaRuntime(cudaGetDevice(&old_device));
		checkCudaRuntime(cudaSetDevice(device_));
		this->context_.reset();
		this->blobsNameMapper_.clear();
		this->outputs_.clear();
		this->inputs_.clear();
		this->outputs_name_.clear();
		this->inputs_name_.clear();
		checkCudaRuntime(cudaSetDevice(old_device));
	}

	void InferImpl::print() {
		if (!context_) {
			printf("Infer print, nullptr.");
			return;
		}
	}

	std::shared_ptr<std::vector<uint8_t>> InferImpl::serial_engine() {
		auto memory = this->context_->engine_->serialize();
		auto output = std::make_shared<std::vector<uint8_t>>((uint8_t*)memory->data(), (uint8_t*)memory->data()+memory->size());
		memory->destroy();
		return output;
	}

	bool InferImpl::load(const std::string& file) {
		//auto data = load_file(file);
		//if (data.empty()) {
		//	return false;
		//}
		//context_.reset(new EngineContext());
		//if (context_->build_model(data.data(), data.size())) {
		//	context_.reset();
		//	return false;
		//}
		//workspace_.reset(new MixMemory());
		//cudaGetDevice(&device_);
		//return true;
	}

	size_t InferImpl::get_device_memory_size() {
		EngineContext* context = (EngineContext*)this->context_.get();
		return context->context_->getEngine().getDeviceMemorySize();
	}

	static TRT::DataType convert_trt_datatype(nvinfer1::DataType dt) {
		switch (dt)
		{
		case nvinfer1::DataType::kFLOAT:
			return TRT::DataType::Float;
		case nvinfer1::DataType::kHALF:
			return TRT::DataType::Float16;
		case nvinfer1::DataType::kINT8:
			return TRT::DataType::UInt8;
		case nvinfer1::DataType::kINT32:
			return TRT::DataType::Int32;
		default:
			printf("Unsupport data type %d", dt);
			return TRT::DataType::Float;
		}
	}

	void InferImpl::build_engine_input_and_output_mapper() {
		EngineContext* context = (EngineContext*)this->context_.get();
		int nbBindings = context->engine_->getNbBindings();
		int max_batch_size = context->engine_->getMaxBatchSize();

		inputs_.clear();
		inputs_name_.clear();
		outputs_.clear();
		outputs_name_.clear();
		bindingsPtr_.clear();
		blobsNameMapper_.clear();
		for (int i = 0; i < nbBindings; ++i) {
			auto dims = context->engine_->getBindingDimensions(i);
			auto type = context->engine_->getBindingDataType(i);
			const char* bindingName = context->engine_->getBindingName(i);
			dims.d[0] = max_batch_size;
			auto newTensor = std::make_shared<Tensor>(dims.nbDims, dims.d, convert_trt_datatype(type));
			newTensor->set_stream(this->context_->stream_);
			newTensor->set_workspace(this->workspace_);
			if (context->engine_->bindingIsInput(i)) {
				inputs_.push_back(newTensor);
				inputs_name_.push_back(bindingName);
			}
			else {
				outputs_.push_back(newTensor);
				outputs_name_.push_back(bindingName);
			}
			blobsNameMapper_[bindingName] = i;
			orderedBlobs_.push_back(newTensor);
		}
		bindingsPtr_.resize(orderedBlobs_.size());
	}

	void InferImpl::set_stream(CUStream stream) {
		this->context_->set_stream(stream);
	}

	CUStream InferImpl::get_stream() {
		return this->context_->stream_;
	}

	int InferImpl::device() {
		return device_;
	}

	void InferImpl::synchronize() {
		checkCudaRuntime(cudaStreamSynchronize(context_->stream_));
	}

	bool InferImpl::is_output_name(const std::string& name) {
		return std::find(outputs_name_.begin(), outputs_name_.end(), name) != outputs_name_.end();
	}

	bool InferImpl::is_input_name(const std::string& name) {
		return std::find(inputs_name_.begin(), inputs_name_.end(), name) != inputs_name_.end();
	}

	/*
	Get current engine
	*/
	void InferImpl::forward(bool sync) {
		EngineContext* context = (EngineContext*)context_.get();
		int inputBatchSize = inputs_[0]->size(0);
		for (int i = 0; i < context->engine_->getNbBindings(); ++i) {
			auto dims = context->engine_->getBindingDimensions(i);
			auto type = context->engine_->getBindingDataType(i);
			dims.d[0] = 1;
			if (context->engine_->bindingIsInput(i)) {
				context->context_->setBindingDimensions(i, dims);
			}
		}

		for (int i = 0; i < outputs_.size(); ++i) {
			// outputs_[i]->resize_single_dim(0, inputBatchSize);
			outputs_[i]->to_gpu(false);
		}

		for (int i = 0; i < orderedBlobs_.size(); ++i) {
			bindingsPtr_[i] = orderedBlobs_[i]->gpu();
		}

		void** bindingsptr = bindingsPtr_.data();
		bool execute_result = context->context_->enqueueV2(bindingsptr, context_->stream_, nullptr);
		if (!execute_result) {
			auto code = cudaGetLastError();
			printf("execute fail!");
		}

		if (sync) {
			synchronize();
		}
	}

	std::shared_ptr<MixMemory> InferImpl::get_workspace() {
		return workspace_;
	}

	int InferImpl::num_input() {
		return static_cast<int>(this->inputs_.size());
	}

	int InferImpl::num_output() {
		return static_cast<int>(this->outputs_.size());
	}

	void InferImpl::set_input(int index, std::shared_ptr<Tensor> tensor) {
		if (index < 0 || index >= inputs_.size()) {
			printf("Input out of range");
		}
		this->inputs_[index] = tensor;
		int order_index = inputs_map_to_ordered_index_[index];
		this->orderedBlobs_[order_index] = tensor;
	}

	void InferImpl::set_output(int index, std::shared_ptr<Tensor> tensor) {
		if (index < 0 || index >= outputs_.size()) {
			printf("Out out of range");
		}
		this->outputs_[index] = tensor;
		int order_index = outputs_map_to_ordered_index_[index];
		this->orderedBlobs_[order_index] = tensor;
	}

	std::shared_ptr<Tensor> InferImpl::input(int index) {
		if (index < 0 || index >= inputs_.size()) {
			printf("input index out of range");
		}
		return this->inputs_[index];
	}

	std::string InferImpl::get_input_name(int index) {
		if (index < 0 || index >= inputs_name_.size()) {
			printf("input index name out of range");
		}
		return inputs_name_[index];
	}

	std::shared_ptr<Tensor> InferImpl::output(int index) {
		if (index < 0 || index >= outputs_.size()) {
			printf("output index out of range");
		}
		return outputs_[index];
	}

	std::string InferImpl::get_output_name(int index) {
		if (index < 0 || index >= outputs_name_.size()) {
			printf("output index name out of range");
		}
		return outputs_name_[index];
	}

	int InferImpl::get_max_batch_size() {
		assert(this->context_ != nullptr);
		return this->context_->engine_->getMaxBatchSize();
	}

	std::shared_ptr<Tensor> InferImpl::tensor(const std::string& name) {
		auto node = this->blobsNameMapper_.find(name);
		if (node == this->blobsNameMapper_.end()) {
			printf("can not find current tensor %s", name.c_str());
		}
	}

	std::shared_ptr<Infer> load_infer(const std::string& file) {
		std::shared_ptr<InferImpl> Infer(new InferImpl());
		if (!Infer->load(file)) {
			Infer.reset();
		}
		return Infer;
	}

	DeviceMemorySummary get_current_device_summary() {
		DeviceMemorySummary info;
		checkCudaRuntime(cudaMemGetInfo(&info.available, &info.total));
		return info;
	}

	int get_device_count() {
		int count = 0;
		checkCudaRuntime(cudaGetDeviceCount(&count));
		return count;
	}

	int get_device() {
		int device = 0;
		checkCudaRuntime(cudaGetDevice(&device));
		return device;
	}

	void set_device(int device_id) {
		if (device_id == -1) {
			return;
		}
		checkCudaRuntime(cudaSetDevice(device_id));
	}

	bool init_nv_plugins() {
		bool flag = initLibNvInferPlugins(&gLogger, "");
		if (!flag) {
			printf("Init lib nvinfer failed.");
		}
		return flag;
	}
}
