#pragma once
#include <queue>
#include <mutex>
#include <string>
#include <future>
#include <memory>
#include <thread>
#include <condition_variable>
#include "trt_infer.h"
#include "monopoly_allocator.h"

template<class Input, class Output, class StartParam = std::tuple<std::string, int>, class JobAdditional=int>
class InferController {
public:
	struct Job {
		Input input;
		Output output;
		JobAdditional additional;
		MonopolyAllocator<TRT::Tensor>::MonopolyDataPointer mono_tensor;
		std::shared_ptr<std::promise<Output>> prom;
	};

	virtual ~InferController()
	{
		stop();
	}

	void stop() {
		run_ = false;
		cond_.notify_all();

		{
			std::unique_lock<std::mutex> l(jobs_lock_);
			while (!jobs_.empty())
			{
				auto& item = jobs_.front();
				if (item.prom) { item.prom->set_value(Output()); }
				jobs_.pop();
			}
		};

		if (worker_) {
			worker_->join();
			worker_.reset();
		}
	}

	bool startUp(const StartParam& param) {
		run_ = true;
		std::promise<bool> prom;
		start_param_ = param;
		worker_ = std::make_shared<std::thread>(&InferController::worker, this, std::ref(prom));
		return prom.get_future().get();
	}

	virtual std::shared_future<Output> commit(const Input& input) {
		Job job;
		job.prom = std::make_shared<std::promise<Output>>();
		if (!preprocess(job, input)) {
			job.prom->set_value(Output());
			return job.prom->get_future();
		}
		{
			std::unique_lock<std::mutex> l(jobs_lock_);
			jobs_.push(job);
		};
		cond_.notify_one();
		return job.prom->get_future();
	}

	virtual std::vector<std::shared_future<Output>> commits(const std::vector<Input>& inputs) {
		int batch_size = std::min((int)inputs.size(), this->tensor_allocator_->capacity());
		std::vector<Job> jobs(intputs.size());
		std::vector<std::shared_future<Output>> results(inputs.size());

		int num_epoch = (inputs.size() + batch_size - 1) / batch_size;
		for (int epoch = 0; epoch < num_epoch; ++epoch) {
			int begin = epoch * batch_size;
			int end = std::min((int)inputs.size(), begin+ batch_size);
			for (int i = begin; i < end; ++i) {
				Job& job = jobs[i];
				job.prom = std::make_shared<std::promise<Output>>();
				if (!preprocess(job, inputs[i])) {
					job.prom->set_value(Output());
				}
				results[i] = job.prom->get_future();
			}
			{
				std::unique_lock<std::mutex> l(jobs_lock_);
				for (int i = begin; i < end; ++i) {
					jobs_.emplace(std::move(jobs[i]));
				}
			}
			cond_.notify_one();
		}
		return results;
	}

protected:
	virtual void worker(std::promise<bool>& result) = 0;
	virtual bool preprocess(Job& job, const Input& input) = 0;
	virtual bool get_jobs_and_wait(std::vector<Job>& fetch_jobs, int max_size) {
		std::unique_lock<std::mutex> l(jobs_lock_);
		cond_.wait(l, [&]() {
			return !run_ || !jobs_.empty();
		});
		if (!run_) { return false; }
		fetch_jobs.clear();
		for (int i = 0; i < max_size && !jobs_.empty(); ++i) {
			fetch_jobs.emplace_back(std::move(jobs_.front()));
			jobs_.pop();
		}
		return true;
	}

	virtual bool get_job_and_wait(Job& fetch_job) {
		std::unique_lock<std::mutex> l(jobs_lock_);
		cond_.wait(l, [&]() {
			return !run_ || !jobs_.empty();
		});

		if (!run_) { return false; }
		fetch_job = std::move(jobs_.front());
		jobs_.pop();
		return true;
	}

	StartParam start_param_;
	std::atomic<bool> run_;
	std::mutex jobs_lock_;
	std::queue<Job> jobs_;
	std::shared_ptr<std::thread> worker_;
	std::condition_variable cond_;
	std::shared_ptr<MonopolyAllocator<TRT::Tensor>> tensor_allocator_;
};
