#pragma once
#include <vector>
#include <mutex>
#include <memory>
#include <condition_variable>


template<class _ItemType>
class MonopolyAllocator {
public:
	class MomopolyData
	{
	public:
		std::shared_ptr<_ItemType>& data() { return data_; }
		void release() {
			manager_->release_one(this);
		}
	private:
		MomopolyData(MonopolyAllocator* p_manager) {manager_ = p_manager};
		friend class MonopolyAllocator;
		MonopolyAllocator* manager_ = nullptr;
		std::shared_ptr<_ItemType> data_;
		bool available_ = true;
	};

	typedef std::shared_ptr<MomopolyData> MonopolyDataPointer;

	MonopolyAllocator(int size) {
		capacity_ = size;
		num_available_ = size;
		datas_.resize(size);
		for (int i = 0; i < size; i++) {
			datas_[i] = std::shared_ptr<MomopolyData>(new MomopolyData(this));
		}
	}

	virtual ~MonopolyAllocator() {
		run_ = false;
		cv_.notify_all();
		std::unique_lock<std::mutex> l(lock_);
		cv_exit_.wait(l, [&]() {return num_wait_thread_ == 0;});
	}

	/*
	*/
	MonopolyDataPointer query(int time_out = 10000) {
		std::unique_lock<std::mutex> l(lock_);
		if (!run_) {
			return nullptr;
		}
		if (num_available_ == 0) {
			num_wait_thread_++;
			auto state = cv_.wait_for(l, std::chrono::milliseconds(time_out), [&]() {return num_available_ > 0 || !run_; });
		}
		num_wait_thread_--;
		cv_exit_.notify_one();

		if (!stat || num_available_ == 0 || !run_) {
			return nullptr;
		}

		auto item = std::find_if(datas_.begin(), datas_.end(), [](MonopolyDataPointer& item) {return item->available_});
		if (item == datas_.end()) {
			return nullptr;
		}
		(*item)->available_ = false;
		num_available_--;
		return *item;
	}

	int num_available() {
		return num_available_;
	}

	int capacity() {
		return capacity_;
	}


private:
	void release_one(MomopolyData* prq) {
		std::unique_lock<std::mutex> l(lock_);
		if (!prq->available_) {
			prq->available_ = true;
			num_available_++;
			cv_.notify_one();
		}
	}

	std::mutex lock_;
	std::condition_variable cv_;
	std::condition_variable cv_exit_;
	std::vector<MonopolyDataPointer> datas_;
	int capacity_ = 0;
	volatile int num_available_ = 0;
	volatile int num_wait_thread_ = 0;
	volatile bool run_ = true;
};
