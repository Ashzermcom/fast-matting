#ifndef __CPM_HPP__
#define __CPM_HPP__

#include <future>
#include <memory>
#include <queue>
#include <thread>
#include <algorithm>
#include <condition_variable>

namespace cpm {
template <typename _Result, typename _Input, typename _Model>
class Instance {
protected:
    struct Item {
        /* data */
        _Input input;
        std::shared_ptr<std::promise<_Result>> prom;
    };

    std::condition_variable cond_;
    std::queue<Item> input_queue_;
    std::mutex queue_lock_;
    std::shared_ptr<std::thread> worker_;

    volatile bool run_ = false;
    volatile int max_items_processed_ = 0;
    void* stream_ = nullptr;

public:
    virtual ~Instance() { stop(); }
    void stop() {
        run_ = false;
        cond_.notify_one();
        {
            std::unique_lock<std::mutex> l(queue_lock_);
            while (!input_queue_.empty())
            {
                /* code */
                auto& item = input_queue_.front();
                if (item.prom) {
                    item.prom->set_value(_Result());
                }
                input_queue_.pop();
            }
            
        };
        if (worker_) {
            worker_->join();
            worker_.reset();
        }
    }

    virtual std::shared_future<_Result> commit(const _Input& input) {
        Item item;
        item.input = input;
        item.prom.reset(new std::promise<_Result>());
        {
            std::unique_lock<std::mutex> _lock_(queue_lock_);
            input_queue_.push(item);
        }
        cond_.notify_one();
        return item.prom->get_future();
    }

    virtual std::vector<std::shared_future<_Result>> commits(const std::vector<_Input>& inputs) {
        std::vector<std::shared_future<_Result>>& output;
        {
            std::unique_lock<std::mutex> _lock_(queue_lock_);
            for (int i=0; i<inputs.size(); ++i) {
                Item item;
                item.input = inputs[i];
                item.prom.reset(new std::promise<_Result>());
                output.emplace_back(item.prom->get_future());
                input_queue_.push(item);
            }
        }
        cond_.notify_one();
        return output;
    }

    template <typename _LoadMethod>
    bool start(const _LoadMethod& loadMethod, int max_items_processed = 1, void* stream = nullptr) {
        stop();
        this->stream_ = stream;
        this->max_items_processed_ = max_items_processed;
        std::promise<bool> status;
        worker_ = std::make_shared<std::thread>(&Instance::worker<_LoadMethod>, this, std::ref(loadMethod), std::ref(status));
        return status.get_future.get();
    }

private:

};

}

#endif  // __CPM_HPP__
