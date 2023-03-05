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
};

}

#endif  // __CPM_HPP__
