/*
 * Copyright (C) 2025-2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef TIMER_TASK_H
#define TIMER_TASK_H

#include <thread>
#include <chrono>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <glog/logging.h>

namespace dynolog_npu {
namespace ipc_monitor {
class TimerTask {
public:
    TimerTask(const std::string& name, int interval)
        : interval(interval), name(name), manual_trigger(false), running(false) {}

    ~TimerTask()
    {
        Stop();
    }

    void Run()
    {
        if (running) {
            LOG(ERROR) << name << " Timer task is already running.";
            return;
        }
        running = true;
        taskThread = std::thread(&TimerTask::TaskRun, this);
    }

    void Trigger()
    {
        std::unique_lock<std::mutex> lock(cv_mutex);
        manual_trigger = true;
        if (running.load()) {
            cv.notify_one();
        }
    }

    // 停止定时任务
    void Stop()
    {
        if (!running) {
            LOG(ERROR) << name << "Timer task is not running.";
            return;
        }

        running = false;
        cv.notify_one();
        if (taskThread.joinable()) {
            taskThread.join();
        }
    }

    void SetInterval(int intervalTimes)
    {
        interval.store(intervalTimes);
    }

    virtual void InitResource() {};
    virtual void ReleaseResource() {};
    virtual void ExecuteTask() = 0;
private:
    // 定时任务线程函数
    void TaskRun()
    {
        LOG(INFO) << name << " Timer task started.";
        InitResource();
        while (running) {
            std::unique_lock<std::mutex> lock(cv_mutex);
            if (interval.load()) {
                cv.wait_for(lock, std::chrono::seconds(interval.load()), [&] {return manual_trigger || !running;});
            } else {
                cv.wait(lock, [&] {return manual_trigger || !running;});
            }
            if (!running) {
                break;
            }
            if (manual_trigger) {
                manual_trigger = false;
            }
            if (running) {
                ExecuteTask();
            }
        }
        ReleaseResource();
        LOG(INFO) << name << " Timer task stopped.";
    }

    std::atomic<int> interval;
    std::string name;
    std::condition_variable cv;
    std::mutex cv_mutex;
    std::atomic<bool> manual_trigger;
    std::atomic<bool> running;
    std::thread taskThread;
};
    
}
}
#endif