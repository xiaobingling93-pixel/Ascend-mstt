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
#ifndef IPC_MONITOR_THREAD_H
#define IPC_MONITOR_THREAD_H

#include <csignal>
#include <sys/prctl.h>
#include <pthread.h>
#include <string>
#include "utils.h"

namespace dynolog_npu {
namespace ipc_monitor {
class Thread {
public:
    Thread()
        : is_alive_(false),
          pid_(0),
          thread_name_("IPCMonitor") {}

    ~Thread()
    {
        if (is_alive_) {
            (void)pthread_cancel(pid_);
            (void)pthread_join(pid_, nullptr);
        }
    }

    void SetThreadName(const std::string &name)
    {
        if (!name.empty()) {
            thread_name_ = name;
        }
    }

    std::string GetThreadName()
    {
        return thread_name_;
    }

    int Start()
    {
        int ret = pthread_create(&pid_, nullptr, Execute, ReinterpretConvert<void*>(this));
        is_alive_ = (ret == 0) ? true : false;
        return ret;
    }

    int Stop()
    {
        return Join();
    }

    int Join()
    {
        int ret = pthread_join(pid_, nullptr);
        is_alive_ = (ret == 0) ? false : true;
        return ret;
    }

private:
    static void* Execute(void *args)
    {
        Thread *thr = ReinterpretConvert<Thread*>(args);
        prctl(PR_SET_NAME, ReinterpretConvert<unsigned long>(thr->GetThreadName().data()));
        thr->Run();
        return nullptr;
    }
    virtual void Run() = 0;

private:
    bool is_alive_;
    pthread_t pid_;
    std::string thread_name_;
};
} // ipc_monitor
} // dynolog_npu
#endif // IPC_MONITOR_THREAD_H
