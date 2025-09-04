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
#ifndef MSPTI_MONITOR_H
#define MSPTI_MONITOR_H

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <set>
#include "mspti.h"
#include "thread.h"
#include "singleton.h"
#include "MsptiDataProcessBase.h"


namespace dynolog_npu {
namespace ipc_monitor {
class MsptiMonitor : public Singleton<MsptiMonitor>, public Thread {
public:
    virtual ~MsptiMonitor();
    void Start();
    void Stop();
    void EnableActivity(msptiActivityKind kind);
    void DisableActivity(msptiActivityKind kind);
    void SetFlushInterval(uint32_t interval);
    bool IsStarted();
    std::set<msptiActivityKind> GetEnabledActivities();
    void Uninit();
    bool CheckAndSetSavePath(const std::string& path);
    void SetClusterConfigData(const std::unordered_map<std::string, std::string>& configData);

private:
    static void BufferRequest(uint8_t **buffer, size_t *size, size_t *maxNumRecords);
    static void BufferComplete(uint8_t *buffer, size_t size, size_t validSize);
    static void BufferConsume(msptiActivity *record);
    static std::shared_ptr<MsptiDataProcessBase> GetDataProcessor();
    static std::atomic<uint32_t> allocCnt;

private:
    void Run() override;

private:
    std::atomic<bool> start_{false};
    std::mutex cvMtx_;
    std::condition_variable cv_;
    msptiSubscriberHandle subscriber_{nullptr};
    std::mutex activityMtx_;
    std::set<msptiActivityKind> enabledActivities_;
    std::atomic<bool> checkFlush_{false};
    std::atomic<uint32_t> flushInterval_{0};
    std::string savePath_;
    std::shared_ptr<MsptiDataProcessBase> dataProcessor_{nullptr};
};
} // namespace ipc_monitor
} // namespace dynolog_npu
#endif // MSPTI_MONITOR_H
