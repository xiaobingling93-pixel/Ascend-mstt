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
#include "MsptiMonitor.h"

#include <unistd.h>
#include <unordered_map>
#include <nlohmann/json.hpp>
#include <glog/logging.h>

#include "DynoLogNpuMonitor.h"
#include "MetricManager.h"
#include "utils.h"

namespace {
constexpr size_t DEFAULT_BUFFER_SIZE = 8 * 1024 * 1024;
constexpr size_t MAX_BUFFER_SIZE = 256 * 1024 * 1024;
constexpr uint32_t MAX_ALLOC_CNT = MAX_BUFFER_SIZE / DEFAULT_BUFFER_SIZE;
}

namespace dynolog_npu {
namespace ipc_monitor {

MsptiMonitor::MsptiMonitor()
    : start_(false),
      subscriber_(nullptr),
      checkFlush_(false),
      flushInterval_(0) {}

MsptiMonitor::~MsptiMonitor()
{
    Uninit();
}

void MsptiMonitor::Start()
{
    if (start_.load()) {
        return;
    }
    SetThreadName("MsptiMonitor");
    if (Thread::Start() != 0) {
        LOG(ERROR) << "MsptiMonitor start failed";
        return;
    }
    start_.store(true);
    metric::MetricManager::GetInstance()->Run();
    LOG(INFO) << "MsptiMonitor start successfully";
}

void MsptiMonitor::Stop()
{
    if (!start_.load()) {
        LOG(WARNING) << "MsptiMonitor is not running";
        return;
    }
    Uninit();
    if (msptiActivityFlushAll(1) != MSPTI_SUCCESS) {
        LOG(WARNING) << "MsptiMonitor stop msptiActivityFlushAll failed";
    }
    LOG(INFO) << "MsptiMonitor stop successfully";
}

void MsptiMonitor::Uninit()
{
    if (!start_.load()) {
        return;
    }
    metric::MetricManager::GetInstance()->Stop();
    start_.store(false);
    cv_.notify_one();
    Thread::Stop();
}

void MsptiMonitor::EnableActivity(msptiActivityKind kind)
{
    if (MSPTI_ACTIVITY_KIND_INVALID < kind && kind < MSPTI_ACTIVITY_KIND_COUNT) {
        std::lock_guard<std::mutex> lock(activityMtx_);
        if (msptiActivityEnable(kind) == MSPTI_SUCCESS) {
            enabledActivities_.insert(kind);
        } else {
            LOG(ERROR) << "MsptiMonitor enableActivity failed, kind: " << static_cast<int32_t>(kind);
        }
        metric::MetricManager::GetInstance()->EnableKindSwitch_(kind, true);
    }
}

void MsptiMonitor::DisableActivity(msptiActivityKind kind)
{
    if (MSPTI_ACTIVITY_KIND_INVALID < kind && kind < MSPTI_ACTIVITY_KIND_COUNT) {
        std::lock_guard<std::mutex> lock(activityMtx_);
        if (msptiActivityDisable(kind) == MSPTI_SUCCESS) {
            enabledActivities_.erase(kind);
        } else {
            LOG(ERROR) << "MsptiMonitor disableActivity failed, kind: " << static_cast<int32_t>(kind);
        }
        metric::MetricManager::GetInstance()->EnableKindSwitch_(kind, false);
    }
}

void MsptiMonitor::SetFlushInterval(uint32_t interval)
{
    flushInterval_.store(interval);
    checkFlush_.store(true);
    if (start_.load()) {
        cv_.notify_one();
    }
    metric::MetricManager::GetInstance()->SetReportInterval(interval);
}

bool MsptiMonitor::IsStarted()
{
    return start_.load();
}

std::set<msptiActivityKind> MsptiMonitor::GetEnabledActivities()
{
    std::lock_guard<std::mutex> lock(activityMtx_);
    return enabledActivities_;
}

void MsptiMonitor::Run()
{
    if (msptiSubscribe(&subscriber_, nullptr, nullptr) != MSPTI_SUCCESS) {
        LOG(ERROR) << "MsptiMonitor run failed, msptiSubscribe failed";
        return;
    }
    if (msptiActivityRegisterCallbacks(BufferRequest, BufferComplete) != MSPTI_SUCCESS) {
        LOG(ERROR) << "MsptiMonitor run failed, msptiActivityRegisterCallbacks failed";
        return;
    }
    while (true) {
        std::unique_lock<std::mutex> lock(cvMtx_);
        if (flushInterval_.load() > 0) {
            cv_.wait_for(lock, std::chrono::seconds(flushInterval_.load()),
                         [&]() { return checkFlush_.load() || !start_.load();});
        } else {
            cv_.wait(lock, [&]() { return checkFlush_.load () || !start_.load();});
        }
        if (!start_.load()) {
            break;
        }
        if (checkFlush_.load()) {
            checkFlush_.store(false);
        }
        if (flushInterval_.load() > 0) {
            if (msptiActivityFlushAll(1) != MSPTI_SUCCESS) {
                LOG(ERROR) << "MsptiMonitor run msptiActivityFlushAll failed";
            }
        }
    }
    if (msptiUnsubscribe(subscriber_) != MSPTI_SUCCESS) {
        LOG(ERROR) << "MsptiMonitor run failed, msptiUnsubscribe failed";
    }
    {
        std::lock_guard<std::mutex> lock(activityMtx_);
        for (auto kind : enabledActivities_) {
            msptiActivityDisable(kind);
        }
        enabledActivities_.clear();
    }
    checkFlush_.store(false);
    flushInterval_.store(0);
}

std::atomic<uint32_t> MsptiMonitor::allocCnt{0};

void MsptiMonitor::BufferRequest(uint8_t **buffer, size_t *size, size_t *maxNumRecords)
{
    if (buffer == nullptr || size == nullptr || maxNumRecords == nullptr) {
        return;
    }
    *maxNumRecords = 0;
    if (allocCnt.load() >= MAX_ALLOC_CNT) {
        *buffer = nullptr;
        *size = 0;
        LOG(ERROR) << "MsptiMonitor BufferRequest failed, allocCnt: " << allocCnt.load();
        return;
    }
    uint8_t *pBuffer = ReinterpretConvert<uint8_t*>(MsptiMalloc(DEFAULT_BUFFER_SIZE, ALIGN_SIZE));
    if (pBuffer == nullptr) {
        *buffer = nullptr;
        *size = 0;
    } else {
        *buffer = pBuffer;
        *size = DEFAULT_BUFFER_SIZE;
        allocCnt++;
        LOG(INFO) << "MsptiMonitor BufferRequest, size: " << *size;
    }
}

void MsptiMonitor::BufferComplete(uint8_t *buffer, size_t size, size_t validSize)
{
    if (validSize > 0 && buffer != nullptr) {
        LOG(INFO) << "MsptiMonitor BufferComplete, size: " << size << ", validSize: " << validSize;
        msptiActivity *record = nullptr;
        msptiResult status = MSPTI_SUCCESS;
        do {
            status = msptiActivityGetNextRecord(buffer, validSize, &record);
            if (status == MSPTI_SUCCESS) {
                BufferConsume(record);
            } else if (status == MSPTI_ERROR_MAX_LIMIT_REACHED) {
                break;
            } else {
                LOG(ERROR) << "MsptiMonitor BufferComplete failed, status: " << static_cast<int32_t>(status);
                break;
            }
        } while (true);
        allocCnt--;
    }
    MsptiFree(buffer);
}

void MsptiMonitor::BufferConsume(msptiActivity *record)
{
    if (record == nullptr) {
        return;
    }
    metric::MetricManager::GetInstance()->ConsumeMsptiData(record);
}
} // namespace ipc_monitor
} // namespace dynolog_npu
