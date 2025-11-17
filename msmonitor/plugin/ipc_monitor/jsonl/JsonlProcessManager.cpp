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

#include "jsonl/JsonlProcessManager.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include "singleton.h"
#include "MsptiMonitor.h"

namespace dynolog_npu {
namespace ipc_monitor {
namespace jsonl {
namespace {
std::string GetCommunicationDataTypeName(msptiCommunicationDataType dataType)
{
    static const std::unordered_map<msptiCommunicationDataType, std::string> DATA_TYPE = {
        {msptiCommunicationDataType::MSPTI_ACTIVITY_COMMUNICATION_INT8, "INT8"},
        {msptiCommunicationDataType::MSPTI_ACTIVITY_COMMUNICATION_INT16, "INT16"},
        {msptiCommunicationDataType::MSPTI_ACTIVITY_COMMUNICATION_INT32, "INT32"},
        {msptiCommunicationDataType::MSPTI_ACTIVITY_COMMUNICATION_INT64, "INT64"},
        {msptiCommunicationDataType::MSPTI_ACTIVITY_COMMUNICATION_UINT8, "UINT8"},
        {msptiCommunicationDataType::MSPTI_ACTIVITY_COMMUNICATION_UINT16, "UINT16"},
        {msptiCommunicationDataType::MSPTI_ACTIVITY_COMMUNICATION_UINT32, "UINT32"},
        {msptiCommunicationDataType::MSPTI_ACTIVITY_COMMUNICATION_UINT64, "UINT64"},
        {msptiCommunicationDataType::MSPTI_ACTIVITY_COMMUNICATION_FP16, "FP16"},
        {msptiCommunicationDataType::MSPTI_ACTIVITY_COMMUNICATION_FP32, "FP32"},
        {msptiCommunicationDataType::MSPTI_ACTIVITY_COMMUNICATION_FP64, "FP64"},
        {msptiCommunicationDataType::MSPTI_ACTIVITY_COMMUNICATION_BFP16, "BFP16"},
        {msptiCommunicationDataType::MSPTI_ACTIVITY_COMMUNICATION_INT128, "INT128"},
        {msptiCommunicationDataType::MSPTI_ACTIVITY_COMMUNICATION_INVALID_TYPE, "INVALID_TYPE"}
    };
    auto it = DATA_TYPE.find(dataType);
    return it != DATA_TYPE.end() ? it->second : "INVALID_TYPE";
}

uint32_t GetRingBufferCapacity()
{
    constexpr uint32_t DEFAULT_CAPACITY = 1024 * 512;
    constexpr uint32_t MIN_CAPACITY = 1024 * 8;
    constexpr uint32_t MAX_CAPACITY = 1024 * 1024 * 2;
    const char* capacityEnvVal = std::getenv("MSMONITOR_JSONL_BUFFER_CAPACITY");
    std::string capacityStr = (capacityEnvVal != nullptr ? capacityEnvVal : "");
    uint32_t capacity = DEFAULT_CAPACITY;
    if (!capacityStr.empty()) {
        if (Str2Uint32(capacity, capacityStr)) {
            capacity = std::clamp(capacity, MIN_CAPACITY, MAX_CAPACITY);
            return capacity;
        } else {
            LOG(WARNING) << "Jsonl GetRingBufferCapacity invalid capacity: " << capacityStr
                         << ", use default capacity: " << DEFAULT_CAPACITY;
        }
    }
    return DEFAULT_CAPACITY;
}

uint32_t GetDataDumpMaxInterval()
{
    constexpr uint32_t DEFAULT_INTERVAL = 30000; // 30s
    constexpr uint32_t MIN_INTERVAL = 1000; // 1s
    const char* intervalEnvVal = std::getenv("MSMONITOR_JSONL_MAX_DUMP_INTERVAL");
    std::string intervalStr = (intervalEnvVal != nullptr ? intervalEnvVal : "");
    uint32_t interval = DEFAULT_INTERVAL;
    if (!intervalStr.empty()) {
        if (Str2Uint32(interval, intervalStr)) {
            interval = std::max(interval, MIN_INTERVAL);
            return interval;
        } else {
            LOG(WARNING) << "Jsonl GetDataDumpMaxInterval invalid interval: " << intervalStr
                         << ", use default interval: " << DEFAULT_INTERVAL;
        }
    }
    return DEFAULT_INTERVAL;
}
} // namecpace

void JsonlProcessManager::SetReportInterval(uint32_t interval)
{
    if (reportInterval_.load() != interval) {
        LOG(INFO) << "JsonlProcessManager SetReportInterval interval: " << interval;
        if (IsRunning()) {
            SaveData();
        }
        SetInterval(interval);
        reportInterval_.store(interval);
    }
}

void JsonlProcessManager::RunPreTask()
{
    sessionStartTime_ = getCurrentTimestamp64();
    LOG(INFO) << "JsonlProcessManager data will be save to: " << savePath_;
    dataDumper_.Init(savePath_, GetRingBufferCapacity(), GetDataDumpMaxInterval());
    dataDumper_.Start();
}

void JsonlProcessManager::ExecuteTask()
{
    if (!SaveData()) {
        LOG(ERROR) << "JsonlProcessManager SaveData failed";
    }
}

void JsonlProcessManager::RunPostTask()
{
    SaveData();

    std::lock_guard<std::mutex> lock(dataMutex_);
    SaveParallelGroupData();
    SaveRankDeviceData();
    sessionStartTime_ = 0;
    reportInterval_.store(0);
    deviceSet_.clear();
    mstxRangeHostData_.clear();
    mstxRangeDeviceData_.clear();
    savePath_.clear();
    dataDumper_.Stop();
    dataDumper_.UnInit();
    LOG(INFO) << "JsonlProcessManager finish";
}

bool JsonlProcessManager::SaveData()
{
    LOG(INFO) << "JsonlProcessManager SaveData";
    return true;
}

bool JsonlProcessManager::SaveParallelGroupData()
{
    const std::string parallel_group_info_key = "parallel_group_info";
    auto clusterConfigData = MsptiMonitor::GetInstance()->GetClusterConfigData();
    auto iter = clusterConfigData.find(parallel_group_info_key);
    if (iter == clusterConfigData.end()) {
        LOG(WARNING) << "JsonlProcessManager SaveParallelGroupData parallel_group_info is not found";
        return true;
    }
    nlohmann::json json = {
        {"kind", parallel_group_info_key},
        {"value", iter->second}
    };
    dataDumper_.Record(std::make_unique<nlohmann::json>(json));
    return true;
}

bool JsonlProcessManager::SaveRankDeviceData()
{
    if (deviceSet_.empty()) {
        return false;
    }
    nlohmann::json json = {
        {"kind", "rank_device_map"},
        {"rank", GetRankId()},
        {"device", std::vector<uint32_t>(deviceSet_.begin(), deviceSet_.end())}
    };
    dataDumper_.Record(std::make_unique<nlohmann::json>(json));
    return true;
}

void JsonlProcessManager::ProcessApiData(msptiActivityApi *record)
{
    uint64_t endTime = record->end;
    if (endTime < sessionStartTime_) {
        return;
    }
    std::lock_guard<std::mutex> lock(dataMutex_);
    nlohmann::json json = {
        {"kind", "API"},
        {"name", std::string(record->name)},
        {"startNs", static_cast<uint64_t>(record->start)},
        {"endNs", endTime},
        {"processId", static_cast<uint32_t>(record->pt.processId)},
        {"threadId", static_cast<uint32_t>(record->pt.threadId)},
        {"correlationId", static_cast<uint64_t>(record->correlationId)}
    };
    dataDumper_.Record(std::make_unique<nlohmann::json>(json));
}

void JsonlProcessManager::ProcessCommunicationData(msptiActivityCommunication *record)
{
    uint64_t endTime = record->end;
    if (endTime < sessionStartTime_) {
        return;
    }
    std::lock_guard<std::mutex> lock(dataMutex_);
    uint32_t deviceId = record->ds.deviceId;
    nlohmann::json json = {
        {"kind", "Communication"},
        {"name", std::string(record->name)},
        {"startNs", static_cast<uint64_t>(record->start)},
        {"endNs", endTime},
        {"deviceId", deviceId},
        {"streamId", static_cast<uint32_t>(record->ds.streamId)},
        {"dataType", GetCommunicationDataTypeName(record->dataType)},
        {"count", static_cast<uint64_t>(record->count)},
        {"commName", std::string(record->commName)},
        {"algType", std::string(record->algType)},
        {"correlationId", static_cast<uint64_t>(record->correlationId)}
    };
    dataDumper_.Record(std::make_unique<nlohmann::json>(json));
    deviceSet_.insert(deviceId);
}

void JsonlProcessManager::ProcessKernelData(msptiActivityKernel *record)
{
    uint64_t endTime = record->end;
    if (endTime < sessionStartTime_) {
        return;
    }
    std::lock_guard<std::mutex> lock(dataMutex_);
    uint32_t deviceId = record->ds.deviceId;
    nlohmann::json json = {
        {"kind", "Kernel"},
        {"name", std::string(record->name)},
        {"startNs", static_cast<uint64_t>(record->start)},
        {"endNs", endTime},
        {"deviceId", deviceId},
        {"streamId", static_cast<uint32_t>(record->ds.streamId)},
        {"type", std::string(record->type)},
        {"correlationId", static_cast<uint64_t>(record->correlationId)}
    };
    dataDumper_.Record(std::make_unique<nlohmann::json>(json));
    deviceSet_.insert(deviceId);
}

void JsonlProcessManager::ProcessMstxData(msptiActivityMarker *record)
{
    if (record->timestamp < sessionStartTime_) {
        return;
    }
    std::lock_guard<std::mutex> lock(dataMutex_);
    if (record->sourceKind == msptiActivitySourceKind::MSPTI_ACTIVITY_SOURCE_KIND_HOST) {
        ProcessMstxHostData(record);
    } else if (record->sourceKind == msptiActivitySourceKind::MSPTI_ACTIVITY_SOURCE_KIND_DEVICE) {
        ProcessMstxDeviceData(record);
    }
}

void JsonlProcessManager::ProcessMstxHostData(msptiActivityMarker *record)
{
    uint64_t connectionId = record->id;
    uint64_t timestamp = record->timestamp;
    std::string message = record->name;
    std::string domain = record->domain;
    if (record->flag == msptiActivityFlag::MSPTI_ACTIVITY_FLAG_MARKER_INSTANTANEOUS ||
        record->flag == msptiActivityFlag::MSPTI_ACTIVITY_FLAG_MARKER_INSTANTANEOUS_WITH_DEVICE) {
        nlohmann::json json = {
            {"kind", "Marker"},
            {"sourceKind", "Host"},
            {"name", message},
            {"startNs", timestamp},
            {"endNs", timestamp},
            {"domain", domain},
            {"processId", static_cast<uint32_t>(record->objectId.pt.processId)},
            {"threadId", static_cast<uint32_t>(record->objectId.pt.threadId)},
            {"id", connectionId}
        };
        dataDumper_.Record(std::make_unique<nlohmann::json>(json));
        if (record->flag == msptiActivityFlag::MSPTI_ACTIVITY_FLAG_MARKER_INSTANTANEOUS_WITH_DEVICE) {
            mstxMarkerHostData_.emplace(connectionId, MstxHostData{timestamp, domain, message});
        }
    } else if (record->flag == msptiActivityFlag::MSPTI_ACTIVITY_FLAG_MARKER_START ||
        record->flag == msptiActivityFlag::MSPTI_ACTIVITY_FLAG_MARKER_START_WITH_DEVICE) {
        mstxRangeHostData_.emplace(connectionId, MstxHostData{timestamp, domain, message});
    } else if (record->flag == msptiActivityFlag::MSPTI_ACTIVITY_FLAG_MARKER_END ||
        record->flag == msptiActivityFlag::MSPTI_ACTIVITY_FLAG_MARKER_END_WITH_DEVICE) {
        auto it = mstxRangeHostData_.find(connectionId);
        if (it != mstxRangeHostData_.end()) {
            nlohmann::json json = {
                {"kind", "Marker"},
                {"sourceKind", "Host"},
                {"name", it->second.message},
                {"startNs", it->second.timestamp},
                {"endNs", timestamp},
                {"domain", it->second.domain},
                {"processId", static_cast<uint32_t>(record->objectId.pt.processId)},
                {"threadId", static_cast<uint32_t>(record->objectId.pt.threadId)},
                {"id", connectionId}
            };
            dataDumper_.Record(std::make_unique<nlohmann::json>(json));
            if (record->flag == msptiActivityFlag::MSPTI_ACTIVITY_FLAG_MARKER_END) {
                mstxRangeHostData_.erase(it);
            }
        }
    }
}

void JsonlProcessManager::ProcessMstxDeviceData(msptiActivityMarker *record)
{
    uint64_t connectionId = record->id;
    uint64_t timestamp = record->timestamp;
    uint32_t deviceId = static_cast<uint32_t>(record->objectId.ds.deviceId);
    if (record->flag == msptiActivityFlag::MSPTI_ACTIVITY_FLAG_MARKER_INSTANTANEOUS_WITH_DEVICE) {
        auto it = mstxMarkerHostData_.find(connectionId);
        nlohmann::json json = {
            {"kind", "Marker"},
            {"sourceKind", "Device"},
            {"name", it != mstxMarkerHostData_.end() ? it->second.message : std::string(record->name)},
            {"startNs", timestamp},
            {"endNs", timestamp},
            {"domain", it != mstxMarkerHostData_.end() ? it->second.domain : std::string(record->domain)},
            {"deviceId", deviceId},
            {"streamId", static_cast<uint32_t>(record->objectId.ds.streamId)},
            {"id", connectionId}
        };
        dataDumper_.Record(std::make_unique<nlohmann::json>(json));
        if (it != mstxMarkerHostData_.end()) {
            mstxMarkerHostData_.erase(it);
        }
    } else if (record->flag == msptiActivityFlag::MSPTI_ACTIVITY_FLAG_MARKER_START_WITH_DEVICE) {
        mstxRangeDeviceData_.emplace(connectionId, MstxDeviceData{timestamp});
    } else if (record->flag == msptiActivityFlag::MSPTI_ACTIVITY_FLAG_MARKER_END_WITH_DEVICE) {
        auto it = mstxRangeDeviceData_.find(connectionId);
        if (it != mstxRangeDeviceData_.end()) {
            auto hostIt = mstxRangeHostData_.find(connectionId);
            nlohmann::json json = {
                {"kind", "Marker"},
                {"sourceKind", "Device"},
                {"name", hostIt != mstxRangeHostData_.end() ? hostIt->second.message : std::string(record->name)},
                {"startNs", it->second.timestamp},
                {"endNs", timestamp},
                {"domain", hostIt != mstxRangeHostData_.end() ? hostIt->second.domain : std::string(record->domain)},
                {"deviceId", deviceId},
                {"streamId", static_cast<uint32_t>(record->objectId.ds.streamId)},
                {"id", connectionId}
            };
            dataDumper_.Record(std::make_unique<nlohmann::json>(json));
            mstxRangeDeviceData_.erase(it);
            if (hostIt != mstxRangeHostData_.end()) {
                mstxRangeHostData_.erase(hostIt);
            }
        }
    }
    deviceSet_.insert(deviceId);
}

ErrCode JsonlProcessManager::ConsumeMsptiData(msptiActivity *record)
{
    if (record == nullptr) {
        LOG(ERROR) << "JsonlProcessManager::ConsumeMsptiData record is null";
        return ErrCode::VALUE;
    }
    switch (record->kind) {
        case msptiActivityKind::MSPTI_ACTIVITY_KIND_API:
            ProcessApiData(ReinterpretConvert<msptiActivityApi*>(record));
            break;
        case msptiActivityKind::MSPTI_ACTIVITY_KIND_COMMUNICATION:
            ProcessCommunicationData(ReinterpretConvert<msptiActivityCommunication*>(record));
            break;
        case msptiActivityKind::MSPTI_ACTIVITY_KIND_KERNEL:
            ProcessKernelData(ReinterpretConvert<msptiActivityKernel*>(record));
            break;
        case msptiActivityKind::MSPTI_ACTIVITY_KIND_MARKER:
            ProcessMstxData(ReinterpretConvert<msptiActivityMarker*>(record));
            break;
        default:
            LOG(WARNING) << record->kind << " is not supported for JsonlProcessManager";
            break;
    }
    return ErrCode::SUC;
}
} // namespace jsonl
} // namespace ipc_monitor
} // namespace dynolog_npu
