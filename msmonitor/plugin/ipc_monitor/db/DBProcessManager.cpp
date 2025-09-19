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

#include "db/DBProcessManager.h"
#include "db/DBConstant.h"
#include "singleton.h"

namespace dynolog_npu {
namespace ipc_monitor {
namespace db {
namespace {
constexpr uint64_t MSTX_CONNECTION_ID_OFFSET = 4000000000ULL;
const std::string MSTX_TASK_TYPE = "MsTx";
const std::string NA = "N/A";
const std::string UNKNOWN = "UNKNOWN";
const std::vector<std::tuple<uint16_t, std::string>> HCCL_DATA_TYPE = {
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

constexpr uint16_t MSTX_MARKER_TYPE = 0;
constexpr uint16_t MSTX_RANGE_TYPE = 2;
const std::vector<std::tuple<uint16_t, std::string>> MSTX_EVENT_TYPE = {
    {0, "marker"},
    {1, "push/pop"},
    {2, "start/end"},
    {3, "marker_ex"}
};

const std::vector<std::tuple<std::string, std::string>> META_DATA = {
    {"SCHEMA_VERSION_MICRO", "1"},
    {"SCHEMA_VERSION_MINOR", "2"},
    {"SCHEMA_VERSION_MAJOR", "0"},
    {"SCHEMA_VERSION", "1.2.0"}
};

constexpr uint16_t API_NODE_TYPE = 10000;
const std::vector<std::tuple<uint16_t, std::string>> API_TYPE = {
    {5000, "runtime"},
    {5500, "hccl"},
    {10000, "node"},
    {15000, "model"},
    {20000, "acl"},
    {50001, "op"},
    {50002, "queue"},
    {50003, "trace"},
    {50004, "mstx"}
};

uint64_t ConcatGlobalTid(uint32_t pid, uint32_t tid)
{
    constexpr uint32_t INT32_BIT_COUNT = 32;
    return (static_cast<uint64_t>(pid) << INT32_BIT_COUNT) | tid;
}

std::string GetMsmonitorDbPath(const std::string &outputPath)
{
    auto identity = join({std::to_string(GetProcessId()), getCurrentTimestamp(), std::to_string(GetRankId())}, "_");
    return outputPath + "/msmonitor_" + identity + ".db";
}
} // namecpace

class IdPool : public Singleton<IdPool> {
public:
    IdPool() = default;
    ~IdPool() = default;
    uint64_t GetUint64Id(const std::string &key);
    StringIdFormat GetStringIdData();
    void Clear();

private:
    std::mutex uint64IdMapMutex_;
    uint64_t uint64Index_{0};
    std::unordered_map<std::string, uint64_t> uint64IdMap_;
};

uint64_t IdPool::GetUint64Id(const std::string &key)
{
    std::lock_guard<std::mutex> lock(uint64IdMapMutex_);
    auto it = uint64IdMap_.find(key);
    if (it != uint64IdMap_.end()) {
        return it->second;
    }
    uint64IdMap_.emplace(key, uint64Index_);
    return uint64Index_++;
}

StringIdFormat IdPool::GetStringIdData()
{
    std::lock_guard<std::mutex> lock(uint64IdMapMutex_);
    StringIdFormat stringIdData;
    stringIdData.reserve(uint64IdMap_.size());
    for (auto it : uint64IdMap_) {
        stringIdData.emplace_back(it.second, it.first);
    }
    return stringIdData;
}

void IdPool::Clear()
{
    std::lock_guard<std::mutex> lock(uint64IdMapMutex_);
    uint64IdMap_.clear();
    uint64Index_ = 0;
}

void DBProcessManager::SetReportInterval(uint32_t interval)
{
    if (reportInterval_.load() != interval) {
        LOG(INFO) << "DBProcessManager SetReportInterval interval: " << interval;
        if (IsRunning()) {
            SaveData();
        }
        SetInterval(interval);
        reportInterval_.store(interval);
    }
}

void DBProcessManager::RunPreTask()
{
    sessionStartTime_ = getCurrentTimestamp64();
}

void DBProcessManager::ExecuteTask()
{
    if (!SaveData()) {
        LOG(ERROR) << "DBProcessManager SaveData failed";
    }
}

bool DBProcessManager::CheckAndInitDB()
{
    std::lock_guard<std::mutex> lock(dbMutex_);
    if (msMonitorDB_.database == nullptr || msMonitorDB_.dbRunner == nullptr) {
        std::shared_ptr<MsMonitorDB> msMonitorDB{nullptr};
        MakeSharedPtr(msMonitorDB);
        msMonitorDB_.database = msMonitorDB;
        auto dbPath = GetMsmonitorDbPath(savePath_);
        LOG(INFO) << "msMonitor db will be save to " << dbPath;
        return msMonitorDB_.database != nullptr && msMonitorDB_.ConstructDBRunner(dbPath);
    }
    return true;
}

bool DBProcessManager::SaveData()
{
    if (!CheckAndInitDB()) {
        LOG(ERROR) << "DBProcessManager init msmonitor db failed";
        return false;
    }

    bool flag = true;
    APIFormat apiData;
    CommunicationOpFormat communicationOpData;
    TaskFormat taskData;
    ComputeTaskInfoFormat computeTaskInfoData;
    MstxFormat mstxData;

    {
        std::lock_guard<std::mutex> lock(dataMutex_);
        apiData = std::move(apiData_);
        communicationOpData = std::move(communicationOpData_);
        taskData = std::move(taskData_);
        computeTaskInfoData = std::move(computeTaskInfoData_);
        mstxData = std::move(mstxData_);
    }

    flag = (apiData.empty() || SaveIncDataToDB(apiData, TABLE_CANN_API)) && flag;
    flag = (communicationOpData.empty() || SaveIncDataToDB(communicationOpData, TABLE_COMMUNICATION_OP)) && flag;
    flag = (taskData.empty() || SaveIncDataToDB(taskData, TABLE_TASK)) && flag;
    flag = (computeTaskInfoData.empty() || SaveIncDataToDB(computeTaskInfoData, TABLE_COMPUTE_TASK_INFO)) && flag;
    flag = (mstxData.empty() || SaveIncDataToDB(mstxData, TABLE_MSTX)) && flag;

    return flag;
}

bool DBProcessManager::SaveConstantData()
{
    bool flag = true;
    flag = InsertDataToDB(HCCL_DATA_TYPE, TABLE_HCCL_DATA_TYPE, msMonitorDB_) && flag;
    flag = InsertDataToDB(MSTX_EVENT_TYPE, TABLE_MSTX_EVENT_TYPE, msMonitorDB_) && flag;
    flag = InsertDataToDB(API_TYPE, TABLE_API_TYPE, msMonitorDB_) && flag;
    flag = InsertDataToDB(META_DATA, TABLE_META_DATA, msMonitorDB_) && flag;

    std::vector<std::tuple<std::string, std::string>> hostInfoData {{GetHostUid(), GetHostName()}};
    flag = InsertDataToDB(hostInfoData, TABLE_HOST_INFO, msMonitorDB_) && flag;

    std::vector<std::tuple<uint64_t, uint64_t>> sessionTimeInfoData {{sessionStartTime_, getCurrentTimestamp64()}};
    flag = InsertDataToDB(sessionTimeInfoData, TABLE_SESSION_TIME_INFO, msMonitorDB_) && flag;

    auto stringIdData = IdPool::GetInstance()->GetStringIdData();
    flag = (stringIdData.empty() || InsertDataToDB(stringIdData, TABLE_STRING_IDS, msMonitorDB_)) && flag;
    return flag;
}

bool DBProcessManager::SaveParallelGroupData()
{
    const std::string parallel_group_info_key = "parallel_group_info";
    auto iter = clusterConfigData.find(parallel_group_info_key);
    if (iter == clusterConfigData.end()) {
        LOG(WARNING) << "DBProcessManager SaveParallelGroupData parallel_group_info is not found";
        return true;
    }
    const std::string& parallel_group_info = iter->second;
    if (!parallel_group_info.empty()) {
        std::vector<std::tuple<std::string, std::string>> data {{parallel_group_info_key, parallel_group_info}};
        return InsertDataToDB(data, TABLE_META_DATA, msMonitorDB_);
    }
    return true;
}

bool DBProcessManager::SaveRankDeviceData()
{
    if (msMonitorDB_.dbRunner->CheckTableExists(TABLE_RANK_DEVICE_MAP)) {
        return true;
    }
    if (deviceSet_.empty()) {
        return false;
    }
    auto rankId = GetRankId();
    std::vector<std::tuple<int32_t, uint32_t>> rankDeviceData;
    rankDeviceData.reserve(deviceSet_.size());
    for (auto deviceId : deviceSet_) {
        rankDeviceData.emplace_back(rankId, deviceId);
    }
    if (!InsertDataToDB(rankDeviceData, TABLE_RANK_DEVICE_MAP, msMonitorDB_)) {
        LOG(ERROR) << "DBProcessManager insert rank device map data failed";
        return false;
    }
    return true;
}

bool DBProcessManager::SaveNpuInfoData()
{
    if (msMonitorDB_.dbRunner->CheckTableExists(TABLE_NPU_INFO)) {
        return true;
    }
    if (deviceSet_.empty()) {
        return false;
    }
    std::vector<std::tuple<uint32_t, std::string>> npuInfoData;
    npuInfoData.reserve(deviceSet_.size());
    for (auto deviceId : deviceSet_) {
        npuInfoData.emplace_back(deviceId, UNKNOWN);
    }
    if (!InsertDataToDB(npuInfoData, TABLE_NPU_INFO, msMonitorDB_)) {
        LOG(ERROR) << "DBProcessManager insert npu info data failed";
        return false;
    }
    return true;
}

void DBProcessManager::RunPostTask()
{
    SaveData();

    std::lock_guard<std::mutex> lock(dataMutex_);
    if (hasSavedData_) {
        if (CheckAndInitDB()) {
            SaveConstantData();
            SaveParallelGroupData();
            SaveRankDeviceData();
            SaveNpuInfoData();
        } else {
            LOG(ERROR) << "DBProcessManager init msmonitor db failed";
        }
    }
    sessionStartTime_ = 0;
    hasSavedData_ = false;
    reportInterval_.store(0);
    deviceSet_.clear();
    apiData_.clear();
    computeTaskInfoData_.clear();
    communicationOpData_.clear();
    taskData_.clear();
    mstxData_.clear();
    mstxRangeHostDataMap_.clear();
    mstxRangeDeviceDataMap_.clear();
    savePath_.clear();
    msMonitorDB_.database = nullptr;
    msMonitorDB_.dbRunner = nullptr;
    IdPool::GetInstance()->Clear();
}

void DBProcessManager::ProcessApiData(msptiActivityApi *record)
{
    uint64_t endTime = record->end;
    if (endTime < sessionStartTime_) {
        return;
    }
    std::lock_guard<std::mutex> lock(dataMutex_);
    uint64_t name = IdPool::GetInstance()->GetUint64Id(record->name);
    uint64_t globalTid = ConcatGlobalTid(record->pt.processId, record->pt.threadId);
    uint64_t connectionId = record->correlationId;
    apiData_.emplace_back(static_cast<uint64_t>(record->start), endTime, API_NODE_TYPE, globalTid, connectionId, name);
}

std::string DBProcessManager::ConstructCommOpName(const std::string &opName, const std::string &groupName)
{
    uint64_t opCount = communicationGroupOpCount_[groupName]++;
    std::string groupId;
    auto it = communicationGroupNameMap_.find(groupName);
    if (it == communicationGroupNameMap_.end()) {
        static const size_t GROUP_ID_LEN = 3;
        auto groupHashId = std::to_string(CalcHashId(groupName));
        if (groupHashId.size() >= GROUP_ID_LEN) {
            groupHashId = groupHashId.substr(groupHashId.size()-GROUP_ID_LEN);
        }
        communicationGroupNameMap_.emplace(groupName, groupHashId);
        groupId = groupHashId;
    } else {
        groupId = it->second;
    }
    return opName + "_" + groupId + "_" + std::to_string(opCount) + "_1";
}

void DBProcessManager::ProcessCommunicationData(msptiActivityCommunication *record)
{
    uint64_t endTime = record->end;
    if (endTime < sessionStartTime_) {
        return;
    }
    std::lock_guard<std::mutex> lock(dataMutex_);
    uint64_t groupName = IdPool::GetInstance()->GetUint64Id(record->commName);
    auto commOpName = ConstructCommOpName(record->name, record->commName);
    uint64_t opName = IdPool::GetInstance()->GetUint64Id(commOpName);
    uint32_t opId = communicationOpId_.fetch_add(1);
    uint64_t algType = IdPool::GetInstance()->GetUint64Id(record->algType);
    uint64_t opType = IdPool::GetInstance()->GetUint64Id(record->name);
    uint64_t connectionId = record->correlationId;
    uint32_t deviceId = record->ds.deviceId;
    communicationOpData_.emplace_back(opName, static_cast<uint64_t>(record->start), endTime,
        connectionId, groupName, opId, 0, 0, static_cast<uint16_t>(record->dataType),
        algType, static_cast<uint64_t>(record->count), opType, deviceId);
    deviceSet_.insert(deviceId);
}

void DBProcessManager::ProcessKernelData(msptiActivityKernel *record)
{
    uint64_t endTime = record->end;
    if (endTime < sessionStartTime_) {
        return;
    }
    std::lock_guard<std::mutex> lock(dataMutex_);
    uint64_t opName = IdPool::GetInstance()->GetUint64Id(record->name);
    uint64_t taskType = IdPool::GetInstance()->GetUint64Id(record->type);
    uint64_t globalTaskId = globalTaskId_.fetch_add(1);
    uint64_t NAId = IdPool::GetInstance()->GetUint64Id(NA);
    computeTaskInfoData_.emplace_back(opName, globalTaskId, UINT32_MAX, UINT32_MAX, taskType,
        NAId, NAId, NAId, NAId, NAId, NAId, NAId, NAId, NAId, NAId);
    uint64_t connectionId = record->correlationId;
    uint32_t deviceId = record->ds.deviceId;
    taskData_.emplace_back(static_cast<uint64_t>(record->start), endTime,
        deviceId, connectionId, globalTaskId, GetProcessId(), taskType, UINT32_MAX,
        static_cast<uint32_t>(record->ds.streamId), UINT32_MAX, UINT32_MAX);
    deviceSet_.insert(deviceId);
}

void DBProcessManager::ProcessMstxData(msptiActivityMarker *record)
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

void DBProcessManager::ProcessMstxHostData(msptiActivityMarker *record)
{
    uint64_t connectionId = record->id + MSTX_CONNECTION_ID_OFFSET;
    uint64_t timestamp = static_cast<uint64_t>(record->timestamp);
    uint64_t message = IdPool::GetInstance()->GetUint64Id(record->name);
    uint64_t domain = IdPool::GetInstance()->GetUint64Id(record->domain);
    uint64_t globalTid = ConcatGlobalTid(record->objectId.pt.processId, record->objectId.pt.threadId);
    if (record->flag == msptiActivityFlag::MSPTI_ACTIVITY_FLAG_MARKER_INSTANTANEOUS ||
        record->flag == msptiActivityFlag::MSPTI_ACTIVITY_FLAG_MARKER_INSTANTANEOUS_WITH_DEVICE) {
        mstxData_.emplace_back(timestamp, timestamp, MSTX_MARKER_TYPE, UINT32_MAX, UINT32_MAX,
            message, globalTid, globalTid, domain, connectionId);
    } else if (record->flag == msptiActivityFlag::MSPTI_ACTIVITY_FLAG_MARKER_START ||
        record->flag == msptiActivityFlag::MSPTI_ACTIVITY_FLAG_MARKER_START_WITH_DEVICE) {
        mstxRangeHostDataMap_.emplace(connectionId, MstxHostData{connectionId, timestamp, globalTid, domain, message});
    } else if (record->flag == msptiActivityFlag::MSPTI_ACTIVITY_FLAG_MARKER_END ||
        record->flag == msptiActivityFlag::MSPTI_ACTIVITY_FLAG_MARKER_END_WITH_DEVICE) {
        auto it = mstxRangeHostDataMap_.find(connectionId);
        if (it != mstxRangeHostDataMap_.end()) {
            mstxData_.emplace_back(it->second.timestamp, timestamp, MSTX_RANGE_TYPE, UINT32_MAX, UINT32_MAX,
                it->second.message, it->second.globalTid, globalTid, it->second.domain, connectionId);
            mstxRangeHostDataMap_.erase(it);
        }
    }
}

void DBProcessManager::ProcessMstxDeviceData(msptiActivityMarker *record)
{
    uint64_t connectionId = record->id + MSTX_CONNECTION_ID_OFFSET;
    uint64_t timestamp = static_cast<uint64_t>(record->timestamp);
    uint64_t taskType = IdPool::GetInstance()->GetUint64Id(MSTX_TASK_TYPE);
    if (record->flag == msptiActivityFlag::MSPTI_ACTIVITY_FLAG_MARKER_INSTANTANEOUS_WITH_DEVICE) {
        taskData_.emplace_back(timestamp, timestamp,
            static_cast<uint32_t>(record->objectId.ds.deviceId), connectionId,
            globalTaskId_.fetch_add(1), GetProcessId(), taskType, UINT32_MAX,
            static_cast<uint32_t>(record->objectId.ds.streamId), UINT32_MAX, UINT32_MAX);
    } else if (record->flag == msptiActivityFlag::MSPTI_ACTIVITY_FLAG_MARKER_START_WITH_DEVICE) {
        mstxRangeDeviceDataMap_.emplace(connectionId,
            MstxDeviceData{connectionId, timestamp, globalTaskId_.fetch_add(1)});
    } else if (record->flag == msptiActivityFlag::MSPTI_ACTIVITY_FLAG_MARKER_END_WITH_DEVICE) {
        auto it = mstxRangeDeviceDataMap_.find(connectionId);
        if (it != mstxRangeDeviceDataMap_.end()) {
            uint32_t deviceId = static_cast<uint32_t>(record->objectId.ds.deviceId);
            taskData_.emplace_back(it->second.timestamp, timestamp,
                deviceId, connectionId, it->second.globalTaskId, GetProcessId(), taskType,
                UINT32_MAX, static_cast<uint32_t>(record->objectId.ds.streamId), UINT32_MAX, UINT32_MAX);
            mstxRangeDeviceDataMap_.erase(it);
            deviceSet_.insert(deviceId);
        }
    }
}

ErrCode DBProcessManager::ConsumeMsptiData(msptiActivity *record)
{
    if (record == nullptr) {
        LOG(ERROR) << "DBProcessManager::ConsumeMsptiData record is null";
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
            LOG(WARNING) << record->kind << " is not supported for DBProcessManager";
            break;
    }
    return ErrCode::SUC;
}
} // namespace db
} // namespace ipc_monitor
} // namespace dynolog_npu
