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
#ifndef IPC_MONITOR_DB_PROCESS_MANAGER_H
#define IPC_MONITOR_DB_PROCESS_MANAGER_H

#include <atomic>
#include <mutex>
#include <unordered_set>
#include "MsptiDataProcessBase.h"
#include "db/DBInfo.h"

namespace dynolog_npu {
namespace ipc_monitor {
namespace db {
// STRING_IDS: id, value
using StringIdFormat = std::vector<std::tuple<uint64_t, std::string>>;
// CANN_API: startNs, endNs, type, globalTid, connectionId, name
using APIFormat = std::vector<std::tuple<uint64_t, uint64_t, uint16_t, uint64_t, uint64_t, uint64_t>>;
// COMMUNICATION_OP: opName, startNs, endNs, connectionId, groupName,
//      opId, relay, retry, dataType, algType, count, opType, deviceId
using CommunicationOpFormat = std::vector<std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
        uint32_t, int32_t, int32_t, uint16_t, uint64_t, uint64_t, uint64_t, uint32_t>>;
// COMPUTE_TASK_INFO: name, globalTaskId, blockDim, mixBlockDim, taskType, opType, inputFormats, inputDataTypes,
//      inputShapes, outputFormats, outputDataTypes, outputShapes, attrInfo, opState, hf32Eligible
using ComputeTaskInfoFormat = std::vector<std::tuple<uint64_t, uint64_t, uint32_t, uint32_t, uint64_t, uint64_t,
        uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t>>;
// TASK: startNs, endNs, deviceId, connectionId, globalTaskId,
//      globalPid, taskType, contextId, streamId, taskId, modelId
using TaskFormat = std::vector<std::tuple<uint64_t, uint64_t, uint32_t, int64_t, uint64_t,
        uint32_t, uint64_t, uint32_t, int32_t, uint32_t, uint32_t>>;
// MSTX: startNs, endNs, eventType, rangeId, category,
//      message, globalTid, endGlobalTid, domainId, connectionId
using MstxFormat = std::vector<std::tuple<uint64_t, uint64_t, uint16_t, uint32_t, uint32_t,
        uint64_t, uint64_t, uint64_t, uint64_t, uint64_t>>;

struct MstxHostData {
    uint64_t connectionId;
    uint64_t timestamp;
    uint64_t globalTid;
    uint64_t domain;
    uint64_t message;
};

struct MstxDeviceData {
    uint64_t connectionId;
    uint64_t timestamp;
    uint64_t globalTaskId;
};

class DBProcessManager : public MsptiDataProcessBase {
public:
    DBProcessManager(std::string savePath)
        : MsptiDataProcessBase("DBProcessManager"), savePath_(std::move(savePath)) {}
    ~DBProcessManager() = default;
    ErrCode ConsumeMsptiData(msptiActivity *record) override;
    void SetReportInterval(uint32_t interval) override;
    void RunPreTask() override;
    void ExecuteTask() override;
    void RunPostTask() override;

private:
    void ProcessApiData(msptiActivityApi *record);
    void ProcessCommunicationData(msptiActivityCommunication *record);
    void ProcessKernelData(msptiActivityKernel *record);
    void ProcessMstxData(msptiActivityMarker *record);
    void ProcessMstxHostData(msptiActivityMarker *record);
    void ProcessMstxDeviceData(msptiActivityMarker *record);
    bool CheckAndInitDB();
    bool SaveData();
    bool SaveConstantData();
    bool SaveParallelGroupData();
    bool SaveRankDeviceData();
    bool SaveNpuInfoData();
    std::string ConstructCommOpName(const std::string &opName, const std::string &groupName);
    template<typename... Args>
    bool SaveIncDataToDB(const std::vector<std::tuple<Args...>> &data, const std::string &tableName);

private:
    uint64_t sessionStartTime_{0};
    std::string savePath_;
    std::mutex dbMutex_;
    DBInfo msMonitorDB_;
    std::atomic<uint32_t> reportInterval_{0};

    std::mutex dataMutex_;
    bool hasSavedData_{false};
    std::unordered_set<uint32_t> deviceSet_;
    // api data
    APIFormat apiData_;
    // communication data
    std::atomic<uint32_t> communicationOpId_{0};
    std::unordered_map<std::string, uint64_t> communicationGroupOpCount_;
    std::unordered_map<std::string, std::string> communicationGroupNameMap_;
    CommunicationOpFormat communicationOpData_;
    // compute task info data
    std::atomic<uint64_t> globalTaskId_{0};
    ComputeTaskInfoFormat computeTaskInfoData_;
    // task data
    TaskFormat taskData_;
    // mstx data
    std::unordered_map<uint64_t, MstxHostData> mstxRangeHostDataMap_;
    std::unordered_map<uint64_t, MstxDeviceData> mstxRangeDeviceDataMap_;
    MstxFormat mstxData_;
};

template<typename... Args>
bool InsertDataToDB(const std::vector<std::tuple<Args...>> &data, const std::string &tableName, DBInfo &msMonitorDB)
{
    LOG(INFO) << "InsertDataToDB tableName: " << tableName;
    if (data.empty()) {
        LOG(WARNING) << tableName << " is empty";
        return true;
    }
    if (msMonitorDB.dbRunner == nullptr) {
        LOG(ERROR) << "msMonitorDB dbRunner is null";
        return false;
    }
    if (msMonitorDB.database == nullptr) {
        LOG(ERROR) << "msMonitorDB database is null";
        return false;
    }
    if (!msMonitorDB.dbRunner->CreateTable(tableName, msMonitorDB.database->GetTableCols(tableName))) {
        LOG(ERROR) << "msMonitorDB " << tableName << " CreateTable failed";
        return false;
    }
    if (!msMonitorDB.dbRunner->InsertData(tableName, data)) {
        LOG(ERROR) << "msMonitorDB " << tableName << " InsertData failed";
        return false;
    }
    return true;
}

template<typename... Args>
bool DBProcessManager::SaveIncDataToDB(const std::vector<std::tuple<Args...>> &data, const std::string &tableName)
{
    if (data.empty()) {
        LOG(WARNING) << tableName << " is empty";
        return true;
    }
    bool ret = InsertDataToDB(data, tableName, msMonitorDB_);
    hasSavedData_ = hasSavedData_ || ret;
    return ret;
}
} // namespace db
} // namespace ipc_monitor
} // namespace dynolog_npu
#endif // IPC_MONITOR_DB_PROCESS_MANAGER_H
