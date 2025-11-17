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
#ifndef MSMONITOR_JSONL_PROCESS_MANAGER_H
#define MSMONITOR_JSONL_PROCESS_MANAGER_H

#include <atomic>
#include <mutex>
#include <unordered_set>
#include <unordered_map>
#include "MsptiDataProcessBase.h"
#include "jsonl/JsonlDataDumper.h"

namespace dynolog_npu {
namespace ipc_monitor {
namespace jsonl {

struct MstxHostData {
    uint64_t timestamp;
    std::string domain;
    std::string message;
};

struct MstxDeviceData {
    uint64_t timestamp;
};

class JsonlProcessManager : public MsptiDataProcessBase {
public:
    JsonlProcessManager(std::string savePath)
        : MsptiDataProcessBase("JsonlProcessManager"), savePath_(std::move(savePath)) {}
    ~JsonlProcessManager() = default;
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
    bool SaveData();
    bool SaveParallelGroupData();
    bool SaveRankDeviceData();

private:
    uint64_t sessionStartTime_{0};
    std::string savePath_;
    std::mutex fileMutex_;
    std::atomic<uint32_t> reportInterval_{0};
    JsonlDataDumper dataDumper_;

    std::mutex dataMutex_;
    std::unordered_set<uint32_t> deviceSet_;
    // mstx data
    std::unordered_map<uint64_t, MstxHostData> mstxMarkerHostData_;
    std::unordered_map<uint64_t, MstxHostData> mstxRangeHostData_;
    std::unordered_map<uint64_t, MstxDeviceData> mstxRangeDeviceData_;
};
} // namespace jsonl
} // namespace ipc_monitor
} // namespace dynolog_npu
#endif // MSMONITOR_JSONL_PROCESS_MANAGER_H
