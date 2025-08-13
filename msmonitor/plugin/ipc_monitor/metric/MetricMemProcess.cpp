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
#include "MetricMemProcess.h"

#include <numeric>

namespace dynolog_npu {
namespace ipc_monitor {
namespace metric {

std::string MemMetric::seriesToJson()
{
    nlohmann::json jsonMsg;
    jsonMsg["kind"] = "Memory";
    jsonMsg["deviceId"] = deviceId;
    jsonMsg["duration"] = duration;
    jsonMsg["timestamp"] = timestamp;
    return jsonMsg.dump();
}

void MetricMemProcess::ConsumeMsptiData(msptiActivity *record)
{
    msptiActivityMemory* mem = ReinterpretConvert<msptiActivityMemory*>(record);
    std::shared_ptr<msptiActivityMemory> tmp;
    MakeSharedPtr(tmp);
    if (tmp == nullptr || memcpy_s(tmp.get(), sizeof(msptiActivityMemory), mem, sizeof(msptiActivityMemory)) != EOK) {
        LOG(ERROR) << "memcpy_s failed " << IPC_ERROR(ErrCode::MEMORY);
        return;
    }
    {
        std::unique_lock<std::mutex> lock(dataMutex);
        records.emplace_back(std::move(tmp));
    }
}

std::vector<MemMetric> MetricMemProcess::AggregatedData()
{
    std::vector<std::shared_ptr<msptiActivityMemory>> copyRecords;
    {
        std::unique_lock<std::mutex> lock(dataMutex);
        copyRecords = std::move(records);
        records.clear();
    }
    if (copyRecords.empty()) {
        return {};
    }
    std::unordered_map<uint32_t, std::vector<std::shared_ptr<msptiActivityMemory>>> deviceId2MemData =
    groupby(copyRecords, [](const std::shared_ptr<msptiActivityMemory>& data) -> std::uint32_t {
        return data->deviceId;
    });
    std::vector<MemMetric> ans;
    auto curTimestamp = getCurrentTimestamp64();
    for (auto& pair: deviceId2MemData) {
        auto deviceId = pair.first;
        auto& memDatas = pair.second;
        MemMetric memMetric{};
        memMetric.duration = std::accumulate(memDatas.begin(), memDatas.end(), 0ULL,
            [](uint64_t acc, std::shared_ptr<msptiActivityMemory> mem) {
                return acc + mem->end - mem->start;
            });
        memMetric.deviceId = deviceId;
        memMetric.timestamp = curTimestamp;
        ans.emplace_back(memMetric);
    }
    return ans;
}

void MetricMemProcess::SendProcessMessage()
{
    auto afterAggregated = AggregatedData();
    for (auto& metric: afterAggregated) {
        SendMessage(metric.seriesToJson());
    }
}

void MetricMemProcess::Clear()
{
    records.clear();
}
}
}
}