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
#include "MetricMemCpyProcess.h"

#include <numeric>

namespace dynolog_npu {
namespace ipc_monitor {
namespace metric {

std::string MemCpyMetric::seriesToJson()
{
    nlohmann::json jsonMsg;
    jsonMsg["kind"] = "MemCpy";
    jsonMsg["deviceId"] = deviceId;
    jsonMsg["duration"] = duration;
    jsonMsg["timestamp"] = timestamp;
    return jsonMsg.dump();
}

void MetricMemCpyProcess::ConsumeMsptiData(msptiActivity *record)
{
    msptiActivityMemcpy* kernel = ReinterpretConvert<msptiActivityMemcpy*>(record);
    std::shared_ptr<msptiActivityMemcpy> tmp;
    MakeSharedPtr(tmp);
    if (tmp == nullptr || memcpy_s(tmp.get(), sizeof(msptiActivityMemcpy), kernel, sizeof(msptiActivityMemcpy)) != EOK) {
        LOG(ERROR) << "memcpy_s failed " << IPC_ERROR(ErrCode::MEMORY);
        return;
    }
    {
        std::unique_lock<std::mutex> lock(dataMutex);
        records.emplace_back(std::move(tmp));
    }
}

std::vector<MemCpyMetric> MetricMemCpyProcess::AggregatedData()
{
    std::vector<std::shared_ptr<msptiActivityMemcpy>> copyRecords;
    {
        std::unique_lock<std::mutex> lock(dataMutex);
        copyRecords = std::move(records);
        records.clear();
    }
    if (copyRecords.empty()) {
        return {};
    }
    std::unordered_map<uint32_t, std::vector<std::shared_ptr<msptiActivityMemcpy>>> deviceId2Memcpy =
    groupby(copyRecords, [](const std::shared_ptr<msptiActivityMemcpy>& data) -> std::uint32_t {
        return data->deviceId;
    });
    std::vector<MemCpyMetric> ans;
    auto curTimestamp = getCurrentTimestamp64();
    for (auto& pair: deviceId2Memcpy) {
        auto deviceId = pair.first;
        MemCpyMetric memCpyMetric{};
        auto& memCpyDatas = pair.second;
        memCpyMetric.duration = std::accumulate(memCpyDatas.begin(), memCpyDatas.end(), 0ULL,
            [](uint64_t acc, std::shared_ptr<msptiActivityMemcpy> memcpy) {
                return acc + memcpy->end - memcpy->start;
            });
        memCpyMetric.deviceId = deviceId;
        memCpyMetric.timestamp = curTimestamp;
        ans.emplace_back(memCpyMetric);
    }
    return ans;
}

void MetricMemCpyProcess::SendProcessMessage()
{
    auto afterAggregated = AggregatedData();
    for (auto& metric: afterAggregated) {
        SendMessage(metric.seriesToJson());
    }
}

void MetricMemCpyProcess::Clear()
{
    records.clear();
}
}
}
}
