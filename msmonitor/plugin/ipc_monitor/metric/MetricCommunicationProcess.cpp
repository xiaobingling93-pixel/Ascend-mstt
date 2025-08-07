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
#include "MetricCommunicationProcess.h"
#include <numeric>
#include <nlohmann/json.hpp>
#include "utils.h"

namespace dynolog_npu {
namespace ipc_monitor {
namespace metric {

std::string CommunicationMetric::seriesToJson()
{
    nlohmann::json jsonMsg;
    jsonMsg["kind"] = "Communication";
    jsonMsg["deviceId"] = deviceId;
    jsonMsg["duration"] = duration;
    jsonMsg["timestamp"] = timestamp;
    return jsonMsg.dump();
}

void MetricCommunicationProcess::ConsumeMsptiData(msptiActivity *record)
{
    msptiActivityCommunication* communicationData = ReinterpretConvert<msptiActivityCommunication*>(record);
    std::shared_ptr<msptiActivityCommunication> tmp;
    MakeSharedPtr(tmp);
    if (tmp == nullptr || memcpy_s(tmp.get(), sizeof(msptiActivityCommunication), communicationData, sizeof(msptiActivityCommunication)) != EOK) {
        LOG(ERROR) << "memcpy_s failed " << IPC_ERROR(ErrCode::MEMORY);
        return;
    }
    {
        std::unique_lock<std::mutex> lock(dataMutex);
        records.emplace_back(std::move(tmp));
    }
}

std::vector<CommunicationMetric> MetricCommunicationProcess::AggregatedData()
{
    std::vector<std::shared_ptr<msptiActivityCommunication>> copyRecords;
    {
        std::unique_lock<std::mutex> lock(dataMutex);
        copyRecords = std::move(records);
        records.clear();
    }
    if (copyRecords.empty()) {
        return {};
    }
    std::unordered_map<uint32_t, std::vector<std::shared_ptr<msptiActivityCommunication>>> deviceId2CommunicationData =
        groupby(copyRecords, [](const std::shared_ptr<msptiActivityCommunication>& data) -> std::uint32_t {
            return data->ds.deviceId;
        });
    std::vector<CommunicationMetric> ans;
    auto curTimestamp = getCurrentTimestamp64();
    for (auto& pair: deviceId2CommunicationData) {
        CommunicationMetric communicationMetric{};
        auto& communicationDatas = pair.second;
        communicationMetric.duration = std::accumulate(communicationDatas.begin(), communicationDatas.end(), 0ULL,
            [](uint64_t acc, std::shared_ptr<msptiActivityCommunication> communication) {
                return acc + communication->end - communication->start;
            });
        communicationMetric.deviceId = pair.first;
        communicationMetric.timestamp = curTimestamp;
        ans.emplace_back(communicationMetric);
    }
    return ans;
}

void MetricCommunicationProcess::SendProcessMessage()
{
    auto afterAggregated = AggregatedData();
    for (auto& metric: afterAggregated) {
        SendMessage(metric.seriesToJson());
    }
}

void MetricCommunicationProcess::Clear()
{
    records.clear();
}
}
}
}
