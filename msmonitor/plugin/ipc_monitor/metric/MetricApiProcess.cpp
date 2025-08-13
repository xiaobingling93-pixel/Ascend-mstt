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
#include "MetricApiProcess.h"

#include <numeric>
#include <nlohmann/json.hpp>

#include "utils.h"

namespace dynolog_npu {
namespace ipc_monitor {
namespace metric {

std::string ApiMetric::seriesToJson()
{
    nlohmann::json jsonMsg;
    jsonMsg["kind"] = "API";
    jsonMsg["deviceId"] = -1;
    jsonMsg["duration"] = duration;
    jsonMsg["timestamp"] = timestamp;
    return jsonMsg.dump();
}

void MetricApiProcess::ConsumeMsptiData(msptiActivity *record)
{
    msptiActivityApi* apiData = ReinterpretConvert<msptiActivityApi*>(record);
    std::shared_ptr<msptiActivityApi> tmp;
    MakeSharedPtr(tmp);
    if (tmp == nullptr || memcpy_s(tmp.get(), sizeof(msptiActivityApi), apiData, sizeof(msptiActivityApi)) != EOK) {
        LOG(ERROR) << "memcpy_s failed " << IPC_ERROR(ErrCode::MEMORY);
        return;
    }
    {
        std::unique_lock<std::mutex> lock(dataMutex);
        records.emplace_back(std::move(tmp));
    }
}

std::vector<ApiMetric> MetricApiProcess::AggregatedData()
{
    std::vector<std::shared_ptr<msptiActivityApi>> copyRecords;
    {
        std::unique_lock<std::mutex> lock(dataMutex);
        copyRecords = std::move(records);
        records.clear();
    }
    ApiMetric apiMetric{};
    auto ans = std::accumulate(copyRecords.begin(), copyRecords.end(), 0ULL,
        [](uint64_t acc, std::shared_ptr<msptiActivityApi> api) {
                    return acc + api->end - api->start;
                });
    apiMetric.duration = ans;
    apiMetric.deviceId = -1;
    apiMetric.timestamp = getCurrentTimestamp64();
    return {apiMetric};
}

void MetricApiProcess::SendProcessMessage()
{
    auto afterAggregated = AggregatedData();
    for (auto& metric: afterAggregated) {
        SendMessage(metric.seriesToJson());
    }
}

void MetricApiProcess::Clear()
{
    records.clear();
}
}
}
}
