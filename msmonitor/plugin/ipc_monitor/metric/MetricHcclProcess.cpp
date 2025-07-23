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
#include "MetricHcclProcess.h"
#include <numeric>
#include <nlohmann/json.hpp>
#include "utils.h"

namespace dynolog_npu {
namespace ipc_monitor {
namespace metric {

std::string HcclMetric::seriesToJson()
{
    nlohmann::json jsonMsg;
    jsonMsg["kind"] = "Hccl";
    jsonMsg["deviceId"] = deviceId;
    jsonMsg["duration"] = duration;
    jsonMsg["timestamp"] = timestamp;
    return jsonMsg.dump();
}

void MetricHcclProcess::ConsumeMsptiData(msptiActivity *record)
{
    msptiActivityHccl* hcclData = ReinterpretConvert<msptiActivityHccl*>(record);
    std::shared_ptr<msptiActivityHccl> tmp;
    MakeSharedPtr(tmp);
    if (tmp == nullptr || memcpy_s(tmp.get(), sizeof(msptiActivityHccl), hcclData, sizeof(msptiActivityHccl)) != EOK) {
        LOG(ERROR) << "memcpy_s failed " << IPC_ERROR(ErrCode::MEMORY);
        return;
    }
    {
        std::unique_lock<std::mutex> lock(dataMutex);
        records.emplace_back(std::move(tmp));
    }
}

std::vector<HcclMetric> MetricHcclProcess::AggregatedData()
{
    std::vector<std::shared_ptr<msptiActivityHccl>> copyRecords;
    {
        std::unique_lock<std::mutex> lock(dataMutex);
        copyRecords = std::move(records);
        records.clear();
    }
    if (copyRecords.empty()) {
        return {};
    }
    std::unordered_map<uint32_t, std::vector<std::shared_ptr<msptiActivityHccl>>> deviceId2HcclData =
        groupby(copyRecords, [](const std::shared_ptr<msptiActivityHccl>& data) -> std::uint32_t {
            return data->ds.deviceId;
        });
    std::vector<HcclMetric> ans;
    auto curTimestamp = getCurrentTimestamp64();
    for (auto& pair: deviceId2HcclData) {
        HcclMetric hcclMetric{};
        auto& hcclDatas = pair.second;
        hcclMetric.duration = std::accumulate(hcclDatas.begin(), hcclDatas.end(), 0ULL,
            [](uint64_t acc, std::shared_ptr<msptiActivityHccl> hccl) {
                return acc + hccl->end - hccl->start;
            });
        hcclMetric.deviceId = pair.first;
        hcclMetric.timestamp = curTimestamp;
        ans.emplace_back(hcclMetric);
    }
    return ans;
}

void MetricHcclProcess::SendProcessMessage()
{
    auto afterAggregated = AggregatedData();
    for (auto& metric: afterAggregated) {
        SendMessage(metric.seriesToJson());
    }
}

void MetricHcclProcess::Clear()
{
    records.clear();
}
}
}
}
