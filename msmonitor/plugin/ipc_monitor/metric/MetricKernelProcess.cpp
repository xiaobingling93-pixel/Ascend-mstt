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
#include "MetricKernelProcess.h"

#include <numeric>

namespace dynolog_npu {
namespace ipc_monitor {
namespace metric {

std::string KernelMetric::seriesToJson()
{
    nlohmann::json jsonMsg;
    jsonMsg["kind"] = "Kernel";
    jsonMsg["deviceId"] = deviceId;
    jsonMsg["duration"] = duration;
    jsonMsg["timestamp"] = timestamp;
    return jsonMsg.dump();
}

void MetricKernelProcess::ConsumeMsptiData(msptiActivity *record)
{
    msptiActivityKernel* kernel = ReinterpretConvert<msptiActivityKernel*>(record);
    std::shared_ptr<msptiActivityKernel> tmp;
    MakeSharedPtr(tmp);
    if (tmp == nullptr || memcpy_s(tmp.get(), sizeof(msptiActivityKernel), kernel, sizeof(msptiActivityKernel)) != EOK) {
        LOG(ERROR) << "memcpy_s failed " << IPC_ERROR(ErrCode::MEMORY);
        return;
    }
    {
        std::unique_lock<std::mutex> lock(dataMutex);
        records.emplace_back(std::move(tmp));
    }
}

std::vector<KernelMetric> MetricKernelProcess::AggregatedData()
{
    std::vector<std::shared_ptr<msptiActivityKernel>> copyRecords;
    {
        std::unique_lock<std::mutex> lock(dataMutex);
        copyRecords = std::move(records);
        records.clear();
    }
    if (copyRecords.empty()) {
        return {};
    }
    std::unordered_map<uint32_t, std::vector<std::shared_ptr<msptiActivityKernel>>> deviceId2KernelData =
        groupby(copyRecords, [](const std::shared_ptr<msptiActivityKernel>& data) -> std::uint32_t {
            return data->ds.deviceId;
        });
    std::vector<KernelMetric> ans;
    auto curTimestamp = getCurrentTimestamp64();
    for (auto& pair: deviceId2KernelData) {
        auto deviceId = pair.first;
        auto& kernelDatas = pair.second;
        KernelMetric kernelMetric{};
        kernelMetric.duration = std::accumulate(kernelDatas.begin(), kernelDatas.end(), 0ULL,
            [](uint64_t acc, std::shared_ptr<msptiActivityKernel> kernel) {
                return acc + kernel->end - kernel->start;
            });
        kernelMetric.deviceId = deviceId;
        kernelMetric.timestamp = curTimestamp;
        ans.emplace_back(kernelMetric);
    }

    return ans;
}

void MetricKernelProcess::SendProcessMessage()
{
    auto afterAggregated = AggregatedData();
    for (auto& metric: afterAggregated) {
        SendMessage(metric.seriesToJson());
    }
}

void MetricKernelProcess::Clear()
{
    records.clear();
}
}
}
}
