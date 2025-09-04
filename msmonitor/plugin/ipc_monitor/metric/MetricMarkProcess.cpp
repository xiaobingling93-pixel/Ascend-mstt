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
#include "MetricMarkProcess.h"

#include <nlohmann/json.hpp>
#include <glog/logging.h>
#include <numeric>

#include "utils.h"


namespace dynolog_npu {
namespace ipc_monitor {
namespace metric {

constexpr size_t COMPLETE_RANGE_DATA_SIZE = 4;

std::string MarkMetric::seriesToJson()
{
    nlohmann::json jsonMsg;
    jsonMsg["kind"] = "Marker";
    jsonMsg["deviceId"] = deviceId;
    jsonMsg["domain"] = domain;
    jsonMsg["name"] = name;
    jsonMsg["duration"] = duration;
    jsonMsg["timestamp"] = timestamp;
    return jsonMsg.dump();
}

bool MetricMarkProcess::TransMarkData2Range(const std::vector<std::shared_ptr<msptiActivityMarker>>& markDatas,
    RangeMarkData& rangemarkData)
{
    if (markDatas.size() != COMPLETE_RANGE_DATA_SIZE) {
        return false;
    }

    for (auto& activityMarker: markDatas) {
        if (activityMarker->flag == MSPTI_ACTIVITY_FLAG_MARKER_START_WITH_DEVICE) {
            if (activityMarker->sourceKind == MSPTI_ACTIVITY_SOURCE_KIND_DEVICE) {
                rangemarkData.deviceId = activityMarker->objectId.ds.deviceId;
                rangemarkData.deviceStart = activityMarker->timestamp;
            } else {
                rangemarkData.start = activityMarker->timestamp;
                rangemarkData.name = activityMarker->name;
            }
        }
        if (activityMarker->flag == MSPTI_ACTIVITY_FLAG_MARKER_END_WITH_DEVICE) {
            if (activityMarker->sourceKind == MSPTI_ACTIVITY_SOURCE_KIND_DEVICE) {
                rangemarkData.deviceEnd = activityMarker->timestamp;
            } else {
                rangemarkData.end = activityMarker->timestamp;
            }
        }
    }
    auto markId = markDatas[0]->id;
    std::string domainName = "default";
    auto it = domainMsg.find(markId);
    if (it != domainMsg.end()) {
        domainName = *it->second;
    }
    rangemarkData.domain = domainName;
    id2Marker.erase(markId);
    domainMsg.erase(markId);
    return true;
}

void MetricMarkProcess::ConsumeMsptiData(msptiActivity *record)
{
    msptiActivityMarker* markerData = ReinterpretConvert<msptiActivityMarker*>(record);
    std::shared_ptr<msptiActivityMarker> tmp;
    MakeSharedPtr(tmp);
    if (tmp == nullptr || memcpy_s(tmp.get(), sizeof(msptiActivityMarker), markerData, sizeof(msptiActivityMarker)) != EOK) {
        LOG(ERROR) << "memcpy_s failed " << IPC_ERROR(ErrCode::MEMORY);
        return;
    }
    {
        std::unique_lock<std::mutex> lock(dataMutex);
        records.emplace_back(std::move(tmp));
        if (markerData->flag == MSPTI_ACTIVITY_FLAG_MARKER_START_WITH_DEVICE &&
            markerData->sourceKind == MSPTI_ACTIVITY_SOURCE_KIND_HOST) {
            std::string domainStr = markerData->domain;
            auto markId = markerData->id;
            domainMsg.emplace(markId, std::make_shared<std::string>(domainStr));
        }
    }
}

std::vector<MarkMetric> MetricMarkProcess::AggregatedData()
{
    std::vector<std::shared_ptr<msptiActivityMarker>> copyRecords;
    {
        std::unique_lock<std::mutex> lock(dataMutex);
        copyRecords = std::move(records);
        records.clear();
    }
    for (auto& record: copyRecords) {
        id2Marker[record->id].emplace_back(std::move(record));
    }
    std::vector<RangeMarkData> rangeDatas;
    for (auto pair = id2Marker.rbegin(); pair != id2Marker.rend(); ++pair) {
        auto markId = pair->first;
        auto markDatas = pair->second;
        RangeMarkData rangeMark{};
        if (TransMarkData2Range(markDatas, rangeMark)) {
            rangeDatas.emplace_back(rangeMark);
        }
    }

    std::vector<MarkMetric> ans;
    MarkMetric hostMarkMetric{};
    MarkMetric devMarkMetric{};
    for (const auto& data : rangeDatas) {
        hostMarkMetric.name = data.name;
        hostMarkMetric.domain = data.domain;
        hostMarkMetric.deviceId = -1;
        hostMarkMetric.duration = data.end - data.start;
        hostMarkMetric.timestamp = data.start;
        ans.emplace_back(hostMarkMetric);

        devMarkMetric.name = data.name;
        devMarkMetric.domain = data.domain;
        devMarkMetric.deviceId = data.deviceId;
        devMarkMetric.duration = data.deviceEnd - data.deviceStart;
        devMarkMetric.timestamp = data.deviceStart;
        ans.emplace_back(devMarkMetric);
    }
    return ans;
}

void MetricMarkProcess::SendProcessMessage()
{
    auto afterAggregated = AggregatedData();
    for (auto& metric: afterAggregated) {
        SendMessage(metric.seriesToJson());
    }
}

void MetricMarkProcess::Clear()
{
    records.clear();
    domainMsg.clear();
    id2Marker.clear();
}
}
}
}
