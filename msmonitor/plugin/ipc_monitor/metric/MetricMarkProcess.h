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
#ifndef METRIC_MARK_PROCESS_H
#define METRIC_MARK_PROCESS_H

#include <vector>
#include <memory>
#include "MetricProcessBase.h"


namespace dynolog_npu {
namespace ipc_monitor {
namespace metric {

struct MarkMetric {
    std::string name;
    std::string domain;
    uint64_t duration;
    uint64_t timestamp;
    uint32_t deviceId;
public:
    std::string seriesToJson();
};

struct RangeMarkData {
    std::string domain;
    uint64_t duration;
    uint64_t start{0};
    uint64_t end{0};
    uint64_t deviceStart{0};
    uint64_t deviceEnd{0};
    uint32_t deviceId;
};


class MetricMarkProcess : public MetricProcessBase {
public:
    MetricMarkProcess() = default;
    void ConsumeMsptiData(msptiActivity *record) override;
    std::vector<MarkMetric> AggregatedData();
    void SendProcessMessage() override;
    void Clear() override;
private:
    bool TransMarkData2Range(const std::vector<std::shared_ptr<msptiActivityMarker>>& markDatas,
        RangeMarkData& rangemarkData);
private:
    std::mutex dataMutex;
    std::unordered_map<uint64_t, std::shared_ptr<std::string>> domainMsg;
    std::vector<std::shared_ptr<msptiActivityMarker>> records;
    std::map<uint64_t, std::vector<std::shared_ptr<msptiActivityMarker>>> id2Marker;
};
}
}
}

#endif