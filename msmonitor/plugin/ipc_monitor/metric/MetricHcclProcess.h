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
#ifndef METRIC_HCCL_PROCESS_H
#define METRIC_HCCL_PROCESS_H

#include <vector>
#include <memory>
#include "MetricProcessBase.h"


namespace dynolog_npu {
namespace ipc_monitor {
namespace metric {

struct HcclMetric {
    std::string kindName;
    uint64_t duration;
    uint64_t timestamp;
    uint32_t deviceId;
public:
    std::string seriesToJson();
};

class MetricHcclProcess : public MetricProcessBase {
public:
    MetricHcclProcess() = default;
    void ConsumeMsptiData(msptiActivity *record) override;
    std::vector<HcclMetric> AggregatedData();
    void SendProcessMessage() override;
    void Clear() override;
private:
    std::mutex dataMutex;
    std::vector<std::shared_ptr<msptiActivityHccl>> records;
};
}
}
}

#endif