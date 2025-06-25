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
#ifndef METRIC_MANAGER_H
#define METRIC_MANAGER_H

#include <vector>
#include <atomic>

#include "utils.h"
#include "singleton.h"
#include "mspti.h"
#include "TimerTask.h"
#include "MetricProcessBase.h"

namespace dynolog_npu {
namespace ipc_monitor {
namespace metric {
class MetricManager : public ipc_monitor::Singleton<MetricManager>, public TimerTask {
public:
    MetricManager();
    ~MetricManager() = default;
    ErrCode ConsumeMsptiData(msptiActivity *record);
    void SetReportInterval(uint32_t intervalTimes);
    void SendMetricMsg();
    void ExecuteTask() override;
    void EnableKindSwitch_(msptiActivityKind kind, bool flag);
    void ReleaseResource() override;
private:
    std::vector<std::atomic<bool>> kindSwitchs_;
    std::vector<std::atomic<bool>> consumeStatus_;
    std::atomic<uint32_t> reportInterval_;
    std::vector<std::shared_ptr<MetricProcessBase>> metrics;
};
}
}
}
#endif