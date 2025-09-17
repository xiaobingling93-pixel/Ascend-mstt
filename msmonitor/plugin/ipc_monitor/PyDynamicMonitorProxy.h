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
#ifndef PYDYNAMIC_MONITOR_PROXY_H
#define PYDYNAMIC_MONITOR_PROXY_H

#include <glog/logging.h>
#include "MonitorBase.h"
#include "DynoLogNpuMonitor.h"

namespace dynolog_npu {
namespace ipc_monitor {

enum RunningState: int32_t {
    INIT = 0,
    FINALIZE = 1
};

class PyDynamicMonitorProxy : public Singleton<PyDynamicMonitorProxy> {
    friend class Singleton<PyDynamicMonitorProxy>;

public:
    PyDynamicMonitorProxy() = default;
    bool InitDyno(int npuId)
    {
        try {
            monitor_ = DynoLogNpuMonitor::GetInstance();
            monitor_->SetNpuId(npuId);
            bool res = monitor_->Init();
            if (res) {
                DynoLogNpuMonitor::GetInstance()->UpdateNpuStatus(RunningState::INIT, MSG_TYPE_TRACE_STATUS);
                DynoLogNpuMonitor::GetInstance()->UpdateNpuStatus(RunningState::INIT, MSG_TYPE_MONITOR_STATUS);
            }
            return res;
        } catch (const std::exception &e) {
            LOG(ERROR) << "Error when init dyno " << e.what();
            return false;
        }
    }

    std::string PollDyno()
    {
        return monitor_->Poll();
    }

    void EnableMsptiMonitor(std::unordered_map<std::string, std::string>& config_map)
    {
        DynoLogNpuMonitor::GetInstance()->EnableMsptiMonitor(config_map);
    }

    void FinalizeDyno()
    {
        DynoLogNpuMonitor::GetInstance()->Finalize();
        DynoLogNpuMonitor::GetInstance()->UpdateNpuStatus(RunningState::FINALIZE, MSG_TYPE_TRACE_STATUS);
        DynoLogNpuMonitor::GetInstance()->UpdateNpuStatus(RunningState::FINALIZE, MSG_TYPE_MONITOR_STATUS);
    }

    void UpdateProfilerStatus(std::unordered_map<std::string, std::string>& status)
    {
        int32_t npuTraceStatus = 0;
        auto it = status.find("profiler_status");
        if (it != status.end() && !it->second.empty()) {
            Str2Int32(npuTraceStatus, it->second);
        } else {
            LOG(ERROR) << "Missing key 'profiler_status'.";
            return;
        }
        DynoLogNpuMonitor::GetInstance()->UpdateNpuStatus(npuTraceStatus, MSG_TYPE_TRACE_STATUS);
    }
private:
    MonitorBase *monitor_ = nullptr;
};

} // namespace ipc_monitor
} // namespace dynolog_npu
#endif // PYDYNAMIC_MONITOR_PROXY_H
