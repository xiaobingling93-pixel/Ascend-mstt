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
#ifndef DYNOLOG_NPU_MONITOR_H
#define DYNOLOG_NPU_MONITOR_H

#include "MonitorBase.h"
#include "NpuIpcClient.h"
#include "MsptiMonitor.h"
#include "singleton.h"
#include "InputParser.h"

namespace dynolog_npu {
namespace ipc_monitor {

class DynoLogNpuMonitor : public MonitorBase, public Singleton<DynoLogNpuMonitor> {
    friend class Singleton<DynoLogNpuMonitor>;

public:
    DynoLogNpuMonitor();
    bool Init() override;
    ErrCode DealMonitorReq(MsptiMonitorCfg& cmd);
    std::string Poll() override;
    void EnableMsptiMonitor(std::unordered_map<std::string, std::string>& cfg_map);
    void Finalize();
    void SetNpuId(int id) override
    {
        npuId_ = id;
    }

    IpcClient *GetIpcClient()
    {
        return &ipcClient_;
    }

private:
    bool isInitialized_ = false;
    int32_t npuId_ = 0;
    IpcClient ipcClient_;
    MsptiMonitor msptiMonitor_;
};

} // namespace ipc_monitor
} // namespace dynolog_npu

#endif // DYNOLOG_NPU_MONITOR_H
