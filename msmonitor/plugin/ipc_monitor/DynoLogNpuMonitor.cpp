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
#include "DynoLogNpuMonitor.h"
#include <glog/logging.h>
#include <algorithm>
#include <iterator>
#include "utils.h"

namespace dynolog_npu {
namespace ipc_monitor {
DynoLogNpuMonitor::DynoLogNpuMonitor()
{
    // init glog
    if (!google::IsGoogleLoggingInitialized()) {
        std::string logPath;
        if (CreateMsmonitorLogPath(logPath)) {
            fprintf(stderr, "[INFO] [%d] Msmonitor log will record to %s\n", GetProcessId(), logPath.c_str());
            logPath = logPath + "/msmonitor_";
            google::InitGoogleLogging("MsMonitor");
            google::SetLogDestination(google::GLOG_INFO, logPath.c_str());
            google::SetLogFilenameExtension(".log");
        } else {
            fprintf(stderr, "Failed to create log path, log will not record\n");
        }
    }
}

bool DynoLogNpuMonitor::Init()
{
    if (isInitialized_) {
        LOG(WARNING) << "DynoLog npu monitor already initialized";
        return true;
    }
    if (!ipcClient_.Init()) {
        LOG(ERROR) << "DynoLog npu monitor ipcClient init failed";
        return false;
    }
    bool res = ipcClient_.RegisterInstance(npuId_);
    if (res) {
        isInitialized_ = true;
        LOG(INFO) << "DynoLog npu monitor initialized successfully";
    }
    return res;
}

ErrCode DynoLogNpuMonitor::DealMonitorReq(MsptiMonitorCfg& cmd)
{
    if (cmd.monitorStop) {
        if (msptiMonitor_.IsStarted()) {
            LOG(INFO) << "Stop mspti monitor thread successfully";
            msptiMonitor_.Stop();
        }
        return ErrCode::SUC;
    }

    if (cmd.reportIntervals <= 0) {
        cmd.reportIntervals = DEFAULT_FLUSH_INTERVAL;
        LOG(WARNING) << "Invalid report interval, set to 60";
    }
    if (cmd.reportIntervals != 0) {
        msptiMonitor_.SetFlushInterval(cmd.reportIntervals);
    }

    if (cmd.monitorStart && !msptiMonitor_.IsStarted()) {
        LOG(INFO) << "Start mspti monitor thread successfully";
        msptiMonitor_.Start();
    }

    if (msptiMonitor_.IsStarted() && !cmd.enableActivities.empty()) {
        auto curActivities = msptiMonitor_.GetEnabledActivities();
        std::vector<msptiActivityKind> enableKinds;
        std::vector<msptiActivityKind> disableKinds;
        std::set_difference(cmd.enableActivities.begin(), cmd.enableActivities.end(), curActivities.begin(), curActivities.end(),
                            std::back_inserter(enableKinds));
        std::set_difference(curActivities.begin(), curActivities.end(), cmd.enableActivities.begin(), cmd.enableActivities.end(),
                            std::back_inserter(disableKinds));
        for (auto activity : enableKinds) {
            msptiMonitor_.EnableActivity(activity);
        }
        for (auto activity : disableKinds) {
            msptiMonitor_.DisableActivity(activity);
        }
    }
    return ErrCode::SUC;
}

std::string DynoLogNpuMonitor::Poll()
{
    std::string res = ipcClient_.IpcClientNpuConfig();
    if (res.size() == 4) {  // res为4，表示dynolog注册进程成功
        LOG(INFO) << "Regist to dynolog daemon successfully";
        return "";
    }
    if (res.empty()) {
        return "";
    }
    LOG(INFO) << "Received NPU configuration successfully";
    return res;
}

void DynoLogNpuMonitor::EnableMsptiMonitor(std::unordered_map<std::string, std::string>& cfg_map)
{
    auto cmd = InputParser::GetInstance()->DynoLogGetOpts(cfg_map);
    if (cmd.isMonitor) {
        auto ans = DealMonitorReq(cmd);
        if (ans != ErrCode::SUC) {
            LOG(ERROR) << "Deal monitor request failed, because" << IPC_ERROR(ans);
        }
    }
}

void DynoLogNpuMonitor::Finalize()
{
    msptiMonitor_.Uninit();
}
} // namespace ipc_monitor
} // namespace dynolog_npu
