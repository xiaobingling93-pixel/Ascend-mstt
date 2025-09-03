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
#include "MsptiMonitor.h"

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
            google::SetStderrLogging(google::GLOG_ERROR);
            google::SetLogDestination(google::GLOG_INFO, logPath.c_str());
            google::SetLogFilenameExtension(".log");
        } else {
            fprintf(stderr, "Failed to create log path, log will not record\n");
        }
    }
    msptiActivityDisableMarkerDomain("communication");  // filter inner communication marker data for now
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
    auto msptiMonitor = MsptiMonitor::GetInstance();
    if (cmd.monitorStop) {
        if (msptiMonitor->IsStarted()) {
            LOG(INFO) << "Stop mspti monitor thread successfully";
            msptiMonitor->Stop();
        }
        return ErrCode::SUC;
    }

    if (cmd.reportIntervals != 0) {
        msptiMonitor->SetFlushInterval(cmd.reportIntervals);
    }

    if (cmd.monitorStart && !msptiMonitor->IsStarted()) {
        if (!cmd.savePath.empty() && !msptiMonitor->CheckAndSetSavePath(cmd.savePath)) {
            LOG(ERROR) << "Invalid log path, mspti monitor start failed";
            return ErrCode::PERMISSION;
        }

        LOG(INFO) << "Start mspti monitor thread successfully";
        msptiMonitor->Start();
    }

    if (msptiMonitor->IsStarted() && !cmd.enableActivities.empty()) {
        auto curActivities = msptiMonitor->GetEnabledActivities();
        std::vector<msptiActivityKind> enableKinds;
        std::vector<msptiActivityKind> disableKinds;
        std::set_difference(cmd.enableActivities.begin(), cmd.enableActivities.end(), curActivities.begin(), curActivities.end(),
                            std::back_inserter(enableKinds));
        std::set_difference(curActivities.begin(), curActivities.end(), cmd.enableActivities.begin(), cmd.enableActivities.end(),
                            std::back_inserter(disableKinds));
        for (auto activity : enableKinds) {
            msptiMonitor->EnableActivity(activity);
        }
        for (auto activity : disableKinds) {
            msptiMonitor->DisableActivity(activity);
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
        UpdateNpuStatus(static_cast<int32_t>(MsptiMonitor::GetInstance()->IsStarted()), MSG_TYPE_MONITOR_STATUS);
    }
}

void DynoLogNpuMonitor::Finalize()
{
    MsptiMonitor::GetInstance()->Uninit();
}

void DynoLogNpuMonitor::UpdateNpuStatus(int32_t status, const std::string& msgType)
{
    bool res = ipcClient_.SendNpuStatus(status, msgType);
    if (res) {
        LOG(INFO) << "Send npu status successfully";
    } else {
        LOG(WARNING) << "Send npu status failed";
    }
}
} // namespace ipc_monitor
} // namespace dynolog_npu
