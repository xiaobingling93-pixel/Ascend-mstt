// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <memory>

// Use glog for FabricManager.h
#define USE_GOOGLE_LOG

#include "dynolog/src/ipcfabric/FabricManager.h"
#include "dynolog/src/Logger.h"

namespace dynolog {
namespace tracing {

class IPCMonitor {
public:
    using FabricManager = dynolog::ipcfabric::FabricManager;
    IPCMonitor(const std::string& ipc_fabric_name = "dynolog");
    virtual ~IPCMonitor() {}

    void loop();
    void dataLoop();

public:
    virtual void processMsg(std::unique_ptr<ipcfabric::Message> msg);
    virtual void processDataMsg(std::unique_ptr<ipcfabric::Message> msg);
    void getLibkinetoOnDemandRequest(std::unique_ptr<ipcfabric::Message> msg);
    void registerLibkinetoContext(std::unique_ptr<ipcfabric::Message> msg);
    void setLogger(std::unique_ptr<Logger> logger);
    void LogData(const nlohmann::json& result);

    std::unique_ptr<ipcfabric::FabricManager> ipc_manager_;
    std::unique_ptr<ipcfabric::FabricManager> data_ipc_manager_;
    std::unique_ptr<Logger> logger_;

  // friend class test_case_name##_##test_name##_Test
    friend class IPCMonitorTest_LibkinetoRegisterAndOndemandTest_Test;
};

} // namespace tracing
} // namespace dynolog
