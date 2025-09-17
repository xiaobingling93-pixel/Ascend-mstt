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

#pragma once

#include <atomic>
#include <thread>

#include "dynolog/src/gpumon/DcgmGroupInfo.h"
#include "dynolog/src/ServiceHandler.h"
#include "dynolog/src/rpc/SimpleJsonServer.h"
#include "dynolog/src/rpc/SimpleJsonServerInl.h"
#include "dynolog/src/tracing/IPCMonitor.h"

DECLARE_int32(port);
DECLARE_bool(use_JSON);
#ifdef USE_PROMETHEUS
DECLARE_bool(use_prometheus);
#endif
DECLARE_bool(use_fbrelay);
DECLARE_bool(use_ODS);
DECLARE_bool(use_scuba);
DECLARE_int32(kernel_monitor_reporting_interval_s);
DECLARE_int32(perf_monitor_reporting_interval_s);
DECLARE_int32(dcgm_reporting_interval_s);
DECLARE_bool(enable_ipc_monitor);
DECLARE_bool(enable_gpu_monitor);
DECLARE_bool(enable_perf_monitor);

namespace dynolog {

class ThreadManager {
public:
    ThreadManager() : still_alive_(true) {}
    ~ThreadManager() { stop(); }

    void run(int argc, char** argv);

private:
    void start_threads();
    void stop();
    void gpu_monitor_loop();
    void kernel_monitor_loop();
    void perf_monitor_loop();

private:
    std::atomic<bool> still_alive_;
    std::shared_ptr<gpumon::DcgmGroupInfo> dcgm_;
    std::shared_ptr<tracing::IPCMonitor> ipcmon_;
    std::unique_ptr<std::thread> ipcmon_thread_;
    std::unique_ptr<std::thread> data_ipcmon_thread_;
    std::unique_ptr<std::thread> gpumon_thread_;
    std::unique_ptr<std::thread> pm_thread_;
    std::unique_ptr<std::thread> km_thread_;
    std::unique_ptr<SimpleJsonServer<ServiceHandler>> server_;
};
}

