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

#include "dynolog/src/ThreadManager.h"

#include <glog/logging.h>
#include <gflags/gflags.h>
#include <chrono>
#include <cstdlib>
#include "dynolog/src/utils.h"
#include "dynolog/src/Logger.h"

#include "dynolog/src/PerfMonitor.h"
#include "dynolog/src/ScubaLogger.h"
#include "dynolog/src/KernelCollector.h"
#include "dynolog/src/ODSJsonLogger.h"
#include "dynolog/src/CompositeLogger.h"
#include "dynolog/src/FBRelayLogger.h"

#include "hbt/src/perf_event/BuiltinMetrics.h"

#ifdef USE_PROMETHEUS
#include "dynolog/src/PrometheusLogger.h"
#endif

#ifdef USE_TENSORBOARD
#include "dynolog/src/DynologTensorBoardLogger.h"
#endif

namespace dynolog {
namespace hbt = facebook::hbt;

namespace {
constexpr int MS_PER_SECOND = 1000;

DEFINE_int32(port, 1778, "Port for listening RPC requests.");
DEFINE_bool(use_JSON, false, "Emit metrics to JSON file through JSON logger");
#ifdef USE_PROMETHEUS
DEFINE_bool(use_prometheus, false, "Emit metrics to Prometheus");
#endif
DEFINE_bool(use_fbrelay, false, "Emit metrics to FB Relay on Lab machines");
DEFINE_bool(use_ODS, false, "Emit metrics to ODS through ODS logger");
DEFINE_bool(use_scuba, false, "Emit metrics to Scuba through Scuba logger");
DEFINE_int32(
kernel_monitor_reporting_interval_s,
60,
"Duration in seconds to read and report metrics for kernel monitor");
DEFINE_int32(
perf_monitor_reporting_interval_s,
60,
"Duration in seconds to read and report metrics for performance monitor");
DEFINE_int32(
dcgm_reporting_interval_s,
10,
"Duration in seconds to read and report metrics for DCGM");
DEFINE_bool(
enable_ipc_monitor,
false,
"Enabled IPC monitor for on system tracing requests.");
DEFINE_bool(
enable_gpu_monitor,
false,
"Enabled GPU monitorng, currently supports NVIDIA GPUs.");
DEFINE_bool(enable_perf_monitor, false, "Enable heartbeat perf monitoring.");

std::unique_ptr<Logger> getLogger(const std::string& scribe_category = "")
{
    std::vector<std::unique_ptr<Logger>> loggers;
#ifdef USE_PROMETHEUS
    if (FLAGS_use_prometheus) {
        loggers.push_back(std::make_unique<PrometheusLogger>());
    }
#endif
#ifdef USE_TENSORBOARD
    if (!FLAGS_metric_log_dir.empty()) {
        loggers.push_back(std::make_unique<DynologTensorBoardLogger>(FLAGS_metric_log_dir));
    }
#endif
    if (FLAGS_use_fbrelay) {
        loggers.push_back(std::make_unique<FBRelayLogger>());
    }
    if (FLAGS_use_ODS) {
        loggers.push_back(std::make_unique<ODSJsonLogger>());
    }
    if (FLAGS_use_JSON) {
        loggers.push_back(std::make_unique<JsonLogger>());
    }
    if (FLAGS_use_scuba && !scribe_category.empty()) {
        loggers.push_back(std::make_unique<ScubaLogger>(scribe_category));
    }
    return std::make_unique<CompositeLogger>(std::move(loggers));
}

auto next_wakeup(int sec)
{
    return std::chrono::steady_clock::now() + std::chrono::seconds(sec);
}
}

void ThreadManager::gpu_monitor_loop()
{
    auto logger = getLogger(FLAGS_scribe_category);

    LOG(INFO) << "Running DCGM loop : interval = " << FLAGS_dcgm_reporting_interval_s << " s.";
    LOG(INFO) << "DCGM fields: " << gpumon::FLAGS_dcgm_fields;

    while (still_alive_.load()) {
        auto wakeup_timepoint = next_wakeup(FLAGS_dcgm_reporting_interval_s);

        dcgm_->update();
        dcgm_->log(*logger);

        /* sleep override */
        std::this_thread::sleep_until(wakeup_timepoint);
    }
}

void ThreadManager::kernel_monitor_loop()
{
    KernelCollector kc;

    LOG(INFO) << "Running kernel monitor loop : interval = " << FLAGS_kernel_monitor_reporting_interval_s << " s.";

    while (still_alive_.load()) {
        auto logger = getLogger();
        auto wakeup_timepoint = next_wakeup(FLAGS_kernel_monitor_reporting_interval_s);

        kc.step();
        kc.log(*logger);
        logger->finalize();

        /* sleep override */
        std::this_thread::sleep_until(wakeup_timepoint);
    }
}

void ThreadManager::perf_monitor_loop()
{
    PerfMonitor pm(
            hbt::CpuSet::makeAllOnline(),
            std::vector<ElemId>{"instructions", "cycles"},
            getDefaultPmuDeviceManager(),
            getDefaultMetrics());

    LOG(INFO) << "Running perf monitor loop : interval = " << FLAGS_perf_monitor_reporting_interval_s << " s.";

    while (still_alive_.load()) {
        auto logger = getLogger();
        auto wakeup_timepoint =
                next_wakeup(FLAGS_perf_monitor_reporting_interval_s);

        pm.step();
        pm.log(*logger);

        logger->finalize();
        /* sleep override */
        std::this_thread::sleep_until(wakeup_timepoint);
    }
}

void ThreadManager::start_threads()
{
    if (FLAGS_enable_ipc_monitor) {
        LOG(INFO) << "Starting IPC Monitor";
        ipcmon_ = std::make_shared<tracing::IPCMonitor>();
        ipcmon_->setLogger(std::move(getLogger()));
        ipcmon_thread_ = std::make_unique<std::thread>([this]() { this->ipcmon_->loop(); });
        data_ipcmon_thread_ = std::make_unique<std::thread>([this]() { this->ipcmon_->dataLoop(); });
    }

    if (FLAGS_enable_gpu_monitor) {
        dcgm_ = gpumon::DcgmGroupInfo::factory(gpumon::FLAGS_dcgm_fields,
                                               FLAGS_dcgm_reporting_interval_s * MS_PER_SECOND);
        gpumon_thread_ = std::make_unique<std::thread>([this]() { this->gpu_monitor_loop(); });
    }

    km_thread_ = std::make_unique<std::thread>([this]() { this->kernel_monitor_loop(); });

    if (FLAGS_enable_perf_monitor) {
        pm_thread_ = std::make_unique<std::thread>([this]() { this->perf_monitor_loop(); });
    }

    // setup service
    auto handler = std::make_shared<ServiceHandler>(dcgm_);

    // use simple json RPC server for now
    server_ = std::make_unique<SimpleJsonServer<ServiceHandler>>(handler, FLAGS_port);
    if (server_) {
        server_->run();
    }
}

void ThreadManager::run(int argc, char** argv)
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    FLAGS_logtostderr = 1;
    google::InitGoogleLogging(argv[0]);

    if (dynolog::IsRoot()) {
        LOG(WARNING) << "Security Warning: Do not run this tool as root. "
                     << "Running with elevated privileges may compromise system security. "
                     << "Use a regular user account.";
    }
    LOG(INFO) << "Starting Ascend Extension for dynolog, version = " DYNOLOG_VERSION
        << ", build git-hash = " DYNOLOG_GIT_REV;

    try {
        start_threads();
    } catch (const std::exception& e) {
        LOG(ERROR) << "ThreadManager run failed: " << e.what();
        // to stop km_thread_, pm_thread_ and gpumon_thread_
        still_alive_.store(false);

        // to stop ipcmon_thread_ and data_ipcmon_thread_
        if (ipcmon_) {
            ipcmon_->release();
        }
    }
}

void ThreadManager::stop()
{
    LOG(INFO) << "Wait for ThreadManager stop.";
    if (km_thread_ && km_thread_->joinable()) {
        km_thread_->join();
    }

    if (pm_thread_ && pm_thread_->joinable()) {
        pm_thread_->join();
    }

    if (gpumon_thread_ && gpumon_thread_->joinable()) {
        gpumon_thread_->join();
    }

    if (ipcmon_thread_ && ipcmon_thread_->joinable()) {
        ipcmon_thread_->join();
    }

    if (data_ipcmon_thread_ && data_ipcmon_thread_->joinable()) {
        data_ipcmon_thread_->join();
    }

    if (server_) {
        server_->stop();
    }
}

}
