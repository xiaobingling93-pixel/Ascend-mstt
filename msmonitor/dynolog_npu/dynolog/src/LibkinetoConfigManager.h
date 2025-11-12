// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>
#include "dynolog/src/LibkinetoTypes.h"

namespace dynolog {

const std::string kLibkinetoTraceStatus = "npuTraceStatus";
const std::string kLibkinetoMonitorStatus = "npuMonitorStatus";

struct NpuStatus {
    int32_t status = -1;
    int32_t currentStep = -1;
    int32_t startStep = -1;
    int32_t stopStep = -1;
    pid_t pid;
    int64_t jobId;
};

class LibkinetoConfigManager {
public:
    LibkinetoConfigManager();
    virtual ~LibkinetoConfigManager();

    int32_t
        registerLibkinetoContext(const std::string& jobId, int32_t pid, int32_t gpu);
    static std::shared_ptr<LibkinetoConfigManager> getInstance();

    std::string getBaseConfig()
    {
        std::lock_guard<std::mutex> guard(mutex_);
        return baseConfig_;
    }

    std::string obtainOnDemandConfig(
        const std::string& jobId,
        const std::vector<int32_t>& pids,
        int32_t configType);

    GpuProfilerResult setOnDemandConfig(
        const std::string& jobId,
        const std::set<int32_t>& pids,
        const std::string& config,
        int32_t configType,
        int32_t limit);

    void updateNpuStatus(const NpuStatus& status, const std::string& msgType);
    NpuStatus getNpuTraceStatus() const;
    NpuStatus getNpuMonitorStatus() const;

    // Return the number of active libkineto processes
    // with the given Chronos / Tangram Job Id
    int processCount(const std::string& jobId) const;

protected:
    struct LibkinetoProcess {
        int32_t pid;
        std::chrono::system_clock::time_point lastRequestTime;
        std::string eventProfilerConfig;
        std::string activityProfilerConfig;
    };

    // A few callbacks for additional instrumentation.
    virtual void onRegisterProcess(const std::set<int32_t>& /* pids */) {}

    virtual void preCheckOnDemandConfig(const LibkinetoProcess& /* process */) {}

    virtual void onSetOnDemandConfig(const std::set<int32_t>& /* pids */) {}

    virtual void onProcessCleanup(const std::set<int32_t>& /* pids */) {}

    // Map of pid ancestry -> LibkinetoProcess
    using ProcessMap = std::map<std::set<int32_t>, LibkinetoProcess>;
    std::map<std::string, ProcessMap> jobs_;

    // Map of gpu id -> pids
    using InstancesPerGpuMap = std::map<int32_t, std::set<int32_t>>;
    // Job id -> InstancesPerGpu
    std::map<std::string, InstancesPerGpuMap> jobInstancesPerGpu_;
    mutable std::mutex mutex_;

    void setOnDemandConfigForProcess(
        GpuProfilerResult& res,
        LibkinetoProcess& process,
        const std::string& config,
        int32_t configType,
        int32_t limit);

private:
    // Garbage collection and config refresh - periodically clean up
    // data from terminated processes.
    void start();
    void runGc();
    void refreshBaseConfig();

    std::string baseConfig_;
    std::thread managerThread_;
    std::atomic_bool stopFlag_{ false };
    std::condition_variable managerCondVar_;
    NpuStatus npuTraceStatus_;
    NpuStatus npuMonitorStatus_;
    int32_t registerCount_ = 0;
};

} // namespace dynolog
