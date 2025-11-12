// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <fmt/format.h>
#include <glog/logging.h>
#include <nlohmann/json.hpp>
#include <unordered_map>
#include "dynolog/src/rpc/SimpleJsonServer.h"
#include "dynolog/src/utils.h"

namespace dynolog {

const std::string CURRENT_STEP = "current_step";
const std::string START_STEP = "start_step";
const std::string STOP_STEP = "stop_step";

const std::unordered_map<int32_t, std::string> PROFILER_STATUS_MAP = {
    {-1, "Uninitialized"},
    {0, "Idle"},
    {1, "Running"},
    {2, "Ready"},
};

enum PROFILER_STATUS: int32_t {
    UNINITIALIZED = -1,
    IDLE = 0,
    RUNNING = 1,
    READY = 2,
};

template <class TServiceHandler = ServiceHandler>
class SimpleJsonServer : public SimpleJsonServerBase {
public:
    explicit SimpleJsonServer(std::shared_ptr<TServiceHandler> handler, int port)
        : SimpleJsonServerBase(port), handler_(std::move(handler)) {
    }

    ~SimpleJsonServer() {}

    std::string processOneImpl(const std::string& request) override;
    nlohmann::json handleSetKinetOnDemandRequest(const nlohmann::json& request);

private:
    std::shared_ptr<TServiceHandler> handler_;
};

/*
description: convert to json and validate the request message
notes: the request should contain:
    { "fn" : "<rpc_function>"
        .. <add other args>
    }
*/
inline nlohmann::json toJson(const std::string& message)
{
    using json = nlohmann::json;
    json result;
    if (message.empty()) {
        return result;
    }
    try {
        if (!(json::accept(message) && CheckJsonDepth(message))) {
            LOG(ERROR) << "Error parsing message = " << message;
            return json();
        }
        result = json::parse(message);
    }
    catch (const std::exception& e) {
        LOG(ERROR) << "Error parsing message = " << message << " : " << e.what();
        return json();
    }

    if (result.empty() || !result.is_object()) {
        LOG(ERROR)
            << "Request message should not be empty and should be json object.";
        return json();
    }

    if (!result.contains("fn")) {
        LOG(ERROR) << "Request must contain a 'fn' field for the RPC call "
            << " request json = " << result.dump();
        return json();
    }

    return result;
}

inline std::string GetCommandStatus(const std::string& configStr)
{
    auto npuTraceStatus = LibkinetoConfigManager::getInstance()->getNpuTraceStatus();
    auto npuMonitorStatus = LibkinetoConfigManager::getInstance()->getNpuMonitorStatus();
    std::string prefix = "NPU_MONITOR_START";
    if (configStr.compare(0, prefix.size(), prefix) == 0) {
        if (npuTraceStatus.status == PROFILER_STATUS::RUNNING || npuMonitorStatus.status == PROFILER_STATUS::UNINITIALIZED) {
            return "ineffective";
        } else if (npuTraceStatus.status == PROFILER_STATUS::IDLE || npuTraceStatus.status == PROFILER_STATUS::READY) {
            return "effective";
        } else {
            return "unknown";
        }
    } else {
        if (npuMonitorStatus.status == PROFILER_STATUS::RUNNING || npuTraceStatus.status == PROFILER_STATUS::UNINITIALIZED) {
            return "ineffective";
        } else if (npuMonitorStatus.status == PROFILER_STATUS::IDLE) {
            return "effective";
        } else {
            return "unknown";
        }
    }
}

inline nlohmann::json GetStatus()
{
    using json = nlohmann::json;
    json response;
    response["nputrace"] = "unknown";
    response["npumonitor"] = "unknown";
    auto npuTraceStatus = LibkinetoConfigManager::getInstance()->getNpuTraceStatus();
    auto npuMonitorStatus = LibkinetoConfigManager::getInstance()->getNpuMonitorStatus();
    auto it = PROFILER_STATUS_MAP.find(npuTraceStatus.status);
    if (it != PROFILER_STATUS_MAP.end()) {
        response["nputrace"] = it->second;
    }
    it = PROFILER_STATUS_MAP.find(npuMonitorStatus.status);
    if (it != PROFILER_STATUS_MAP.end()) {
        response["npumonitor"] = it->second;
    }
    response[CURRENT_STEP] = npuTraceStatus.currentStep;
    if (npuTraceStatus.status == PROFILER_STATUS::RUNNING || npuTraceStatus.status == PROFILER_STATUS::READY) {
        response[START_STEP] = npuTraceStatus.startStep;
        response[STOP_STEP] = npuTraceStatus.stopStep;
    }
    return response;
}

template <class TServiceHandler>
nlohmann::json SimpleJsonServer<TServiceHandler>::handleSetKinetOnDemandRequest(const nlohmann::json& request)
{
    using json = nlohmann::json;
    json response;
    if (!request.contains("config") || !request.contains("pids")) {
        response["status"] = "failed";
        return response;
    }
    try {
        std::string config = request.value("config", "");
        std::vector<int> pids = request.at("pids").get<std::vector<int>>();
        std::set<int> pids_set{ pids.begin(), pids.end() };
        int job_id = request.value("job_id", 0);
        int process_limit = request.value("process_limit", 1000);
        auto commandStatus = GetCommandStatus(config);
        if (commandStatus == "effective") {
            auto result = handler_->setKinetOnDemandRequest(job_id, pids_set, config, process_limit);
            if (result.processesMatched.empty()) {
                commandStatus = "ineffective";
            } else {
                response["processesMatched"] = result.processesMatched;
                response["eventProfilersTriggered"] = result.eventProfilersTriggered;
                response["activityProfilersTriggered"] = result.activityProfilersTriggered;
                response["eventProfilersBusy"] = result.eventProfilersBusy;
                response["activityProfilersBusy"] = result.activityProfilersBusy;
            }
        }
        response["commandStatus"] = commandStatus;
    }
    catch (const std::exception& ex) {
        LOG(ERROR) << "setKinetOnDemandRequest: parsing exception = " << ex.what();
        response["status"] = fmt::format("failed with exception = {}", ex.what());
    }
    return response;
}


template <class TServiceHandler>
std::string SimpleJsonServer<TServiceHandler>::processOneImpl(const std::string& request_str)
{
    using json = nlohmann::json;
    json request = toJson(request_str);
    json response;

    if (request.empty()) {
        LOG(ERROR) << "Failed parsing request, continuing ...";
        return "";
    }

    if (request["fn"] == "getStatus") {
        response = GetStatus();
    } else if (request["fn"] == "getVersion") {
        response["version"] = handler_->getVersion();
    } else if (request["fn"] == "setKinetOnDemandRequest") {
        response = handleSetKinetOnDemandRequest(request);
    } else {
        LOG(ERROR) << "Unknown RPC call = " << request["fn"];
        return "";
    }

    return response.dump();
}
} // namespace dynolog
