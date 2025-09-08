// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <fmt/format.h>
#include <glog/logging.h>
#include <nlohmann/json.hpp>
#include "dynolog/src/rpc/SimpleJsonServer.h"

namespace dynolog {

template <class TServiceHandler = ServiceHandler>
class SimpleJsonServer : public SimpleJsonServerBase {
 public:
  explicit SimpleJsonServer(std::shared_ptr<TServiceHandler> handler, int port)
      : SimpleJsonServerBase(port), handler_(std::move(handler)) {}

  ~SimpleJsonServer() {}

  std::string processOneImpl(const std::string& request) override;
  nlohmann::json handleSetKinetOnDemandRequest(const nlohmann::json& request);

 private:
  std::shared_ptr<TServiceHandler> handler_;
};

// convert to json and validate the request message
// the request should contain :
//   { "fn" : "<rpc_function>"
//    .. <add other args>
//   }

nlohmann::json toJson(const std::string& message) {
  using json = nlohmann::json;
  json result;
  if (message.empty()) {
    return result;
  }
  try {
    result = json::parse(message);
  } catch (json::parse_error&) {
    LOG(ERROR) << "Error parsing message = " << message;
    return result;
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

std::string GetCommandStatus(const std::string& configStr)
{
    auto npuTraceStatus = LibkinetoConfigManager::getInstance()->getNpuTraceStatus();
    auto npuMonitorStatus = LibkinetoConfigManager::getInstance()->getNpuMonitorStatus();
    std::string prefix = "NPU_MONITOR_START";
    if (configStr.compare(0, prefix.size(), prefix) == 0) {
        if (npuTraceStatus == 1) {
            return "ineffective";
        }
        else if (npuTraceStatus == 0) {
            return "effective";
        }
        else {
            return "unknown";
        }
    } else {
        if (npuMonitorStatus == 1) {
            return "ineffective";
        }
        else if (npuMonitorStatus == 0) {
            return "effective";
        }
        else {
            return "unknown";
        }
    }
}

template <class TServiceHandler>
nlohmann::json SimpleJsonServer<TServiceHandler>::handleSetKinetOnDemandRequest(const nlohmann::json& request) {
  using json = nlohmann::json;
  json response;
  if (!request.contains("config") || !request.contains("pids")) {
    response["status"] = "failed";
    return response;
  }
  try {
    std::string config = request.value("config", "");
    std::vector<int> pids = request.at("pids").get<std::vector<int>>();
    std::set<int> pids_set{pids.begin(), pids.end()};
    int job_id = request.value("job_id", 0);
    int process_limit = request.value("process_limit", 1000);
    auto commandStatus = GetCommandStatus(config);
    if (commandStatus == "effective") {
      auto result = handler_->setKinetOnDemandRequest(job_id, pids_set, config, process_limit);
      response["processesMatched"] = result.processesMatched;
      response["eventProfilersTriggered"] = result.eventProfilersTriggered;
      response["activityProfilersTriggered"] = result.activityProfilersTriggered;
      response["eventProfilersBusy"] = result.eventProfilersBusy;
      response["activityProfilersBusy"] = result.activityProfilersBusy;
    }
    response["commandStatus"] = commandStatus;
  } catch (const std::exception& ex) {
    LOG(ERROR) << "setKinetOnDemandRequest: parsing exception = " << ex.what();
    response["status"] = fmt::format("failed with exception = {}", ex.what());
  }
  return response;
}


template <class TServiceHandler>
std::string SimpleJsonServer<TServiceHandler>::processOneImpl(
    const std::string& request_str) {
  using json = nlohmann::json;
  json request = toJson(request_str);
  json response;

  if (request.empty()) {
    LOG(ERROR) << "Failed parsing request, continuing ...";
    return "";
  }

  if (request["fn"] == "getStatus") {
    response["status"] = handler_->getStatus();
  } else if (request["fn"] == "getVersion") {
    response["version"] = handler_->getVersion();
  } else if (request["fn"] == "setKinetOnDemandRequest") {
    response = handleSetKinetOnDemandRequest(request);
  } else if (request["fn"] == "dcgmProfPause") {
    if (!request.contains("duration_s")) {
      response["status"] = "failed";
    } else {
      int duration_s = request.value("duration_s", 300);
      bool result = handler_->dcgmProfPause(duration_s);
      response["status"] = result;
    }
  } else if (request["fn"] == "dcgmProfResume") {
    bool result = handler_->dcgmProfResume();
    response["status"] = result;
  } else {
    LOG(ERROR) << "Unknown RPC call = " << request["fn"];
    return "";
  }

  return response.dump();
}

} // namespace dynolog
