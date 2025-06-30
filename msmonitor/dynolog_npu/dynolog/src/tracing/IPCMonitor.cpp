// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "dynolog/src/tracing/IPCMonitor.h"
#include <glog/logging.h>
#include <nlohmann/json.hpp>
#include <cstring>
#include <unistd.h>
#include <cstdint>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "dynolog/src/LibkinetoConfigManager.h"
#include "dynolog/src/ipcfabric/Utils.h"

namespace dynolog {
namespace tracing {

constexpr int kSleepUs = 10000;
constexpr int kDataMsgSleepUs = 1000;
const std::string kLibkinetoRequest = "req";
const std::string kLibkinetoContext = "ctxt";
const std::string kLibkinetoData = "data";

IPCMonitor::IPCMonitor(const std::string& ipc_fabric_name)
{
    ipc_manager_ = FabricManager::factory(ipc_fabric_name);
    data_ipc_manager_ = FabricManager::factory(ipc_fabric_name + "_data");
    // below ensures singleton exists
    LOG(INFO) << "Kineto config manager : active processes = "
        << LibkinetoConfigManager::getInstance()->processCount("0");
}

void IPCMonitor::loop()
{
    while (ipc_manager_) {
        if (ipc_manager_->recv()) {
            std::unique_ptr<ipcfabric::Message> msg = ipc_manager_->retrieve_msg();
            processMsg(std::move(msg));
        }
        /* sleep override */
        usleep(kSleepUs);
    }
}

void IPCMonitor::dataLoop()
{
    while (data_ipc_manager_) {
        if (data_ipc_manager_->recv()) {
            std::unique_ptr<ipcfabric::Message> msg = data_ipc_manager_->retrieve_msg();
            processDataMsg(std::move(msg));
        }
        /* sleep override */
        usleep(kDataMsgSleepUs);
    }
}

void IPCMonitor::processMsg(std::unique_ptr<ipcfabric::Message> msg)
{
    if (!ipc_manager_) {
        LOG(ERROR) << "Fabric Manager not initialized";
        return;
    }
    // sizeof(msg->metadata.type) = 32, well above the size of the constant
    // strings we are comparing against. memcmp is safe
    if (memcmp( // NOLINT(facebook-security-vulnerable-memcmp)
        msg->metadata.type,
        kLibkinetoContext.data(),
        kLibkinetoContext.size()) == 0) {
        registerLibkinetoContext(std::move(msg));
    } else if (
        memcmp( // NOLINT(facebook-security-vulnerable-memcmp)
            msg->metadata.type,
            kLibkinetoRequest.data(),
            kLibkinetoRequest.size()) == 0) {
        getLibkinetoOnDemandRequest(std::move(msg));
    } else {
        LOG(ERROR) << "TYPE UNKOWN: " << msg->metadata.type;
    }
}

void tracing::IPCMonitor::setLogger(std::unique_ptr<Logger> logger)
{
    logger_ = std::move(logger);
}

void IPCMonitor::LogData(const nlohmann::json& result)
{
    auto timestamp = result["timestamp"].get<uint64_t>();
    logger_->logUint("timestamp", timestamp);
    auto duration = result["duration"].get<uint64_t>();
    logger_->logUint("duration", duration);
    auto deviceId = result["deviceId"].get<std::uint64_t>();
    logger_->logStr("deviceId", std::to_string(deviceId));
    auto kind = result["kind"].get<std::string>();
    logger_->logStr("kind", kind);
    if (result.contains("domain") && result["domain"].is_string()) {
        auto domain = result["domain"].get<std::string>();
        logger_->logStr("domain", domain);
    }
    logger_->finalize();
}

void IPCMonitor::processDataMsg(std::unique_ptr<ipcfabric::Message> msg)
{
    if (!data_ipc_manager_) {
        LOG(ERROR) << "Fabric Manager not initialized";
        return;
    }
    if (memcmp( // NOLINT(facebook-security-vulnerable-memcmp)
        msg->metadata.type,
        kLibkinetoData.data(),
        kLibkinetoData.size()) == 0) {
        std::string message = std::string((char*)msg->buf.get(), msg->metadata.size);
        try {
            nlohmann::json result = nlohmann::json::parse(message);
            LOG(INFO) << "Received data message : " << result;
            LogData(result);
        } catch (nlohmann::json::parse_error&) {
        LOG(ERROR) << "Error parsing message = " << message;
        return;
        }
    } else {
        LOG(ERROR) << "TYPE UNKOWN: " << msg->metadata.type;
    }
}

void IPCMonitor::getLibkinetoOnDemandRequest(
    std::unique_ptr<ipcfabric::Message> msg)
{
    if (!ipc_manager_) {
        LOG(ERROR) << "Fabric Manager not initialized";
        return;
    }
    std::string ret_config = "";
    ipcfabric::LibkinetoRequest* req =
        (ipcfabric::LibkinetoRequest*)msg->buf.get();
    if (req->n == 0) {
        LOG(ERROR) << "Missing pids parameter for type " << req->type;
        return;
    }
    std::vector<int32_t> pids(req->pids, req->pids + req->n);
    try {
        ret_config = LibkinetoConfigManager::getInstance()->obtainOnDemandConfig(
            std::to_string(req->jobid), pids, req->type);
    } catch (const std::runtime_error& ex) {
        LOG(ERROR) << "Kineto config manager exception : " << ex.what();
    }
    std::unique_ptr<ipcfabric::Message> ret =
        ipcfabric::Message::constructMessage<decltype(ret_config)>(
            ret_config, kLibkinetoRequest);
    if (!ipc_manager_->sync_send(*ret, msg->src)) {
        LOG(ERROR) << "Failed to return config to libkineto: IPC sync_send fail";
    }
    return;
}

void IPCMonitor::registerLibkinetoContext(
    std::unique_ptr<ipcfabric::Message> msg)
{
    if (!ipc_manager_) {
        LOG(ERROR) << "Fabric Manager not initialized";
        return;
    }
    ipcfabric::LibkinetoContext* ctxt =
        (ipcfabric::LibkinetoContext*)msg->buf.get();
    int32_t size = -1;
    try {
        size = LibkinetoConfigManager::getInstance()->registerLibkinetoContext(
            std::to_string(ctxt->jobid), ctxt->pid, ctxt->gpu);
    } catch (const std::runtime_error& ex) {
    LOG(ERROR) << "Kineto config manager exception : " << ex.what();
    }
    std::unique_ptr<ipcfabric::Message> ret =
        ipcfabric::Message::constructMessage<decltype(size)>(
            size, kLibkinetoContext);
    if (!ipc_manager_->sync_send(*ret, msg->src)) {
        LOG(ERROR) << "Failed to send ctxt from dyno: IPC sync_send fail";
    }
    return;
}

} // namespace tracing
} // namespace dynolog
