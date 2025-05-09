#ifndef METRIC_PROCESS_BASE_H
#define METRIC_PROCESS_BASE_H

#include <nlohmann/json.hpp>
#include <glog/logging.h>

#include "DynoLogNpuMonitor.h"
#include "NpuIpcClient.h"
#include "mspti.h"

namespace dynolog_npu {
namespace ipc_monitor {
namespace metric {
class MetricProcessBase
{
public:
    void SendMessage(std::string message)
    {
        if (message.empty()) {
            LOG(ERROR) << "SendMessage message is empty";
            return;
        }
        static const std::string destName = DYNO_IPC_NAME + "_data";
        static const int maxRetry = 5, retryWaitTimeUs = 1000;
        auto msg = Message::ConstructStrMessage(message, MSG_TYPE_DATA);
        if (!msg) {
            LOG(ERROR) << "ConstructStrMessage failed, message: " << message;
            return;
        }
        auto ipcClient = DynoLogNpuMonitor::GetInstance()->GetIpcClient();
        if (!ipcClient) {
            LOG(ERROR) << "DynoLogNpuMonitor ipcClient is nullptr";
            return;
        }
        if (!ipcClient->SyncSendMessage(*msg, destName, maxRetry, retryWaitTimeUs)) {
            LOG(ERROR) << "send mspti message failed: " << message;
        }
    }
    virtual void ConsumeMsptiData(msptiActivity *record) = 0;
    virtual void Clear() = 0;
    virtual void SendProcessMessage() = 0;
};
}
}
}
#endif