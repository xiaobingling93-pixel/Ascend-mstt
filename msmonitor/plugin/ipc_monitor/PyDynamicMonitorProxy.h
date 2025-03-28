#ifndef PYDYNAMIC_MONITOR_PROXY_H
#define PYDYNAMIC_MONITOR_PROXY_H

#include <glog/logging.h>
#include <memory>
#include "MonitorBase.h"
#include "DynoLogNpuMonitor.h"

namespace dynolog_npu {
namespace ipc_monitor {

class PyDynamicMonitorProxy {
public:
    PyDynamicMonitorProxy() = default;
    bool InitDyno(int npuId)
    {
        try {
            if (!google::IsGoogleLoggingInitialized()) {
                google::InitGoogleLogging("DynoLogNpuMonitor");
                google::SetLogDestination(google::GLOG_INFO, "/var/log/dynolog_npu_");
                google::SetLogFilenameExtension(".log");
            }
            monitor_ = DynoLogNpuMonitor::GetInstance();
            monitor_->SetNpuId(npuId);
            bool res = monitor_->Init();
            LOG(ERROR) << res;
            return res;
        } catch (const std::exception &e) {
            LOG(ERROR) << "Error when init dyno " << e.what();
            return false;
        }
    }

    std::string PollDyno()
    {
        return monitor_->Poll();
    };

    void EnableMsptiMonitor(std::unordered_map<std::string, std::string>& config_map)
    {
        LOG(WARNING) << "EnableMsptiMonitor is not support now";
    }

    void FinalizeDyno()
    {
        LOG(WARNING) << "FinalizeDyno is not support now";
    }
private:
    MonitorBase *monitor_ = nullptr;
};

} // namespace ipc_monitor
} // namespace dynolog_npu

#endif
