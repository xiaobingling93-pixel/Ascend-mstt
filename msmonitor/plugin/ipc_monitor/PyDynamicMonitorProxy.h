#ifndef PYDYNAMIC_MONITOR_PROXY_H
#define PYDYNAMIC_MONITOR_PROXY_H

#include <glog/logging.h>
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
            monitor_ = DynoLogNpuMonitor::GetInstance();
            monitor_->SetNpuId(npuId);
            bool res = monitor_->Init();
            return res;
        } catch (const std::exception &e) {
            LOG(ERROR) << "Error when init dyno " << e.what();
            return false;
        }
    }

    std::string PollDyno()
    {
        return monitor_->Poll();
    }

    void EnableMsptiMonitor(std::unordered_map<std::string, std::string>& config_map)
    {
        DynoLogNpuMonitor::GetInstance()->EnableMsptiMonitor(config_map);
    }

    void FinalizeDyno()
    {
        DynoLogNpuMonitor::GetInstance()->Finalize();
    }
private:
    MonitorBase *monitor_ = nullptr;
};

} // namespace ipc_monitor
} // namespace dynolog_npu
#endif // PYDYNAMIC_MONITOR_PROXY_H
