#ifndef PYDYNAMIC_MONITOR_PROXY_H
#define PYDYNAMIC_MONITOR_PROXY_H

#include <iostream>
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
             monitor_ = DynoLogNpuMonitor::GetInstance();
             monitor_->SetNpuId(npuId);
             bool res = monitor_->Init();
             return res;
         } catch (const std::exception &e) {
             std::cout << "[ERROR] Error when init dyno " << e.what() << std::endl;
             return false;
         }
    }

    std::string PollDyno()
    {
         return monitor_->Poll();
    };

private:
    MonitorBase *monitor_ = nullptr;
};

} // namespace ipc_monitor
} // namespace dynolog_npu

#endif
