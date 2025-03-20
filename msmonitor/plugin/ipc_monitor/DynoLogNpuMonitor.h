#ifndef DYNOLOG_NPU_MONITOR_H
#define DYNOLOG_NPU_MONITOR_H

#include "MonitorBase.h"
#include "NpuIpcClient.h"
#include "singleton.h"

namespace dynolog_npu {
namespace ipc_monitor {

class DynoLogNpuMonitor : public MonitorBase, public Singleton<DynoLogNpuMonitor> {
    friend class Singleton<DynoLogNpuMonitor>;

public:
    DynoLogNpuMonitor() = default;
    bool Init() override;
    std::string Poll() override;
    void SetNpuId(int id) override
    {
        npuId_ = id;
    }

private:
    bool isInitialized_ = false;
    int32_t npuId_ = 0;
    IpcClient ipcClient_;
};

} // namespace ipc_monitor
} // namespace dynolog_npu

#endif

