#ifndef DYNOLOG_NPU_MONITOR_H
#define DYNOLOG_NPU_MONITOR_H

#include "MonitorBase.h"
#include "NpuIpcClient.h"
#include "MsptiMonitor.h"
#include "singleton.h"
#include "InputParser.h"

namespace dynolog_npu {
namespace ipc_monitor {

class DynoLogNpuMonitor : public MonitorBase, public Singleton<DynoLogNpuMonitor> {
    friend class Singleton<DynoLogNpuMonitor>;

public:
    DynoLogNpuMonitor();
    bool Init() override;
    ErrCode DealMonitorReq(const MsptiMonitorCfg& cmd);
    std::string Poll() override;
    void EnableMsptiMonitor(std::unordered_map<std::string, std::string>& cfg_map);
    void Finalize();
    void SetNpuId(int id) override
    {
        npuId_ = id;
    }

    IpcClient *GetIpcClient()
    {
        return &ipcClient_;
    }

private:
    bool isInitialized_ = false;
    int32_t npuId_ = 0;
    IpcClient ipcClient_;
    MsptiMonitor msptiMonitor_;
};

} // namespace ipc_monitor
} // namespace dynolog_npu

#endif // DYNOLOG_NPU_MONITOR_H
