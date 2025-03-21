#include "DynoLogNpuMonitor.h"
#include "utils.h"

namespace dynolog_npu {
namespace ipc_monitor {

bool DynoLogNpuMonitor::Init()
{
    if (isInitialized_) {
        LOG(ERROR) << "DynoLog npu monitor already initialized";
        return true;
    }
    bool res = ipcClient_.RegisterInstance(npuId_);
    if (res) {
        isInitialized_ = true;
        LOG(INFO) << "DynoLog npu monitor initialized success!";
    }
    return res;
}

std::string DynoLogNpuMonitor::Poll()
{
    std::string res = ipcClient_.IpcClientNpuConfig();
    return res;
}

} // namespace ipc_monitor
} // namespace dynolog_npu