#include "DynoLogNpuMonitor.h"

#include <iostream>

#include "utils.h"

namespace dynolog_npu {
namespace ipc_monitor {

bool DynoLogNpuMonitor::Init()
{
    if (isInitialized_) {
        std::cout << "[WRARNING] DynoLog npu monitor already initialized" << std::endl;
        return true;
    }
    bool res = ipcClient_.RegisterInstance(npuId_);
    if (res) {
        isInitialized_ = true;
        std::cout << "[INFO] DynoLog npu monitor initialized success !" << std::endl;
    }
    return res;
}

std::string DynoLogNpuMonitor::Poll()
{
    std::string res = ipcClient_.IpcClientNpuConfig();
    if (res.empty()) {
        std::cout << "[INFO] Request for dynolog server is empty !" << std::endl;
        return "";
    }
    std::cout << "[INFO] Received NPU configuration successfully" << std::endl;
    return res;
}

} // namespace ipc_monitor
} // namespace dynolog_npu