#ifndef MONITOR_BASE_H
#define MONITOR_BASE_H

#include <string>

namespace dynolog_npu {
namespace ipc_monitor {

class MonitorBase {
public:
    virtual bool Init() = 0;
    virtual std::string Poll() = 0;
    virtual void SetNpuId(int id) = 0;
};

} // namespace ipc_monitor
} // namespace dynolog_npu
#endif // MONITOR_BASE_H
