#ifndef INPUT_PARSER_H
#define INPUT_PARSER_H

#include <unordered_map>
#include <singleton.h>
#include <set>
#include <mspti.h>

namespace dynolog_npu {
namespace ipc_monitor {

struct MsptiMonitorCfg
{
    std::set<msptiActivityKind> enableActivities;
    uint32_t reportIntervals;
    bool monitorStart;
    bool monitorStop;
    bool isMonitor;
};


class InputParser: public dynolog_npu::ipc_monitor::Singleton<InputParser> {
public:
    MsptiMonitorCfg DynoLogGetOpts(std::unordered_map<std::string, std::string>& cmd);
};

} // namespace ipc_monitor
} // namespace dynolog_npu

#endif