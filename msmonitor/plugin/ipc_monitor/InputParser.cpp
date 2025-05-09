#include "InputParser.h"
#include <unordered_set>
#include <unordered_map>
#include "utils.h"

namespace dynolog_npu {
namespace ipc_monitor {

const std::string MSPTI_ACTIVITY_KIND_KEY = "MSPTI_ACTIVITY_KIND";
const std::string REPORT_INTERVAL_S_KEY = "REPORT_INTERVAL_S";
const std::string NPU_MONITOR_START_KEY = "NPU_MONITOR_START";
const std::string NPU_MONITOR_STOP_KEY = "NPU_MONITOR_STOP";

const std::unordered_set<std::string> cfgMap {
    "MSPTI_ACTIVITY_KIND",
    "REPORT_INTERVAL_S",
    "NPU_MONITOR_START",
    "NPU_MONITOR_STOP",
    "REQUEST_TRACE_ID"
};

const std::unordered_map<std::string, msptiActivityKind> kindStrMap {
    {"Marker", MSPTI_ACTIVITY_KIND_MARKER},
    {"Kernel", MSPTI_ACTIVITY_KIND_KERNEL},
    {"API", MSPTI_ACTIVITY_KIND_API},
    {"Hccl", MSPTI_ACTIVITY_KIND_HCCL},
    {"Memory", MSPTI_ACTIVITY_KIND_MEMORY},
    {"MemSet", MSPTI_ACTIVITY_KIND_MEMSET},
    {"MemCpy", MSPTI_ACTIVITY_KIND_MEMCPY}
};

std::set<msptiActivityKind> str2Kinds(const std::string& kindStrs)
{
    std::set<msptiActivityKind> res;
    auto kindStrList = split(kindStrs, ',');
    for (auto& kindStr : kindStrList) {
        auto kind = kindStrMap.find(kindStr);
        if (kind == kindStrMap.end()) {
            return {MSPTI_ACTIVITY_KIND_INVALID};
        }
        res.insert(kind->second);
    }
    return res;
}

MsptiMonitorCfg InputParser::DynoLogGetOpts(std::unordered_map<std::string, std::string>& cmd)
{
    if (cmd.count("NPU_MONITOR_SRART")) {
        return {{MSPTI_ACTIVITY_KIND_INVALID}, 0, false, false, false};
    }
    auto activityKinds = str2Kinds(cmd[MSPTI_ACTIVITY_KIND_KEY]);
    uint32_t reportTimes = 0;
    Str2Uint32(reportTimes, cmd[REPORT_INTERVAL_S_KEY]);
    bool startSwitch = false;
    Str2Bool(startSwitch, cmd[NPU_MONITOR_START_KEY]);
    bool endSwitch = false;
    Str2Bool(endSwitch, cmd[NPU_MONITOR_STOP_KEY]);
    return {activityKinds, reportTimes, startSwitch, endSwitch, true};
}
}
}