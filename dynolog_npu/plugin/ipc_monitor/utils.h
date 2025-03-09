#ifndef IPC_MONITOR_UTILS_H
#define IPC_MONITOR_UTILS_H
#include <sys/types.h>
#include <unistd.h>
#include <cstdint>
#include <vector>
#include <string>
#include <random>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <glog/logging.h>
#include <unordered_map>


namespace dynolog_npu {
namespace ipc_monitor {

constexpr int MaxParentPids = 5;
int32_t GetProcessId();
std::string GenerateUuidV4();
std::vector<int32_t> GetPids();
std::pair<int32_t, std::string> GetParentPidAndCommand(int32_t pid);
std::vector<std::pair<int32_t, std::string>> GetPidCommandPairsofAncestors();
std::string getCurrentTimestamp();

enum class SubModule {
    IPC = 0
};

enum class ErrCode {
    SUC = 0,
    PARAM = 1,
    TYPE = 2,
    VALUE = 3,
    PTR = 4,
    INTERNAL = 5,
    MEMORY = 6,
    NOT_SUPPORT = 7,
    NOT_FOUND = 8,
    UNAVAIL = 9,
    SYSCALL = 10,
    TIMEOUT = 11,
    PERMISSION = 12,
};


std::string formatErrorCode(SubModule submodule, ErrCode errorCode);

#define IPC_ERROR(error) formatErrorCode(SubModule::IPC, error)

template<typename T, typename V>
inline T ReinterpretConvert(V ptr) {
    return reinterpret_cast<T>(ptr);
}


} // namespace ipc_monitor
} // namespace dynolog_npu

#endif

