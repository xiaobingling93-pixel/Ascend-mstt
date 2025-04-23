#ifndef IPC_MONITOR_UTILS_H
#define IPC_MONITOR_UTILS_H

#include <cstdint>
#include <vector>
#include <string>
#include <utility>
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
uint64_t getCurrentTimestamp64();
bool Str2Uint32(uint32_t& dest, const std::string& str);
bool Str2Bool(bool& dest, const std::string& str);
std::string& trim(std::string& str);
std::vector<std::string> split(const std::string& str, char delimiter);

constexpr size_t ALIGN_SIZE = 8;
void *MsptiMalloc(size_t size, size_t alignment);

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
template <typename Container, typename KeyFunc>
auto groupby(const Container& vec, KeyFunc keyFunc) {
    using KeyType = decltype(keyFunc(*vec.begin()));
    using ValueType = typename Container::value_type;
    std::unordered_map<KeyType, std::vector<ValueType>> grouped;
    for (const auto& item : vec) {
        grouped[keyFunc(item)].push_back(item);
    }
    return grouped;
}
} // namespace ipc_monitor
} // namespace dynolog_npu
#endif // IPC_MONITOR_UTILS_H
