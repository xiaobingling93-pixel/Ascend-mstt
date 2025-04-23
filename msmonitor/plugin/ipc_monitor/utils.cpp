#include "utils.h"
#include <glog/logging.h>
#include <glog/stl_logging.h>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <random>
#include <unordered_map>

namespace dynolog_npu {
namespace ipc_monitor {
std::unordered_map<SubModule, std::string> submoduleMap = {
    {SubModule::IPC, "IPC"},
};

std::unordered_map<ErrCode, std::string> errCodeMap = {
    {ErrCode::SUC, "success"},
    {ErrCode::PARAM, "invalid parameter"},
    {ErrCode::TYPE, "invalid type"},
    {ErrCode::VALUE, "invalid value"},
    {ErrCode::PTR, "invalid pointer"},
    {ErrCode::INTERNAL, "internal error"},
    {ErrCode::MEMORY, "memory error"},
    {ErrCode::NOT_SUPPORT, "feature not supported"},
    {ErrCode::NOT_FOUND, "resource not found"},
    {ErrCode::UNAVAIL, "resource unavailable"},
    {ErrCode::SYSCALL, "system call failed"},
    {ErrCode::TIMEOUT, "timeout error"},
    {ErrCode::PERMISSION, "permission error"},
};

std::string getCurrentTimestamp()
{
    auto now = std::chrono::system_clock::now();
    auto micros = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());

    std::time_t currentTime = std::chrono::system_clock::to_time_t(now);
    std::tm* timeInfo = std::localtime(&currentTime);

    auto milli_time = std::chrono::duration_cast<std::chrono::milliseconds>(micros).count() % 1000;
    auto micro_time = micros.count() % 1000;

    std::ostringstream oss;
    oss << std::put_time(timeInfo, "%Y-%m-%d-%H:%M:%S");
    return oss.str();
}

uint64_t getCurrentTimestamp64()
{
    auto now = std::chrono::system_clock::now();
    auto micros = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
    auto milli_time = std::chrono::duration_cast<std::chrono::milliseconds>(micros).count();
    return milli_time;
}

std::string formatErrorCode(SubModule submodule, ErrCode errorCode)
{
    std::ostringstream oss;
    oss << "\n[ERROR] " << getCurrentTimestamp() << " (PID:" << getpid() << ")";
    oss << "ERR" << std::setw(2) << std::setfill('0') << static_cast<int>(submodule); // 2: 字段宽度
    oss << std::setw(3) << std::setfill('0') << static_cast<int>(errorCode); // 3: 字段宽度
    oss << " " << submoduleMap[submodule] << " " << errCodeMap[errorCode];
    return oss.str();
};

int32_t GetProcessId()
{
    return static_cast<int32_t>(getpid());
}

std::pair<int32_t, std::string> GetParentPidAndCommand(int32_t pid)
{
    std::string fileName = "/proc/" + std::to_string(pid) + "/stat";
    std::ifstream statFile(fileName);
    if (!statFile) {
        return std::make_pair(0, "");
    }
    int32_t parentPid = 0;
    std::string command;
    std::string line;
    if (std::getline(statFile, line)) {
        int ret = sscanf(line.c_str(), "%*d (%[^)]) %*c %d", command.data(), &parentPid);
        if (ret == 2) { // 2: 接收到2个字符
            return std::make_pair(parentPid, command);
        }
    }
    LOG(WARNING) << "Failed to parse /proc/" << pid << "/stat";
    return std::make_pair(0, "");
}

std::vector<std::pair<int32_t, std::string>> GetPidCommandPairsofAncestors()
{
    std::vector<std::pair<int32_t, std::string>> process_pids_and_cmds;
    process_pids_and_cmds.reserve(MaxParentPids + 1);
    int32_t current_pid = GetProcessId();
    for (int i = 0; i <= MaxParentPids && (i == 0 || current_pid > 1); i++) {
        std::pair<int32_t, std::string> parent_pid_and_cmd = GetParentPidAndCommand(current_pid);
        process_pids_and_cmds.push_back(std::make_pair(current_pid, parent_pid_and_cmd.second));
        current_pid = parent_pid_and_cmd.first;
    }
    return process_pids_and_cmds;
}

std::vector<int32_t> GetPids()
{
    const auto &pids = GetPidCommandPairsofAncestors();
    std::vector<int32_t> res;
    res.reserve(pids.size());
    for (const auto &pidPair : pids) {
        res.push_back(pidPair.first);
    }
    LOG(INFO) << "Success to get parent pid: " << res;
    return res;
}

std::string GenerateUuidV4()
{
    static std::random_device randomDevice;
    static std::mt19937 gen(randomDevice());
    static std::uniform_int_distribution<> dis(0, 15);  // range (0, 15)
    static std::uniform_int_distribution<> dis2(8, 11); // range (8, 11)

    std::stringstream stringStream;
    stringStream << std::hex;
    for (int i = 0; i < 8; i++) {  // 8 times
        stringStream << dis(gen);
    }
    stringStream << "-";
    for (int j = 0; j < 4; j++) {  // 4 times
        stringStream << dis(gen);
    }
    stringStream << "-4"; // add -4
    for (int k = 0; k < 3; k++) { // 3 times
        stringStream << dis(gen);
    }
    stringStream << "-";
    stringStream << dis2(gen);
    for (int m = 0; m < 3; m++) { // 3 times
        stringStream << dis(gen);
    }
    stringStream << "-";
    for (int n = 0; n < 12; n++) { // 12 times
        stringStream << dis(gen);
    }
    return stringStream.str();
}

bool Str2Uint32(uint32_t& dest, const std::string& str)
{
    if (str.empty()) {
        LOG(ERROR) << "Str to uint32 failed, input string is null";
        return false;
    }
    size_t pos = 0;
    try {
        dest = static_cast<uint32_t>(std::stoul(str, &pos));
    } catch(...) {
        LOG(ERROR) << "Str to uint32 failed, input string is " << str;
        return false;
    }
    if (pos != str.size()) {
        LOG(ERROR) << "Str to uint32 failed, input string is " << str;
        return false;
    }
    return true;
}

bool Str2Bool(bool& dest, const std::string& str)
{
    std::string lower_str = str;
    std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(), ::tolower);

    if (lower_str == "true" || lower_str == "1") {
        dest = true;
        return true;
    }

    if (lower_str == "false" || lower_str == "0") {
        dest = false;
        return true;
    }
    LOG(ERROR) << "Str to bool failed, input string is " << str;
    return false;
}

std::string& trim(std::string& str)
{
    if (str.empty()) {
        return str;
    }
    str.erase(0, str.find_first_not_of(" "));
    str.erase(str.find_last_not_of(" ") + 1);
    return str;
}

// split函数
std::vector<std::string> split(const std::string& str, char delimiter)
{
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(str);

    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }

    return tokens;
}

void *MsptiMalloc(size_t size, size_t alignment)
{
    if (alignment > 0) {
        size = (size + alignment - 1) / alignment * alignment;
    }
#if defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
    void *ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        ptr = nullptr;
    }
    return ptr;
#else
    return malloc(size);
#endif
}

} // namespace ipc_monitor
} // namespace dynolog_npu
