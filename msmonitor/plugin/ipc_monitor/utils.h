/*
 * Copyright (C) 2025-2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef IPC_MONITOR_UTILS_H
#define IPC_MONITOR_UTILS_H

#include <cstdint>
#include <vector>
#include <string>
#include <utility>
#include <unordered_map>
#include <sys/types.h>
#include <memory>

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
void MsptiFree(uint8_t *ptr);
const mode_t DATA_FILE_AUTHORITY = 0640;
const mode_t DATA_DIR_AUTHORITY = 0750;
const int DEFAULT_FLUSH_INTERVAL = 60;

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
inline T ReinterpretConvert(V ptr)
{
    return reinterpret_cast<T>(ptr);
}

template<typename Types, typename... Args>
inline void MakeSharedPtr(std::shared_ptr<Types>& ptr, Args&&... args)
{
    try {
        ptr = std::make_shared<Types>(std::forward<Args>(args)...);
    } catch(std::bad_alloc& e) {
        throw;
    } catch (...) {
        ptr = nullptr;
        return;
    }
}

template <typename Container, typename KeyFunc>
auto groupby(const Container& vec, KeyFunc keyFunc)
{
    using KeyType = decltype(keyFunc(*vec.begin()));
    using ValueType = typename Container::value_type;
    std::unordered_map<KeyType, std::vector<ValueType>> grouped;
    for (const auto& item : vec) {
        grouped[keyFunc(item)].push_back(item);
    }
    return grouped;
}

bool CreateMsmonitorLogPath(std::string& path);

struct PathUtils {
    static bool IsFileExist(const std::string &path);
    static bool IsFileWritable(const std::string &path);
    static bool IsDir(const std::string &path);
    static bool CreateDir(const std::string &path);
    static std::string RealPath(const std::string &path);
    static std::string RelativeToAbsPath(const std::string &path);
    static std::string DirName(const std::string &path);
    static bool CreateFile(const std::string &path);
    static bool IsSoftLink(const std::string &path);
    static bool DirPathCheck(const std::string &path);
};
} // namespace ipc_monitor
} // namespace dynolog_npu
#endif // IPC_MONITOR_UTILS_H
