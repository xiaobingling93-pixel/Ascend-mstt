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

#include "dynolog/src/utils.h"

#include <unistd.h>
#include <climits>
#include <cstdint>
#include <cstring>
#include <sys/stat.h>
#include <unordered_map>
#include <glog/logging.h>

namespace dynolog {

namespace {
const uint16_t MAX_PATH_SIZE = 1024;
constexpr int INPUT_DIR_CHECK_MODE = R_OK | X_OK;
constexpr mode_t OTHERS_WRITE_MASK = S_IWGRP | S_IWOTH;
const std::unordered_map<std::string, std::string> INVALID_CHAR = {
    {"\n", "\\n"}, {"\f", "\\f"}, {"\r", "\\r"}, {"\b", "\\b"}, {"\t", "\\t"},
    {"\v", "\\v"}, {"\u007F", "\\u007F"}, {"\"", "\\\""}, {"'", "\'"},
    {"\\", "\\\\"}, {"%", "\\%"}, {">", "\\>"}, {"<", "\\<"}, {"|", "\\|"},
    {"&", "\\&"}, {"$", "\\$"}
};
}

std::string Rstrip(const std::string &str1, const std::string &str2)
{
    size_t end = str1.find_last_not_of(str2);
    return end == std::string::npos ? "" : str1.substr(0, end + 1);
}

bool PathUtils::Access(const std::string &path, const int &mode)
{
    if (path.empty()) {
        LOG(ERROR) << "The file path is empty.";
        return false;
    }
    return access(path.c_str(), mode) == 0;
}

bool PathUtils::Exist(const std::string &path)
{
    return Access(path, F_OK);
}

bool PathUtils::IsSoftLink(const std::string &path)
{
    if (path.empty()) {
        LOG(ERROR) << "The file path is empty.";
        return false;
    }
    std::string tmpPath = Rstrip(path, "./");
    struct stat fileStat;
    if (lstat(tmpPath.c_str(), &fileStat) != 0) {
        LOG(ERROR) << "The file stat failed, path: " << path;
        return false;
    }
    return S_ISLNK(fileStat.st_mode);
}

bool PathUtils::IsFile(const std::string &path)
{
    if (path.empty()) {
        LOG(ERROR) << "The file path is empty.";
        return false;
    }
    struct stat fileStat;
    if (stat(path.c_str(), &fileStat) != 0) {
        LOG(ERROR) << "The file stat failed, path: " << path;
        return false;
    }
    return fileStat.st_mode & S_IFREG;
}

bool PathUtils::IsWritableByOthers(const std::string &path)
{
    struct stat fileStat;
    if (stat(path.c_str(), &fileStat) != 0) {
        LOG(ERROR) << "The file stat failed, path: " << path;
        return true;
    }
    return (fileStat.st_mode & OTHERS_WRITE_MASK) != 0;
}

bool PathUtils::IsOwner(const std::string &path)
{
    struct stat info;
    if (stat(path.c_str(), &info) != 0) {
        LOG(ERROR) << "The file stat failed, path: " << path;
        return false;
    }
    uid_t current_uid = getuid();
    if (info.st_uid != current_uid) {
        return false;
    }
    return true;
}


bool PathUtils::DirPathCheck(const std::string &path)
{
    if (path.empty()) {
        LOG(ERROR) << "The path is empty.";
        return false;
    }
    if (path.size() > MAX_PATH_SIZE) {
        LOG(ERROR) << "The length of path is too long, max allowed: " << MAX_PATH_SIZE;
        return false;
    }
    for (auto &item : INVALID_CHAR) {
        if (path.find(item.first) != std::string::npos) {
            LOG(ERROR) << "The path contains invalid character: " << item.first;
            return false;
        }
    }
    if (!Exist(path)) {
        LOG(ERROR) << "The path does not exist: " << path;
        return false;
    }
    if (IsFile(path)) {
        LOG(ERROR) << "The path is a file: " << path;
        return false;
    }
    if (IsSoftLink(path)) {
        LOG(ERROR) << "The path is a soft link: " << path;
        return false;
    }
    if (IsRoot()) {
        return true;
    }
    if (!IsOwner(path)) {
        LOG(ERROR) << "The path is not owned by current user: " << path;
        return false;
    }
    if (!Access(path, INPUT_DIR_CHECK_MODE)) {
        LOG(ERROR) << "The path is not readable: " << path;
        return false;
    }
    if (IsWritableByOthers(path)) {
        LOG(ERROR) << "The path is writable by others: " << path;
        return false;
    }
    return true;
}

bool IsRoot()
{
    return getuid() == 0;
}
}
