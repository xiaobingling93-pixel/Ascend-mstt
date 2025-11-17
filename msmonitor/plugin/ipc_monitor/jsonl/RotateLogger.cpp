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

#include "jsonl/RotateLogger.h"
#include <algorithm>
#include <filesystem>
#include <glog/logging.h>
#include "utils.h"

namespace fs = std::filesystem;

namespace dynolog_npu {
namespace ipc_monitor {
namespace jsonl {
namespace {
std::string GetMsmonitorJsonlName(const std::string &outputPath)
{
    auto identity = join({std::to_string(GetProcessId()), getCurrentTimestamp(), std::to_string(GetRankId())}, "_");
    return outputPath + "/msmonitor_" + identity + ".jsonl";
}
}

RotateLogger::~RotateLogger()
{
    UnInit();
}

void RotateLogger::UnInit()
{
    if (logFile_ != nullptr) {
        std::fclose(logFile_);
        logFile_ = nullptr;
    }
}

void RotateLogger::Log(std::string message)
{
    if (message.empty()) {
        LOG(WARNING) << "Empty message";
        return;
    }
    if (curLines_ >= maxLines_) {
        Rotate();
    }
    if (logFile_ == nullptr && !OpenNewFile()) {
        LOG(ERROR) << "RotateLogger open new log file failed";
        return;
    }
    std::fwrite(message.c_str(), sizeof(char), message.size(), logFile_);
    ++curLines_;
}

bool RotateLogger::OpenNewFile()
{
    if (logFile_ != nullptr) {
        std::fclose(logFile_);
        logFile_ = nullptr;
    }
    auto fileName = GetMsmonitorJsonlName(logDir_);
    if (!PathUtils::CreateFile(fileName)) {
        LOG(ERROR) << "RotateLogger create log file failed, path: " << fileName;
        return false;
    }
    logFile_ = std::fopen(fileName.c_str(), "ab");
    if (logFile_ == nullptr) {
        LOG(ERROR) << "RotateLogger open log file failed, path: " << fileName;
        return false;
    }
    curLines_ = 0;
    logFiles_.emplace_back(std::move(fileName));
    return true;
}

void RotateLogger::Rotate()
{
    if (logFile_ != nullptr) {
        std::fclose(logFile_);
        logFile_ = nullptr;
    }
    if (maxFiles_ > 0) {
        ManageFiles();
    }
    OpenNewFile();
}

void RotateLogger::ManageFiles()
{
    if (logFiles_.size() < maxFiles_) {
        return;
    }
    auto end = std::remove_if(logFiles_.begin(), logFiles_.end(), [](const std::string &file) {
        return !PathUtils::IsFileExist(file) || !PathUtils::IsOwner(file);
    });
    logFiles_.erase(end, logFiles_.end());
    std::sort(logFiles_.begin(), logFiles_.end(), [](const std::string &a, const std::string &b) {
        return fs::last_write_time(a) < fs::last_write_time(b);
    });
    int filesToRemove = logFiles_.size() - maxFiles_ + 1;
    for (auto it = logFiles_.begin(); it != logFiles_.begin() + filesToRemove; ++it) {
        std::error_code ec;
        if (!fs::remove(*it, ec)) {
            LOG(ERROR) << "RotateLogger remove log file failed, path: " << *it << ", error: " << ec.message();
        } else {
            LOG(INFO) << "RotateLogger remove log file, path: " << *it;
        }
    }
    logFiles_.erase(logFiles_.begin(), logFiles_.begin() + filesToRemove);
}
} // namespace jsonl
} // namespace ipc_monitor
} // namespace dynolog_npu
