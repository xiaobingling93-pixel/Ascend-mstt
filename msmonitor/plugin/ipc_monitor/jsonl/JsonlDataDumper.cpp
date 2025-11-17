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

#include "jsonl/JsonlDataDumper.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include "utils.h"

namespace dynolog_npu {
namespace ipc_monitor {
namespace jsonl {
namespace {
uint32_t GetRotateLogLines()
{
    constexpr uint32_t DEFAULT_LINES = 10000;
    constexpr uint32_t MAX_LINES = 500000;
    constexpr uint32_t MIN_LINES = 100;
    const char* linesEnvVal = std::getenv("MSMONITOR_JSONL_ROTATE_LOG_LINES");
    std::string linesStr = (linesEnvVal != nullptr ? linesEnvVal : "");
    uint32_t lines = DEFAULT_LINES;
    if (!linesStr.empty()) {
        if (Str2Uint32(lines, linesStr)) {
            lines = std::clamp(lines, MIN_LINES, MAX_LINES);
            return lines;
        } else {
            LOG(WARNING) << "Jsonl GetRotateLogLines invalid lines: " << linesStr
                         << ", use default lines: " << DEFAULT_LINES;
        }
    }
    return DEFAULT_LINES;
}

int32_t GetRotateLogMaxFiles()
{
    constexpr int32_t DEFAULT_NOT_ROTATE = -1;
    constexpr int32_t MIN_ROTATE_FILES = 2;
    const char* maxFilesEnvVal = std::getenv("MSMONITOR_JSONL_ROTATE_LOG_FILES");
    if (maxFilesEnvVal == nullptr) {
        return DEFAULT_NOT_ROTATE;
    }
    std::string maxFilesStr = maxFilesEnvVal;
    int32_t maxFiles = DEFAULT_NOT_ROTATE;
    if (!maxFilesStr.empty()) {
        if (Str2Int32(maxFiles, maxFilesStr) && maxFiles >= MIN_ROTATE_FILES) {
            return maxFiles;
        } else {
            LOG(WARNING) << "Jsonl GetRotateLogMaxFiles invalid maxFiles: " << maxFilesStr
                         << ", rotate log maxFiles must >= " << MIN_ROTATE_FILES;
        }
    }
    return DEFAULT_NOT_ROTATE;
}
}

void JsonlDataDumper::Init(const std::string &dirPath, size_t capacity, uint32_t maxDumpIntervalMs)
{
    dumpDir_ = dirPath;
    dataBuf_.Init(capacity);
    init_.store(true);
    maxDumpIntervalMs_ = std::chrono::milliseconds(maxDumpIntervalMs);
    lastDumpTime_ = std::chrono::steady_clock::now();
    auto logLines = GetRotateLogLines();
    auto maxFiles = GetRotateLogMaxFiles();
    rotateLogger_ = std::make_unique<RotateLogger>(dumpDir_, logLines, maxFiles);
    LOG(INFO) << "JsonlDataDumper Init, logLines: " << logLines
              << (maxFiles < 0 ? " not rotate" : ", maxFiles: " + std::to_string(maxFiles));
}

void JsonlDataDumper::UnInit()
{
    if (init_.load()) {
        dataBuf_.UnInit();
        init_.store(false);
        start_.store(false);
    }
    if (rotateLogger_ != nullptr) {
        rotateLogger_->UnInit();
    }
}

void JsonlDataDumper::Start()
{
    if (!init_.load() || Thread::Start() != 0) {
        return;
    }
    start_.store(true);
}

void JsonlDataDumper::Stop()
{
    if (start_.load() == true) {
        start_.store(false);
        Thread::Stop();
    }
    Flush();
}

void JsonlDataDumper::DumpData()
{
    uint32_t batchSize = 0;
    while (batchSize < kNotifyInterval) {
        std::unique_ptr<nlohmann::json> json = nullptr;
        if (!dataBuf_.Pop(json) || json == nullptr) {
            break;
        }
        std::string encodeData = json->dump() + "\n";
        rotateLogger_->Log(encodeData);
        ++batchSize;
    }
    lastDumpTime_ = std::chrono::steady_clock::now();
}

void JsonlDataDumper::Run()
{
    while (true) {
        if (!start_.load()) {
            break;
        }
        if (dataBuf_.Size() > kNotifyInterval) {
            DumpData();
        } else {
            auto curTime = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(curTime - lastDumpTime_) >= maxDumpIntervalMs_) {
                DumpData();
            } else {
                usleep(kMaxWaitTimeUs);
            }
        }
    }
}

void JsonlDataDumper::Flush()
{
    while (dataBuf_.Size() != 0) {
        DumpData();
    }
}

void JsonlDataDumper::Record(std::unique_ptr<nlohmann::json> data)
{
    if (!start_.load() || data == nullptr) {
        return;
    }
    dataBuf_.Push(std::move(data));
}
} // namespace jsonl
} // namespace ipc_monitor
} // namespace dynolog_npu
