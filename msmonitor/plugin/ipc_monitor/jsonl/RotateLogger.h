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

#ifndef IPC_MONITOR_ROTATE_LOGGER_H
#define IPC_MONITOR_ROTATE_LOGGER_H

#include <cstdio>
#include <string>
#include <vector>

namespace dynolog_npu {
namespace ipc_monitor {
namespace jsonl {

class RotateLogger {
public:
    RotateLogger(const std::string& logDir, const uint32_t maxLines, const int32_t maxFiles)
        : logDir_(logDir), maxLines_(maxLines), maxFiles_(maxFiles) {}
    ~RotateLogger();
    void UnInit();
    void Log(std::string message);

private:
    bool OpenNewFile();
    void Rotate();
    void ManageFiles();

private:
    std::string logDir_;
    uint32_t maxLines_{10000};
    int32_t maxFiles_{-1};
    uint32_t curLines_{0};
    std::FILE* logFile_{nullptr};
    std::vector<std::string> logFiles_;
};
} // namespace jsonl
} // namespace ipc_monitor
} // namespace dynolog_npu
#endif
