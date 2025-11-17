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

#ifndef MSMONITOR_JSONL_DATA_DUMPER_H_
#define MSMONITOR_JSONL_DATA_DUMPER_H_

#include <atomic>
#include <memory>
#include <chrono>
#include <nlohmann/json.hpp>
#include "thread.h"
#include "RingBuffer.h"
#include "jsonl/RotateLogger.h"

namespace dynolog_npu {
namespace ipc_monitor {
namespace jsonl {
constexpr uint32_t kMaxWaitTimeUs = 1024;
constexpr uint32_t kNotifyInterval = 256;

class JsonlDataDumper : public Thread {
public:
    explicit JsonlDataDumper() : dumpDir_(""), start_(false), init_(false) {}
    virtual ~JsonlDataDumper() { UnInit(); }
    void Init(const std::string &dirPath, size_t capacity, uint32_t maxDumpIntervalMs);
    void UnInit();
    void Record(std::unique_ptr<nlohmann::json> data);
    void Start();
    void Stop();

private:
    void Flush();
    void Run();
    void DumpData();

private:
    std::string dumpDir_;
    std::atomic<bool> start_;
    std::atomic<bool> init_;
    std::chrono::milliseconds maxDumpIntervalMs_;
    std::chrono::time_point<std::chrono::steady_clock> lastDumpTime_;
    RingBuffer<std::unique_ptr<nlohmann::json>> dataBuf_;
    std::unique_ptr<RotateLogger> rotateLogger_{nullptr};
};
} // namespace jsonl
} // namespace ipc_monitor
} // namespace dynolog_npu

#endif // MSMONITOR_JSONL_DATA_DUMPER_H_
