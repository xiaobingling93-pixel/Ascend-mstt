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
#pragma once

#include <unordered_set>
#include <atomic>
#include <mutex>
#include <filesystem>

#include "dynolog/src/Logger.h"

#include "MsMonitorMetrics.h"

#ifdef USE_TENSORBOARD

#include "tensorboard_logger.h"

DECLARE_string(metric_log_dir);

namespace dynolog {

class TensorBoardLoggerImpl {
public:
    explicit TensorBoardLoggerImpl(std::string log_path, std::string tag = "")
        : log_path_(log_path), tag_(tag) {};
    void log(const std::string& key, double val, uint64_t step);
private:
    std::filesystem::path log_path_;
    std::string tag_;
    static const std::string log_file_name_;
    std::shared_ptr<TensorBoardLogger> logger_;
};

class TensorBoardLoggerManager {
public:
    struct LoggingGuard {
        std::shared_ptr<TensorBoardLoggerManager> manager;
        std::lock_guard<std::mutex> lock_guard;
    };

    void log(const MsptiMetricDesc& desc);

    static void logPath(const std::string& cfg_log_path)
    {
        log_path_ = cfg_log_path;
    }

    static LoggingGuard singleton();

    bool isValidMetric(const MsptiMetricDesc& desc);

    uint64_t getCurStep(const std::string& device, const std::string& kind);

private:
    std::lock_guard<std::mutex> lock()
    {
        return std::lock_guard{mutex_};
    }
    static std::shared_ptr<TensorBoardLoggerManager> singleton_();

    std::mutex mutex_;
    static std::filesystem::path log_path_;

    std::unordered_map<std::string, std::shared_ptr<TensorBoardLoggerImpl>> device_loggers_;
    std::map<std::pair<std::string, std::string>, std::uint64_t> device_kind2_step_;
};

class DynologTensorBoardLogger final : public Logger {
public:
    explicit DynologTensorBoardLogger(const std::string& metric_log_dir);
    void setTimestamp(Timestamp ts) override {}

    void logInt(const std::string& key, int64_t val) override
    {
        kvs_[key] = static_cast<double>(val);
    }

    void logFloat(const std::string& key, float val) override
    {
        kvs_[key] = static_cast<double>(val);
    }

    void logUint(const std::string& key, uint64_t val) override
    {
        kvs_[key] = static_cast<double>(val);
    }

  // logStr for dynamic metris
    void logStr(const std::string& key, const std::string& val) override
    {
        if (validDynamicMetrics_.count(key)) {
            dynamic_metrics_[key] = val;
        }
    }

    void finalize() override;

private:
    bool validateLogDir(const std::string& path);

private:
    std::unordered_map<std::string, double> kvs_;
    std::unordered_map<std::string, std::string> dynamic_metrics_;
    std::string logPath_;
    std::string hostName_;
};

} // namespace dynolog
#endif