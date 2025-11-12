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
#include "InputParser.h"
#include <unordered_set>
#include <unordered_map>
#include <functional>
#include <limits>
#include <climits>
#include <algorithm>
#include <glog/logging.h>
#include "utils.h"

namespace dynolog_npu {
namespace ipc_monitor {
namespace {

const std::string MSPTI_ACTIVITY_KIND_KEY = "MSPTI_ACTIVITY_KIND";
const std::string REPORT_INTERVAL_S_KEY = "REPORT_INTERVAL_S";
const std::string NPU_MONITOR_START_KEY = "NPU_MONITOR_START";
const std::string NPU_MONITOR_STOP_KEY = "NPU_MONITOR_STOP";
const std::string NPU_MONITOR_SAVE_PATH = "NPU_MONITOR_LOG_FILE";

const std::unordered_map<std::string, msptiActivityKind> kindStrMap = {
    {"Marker", MSPTI_ACTIVITY_KIND_MARKER},
    {"Kernel", MSPTI_ACTIVITY_KIND_KERNEL},
    {"API", MSPTI_ACTIVITY_KIND_API},
    {"Hccl", MSPTI_ACTIVITY_KIND_HCCL},
    {"Memory", MSPTI_ACTIVITY_KIND_MEMORY},
    {"MemSet", MSPTI_ACTIVITY_KIND_MEMSET},
    {"MemCpy", MSPTI_ACTIVITY_KIND_MEMCPY},
    {"Communication", MSPTI_ACTIVITY_KIND_COMMUNICATION}
};

bool isValidKind(const std::string& kindStrs)
{
    auto kindStrList = split(kindStrs, ',');
    return std::all_of(kindStrList.begin(), kindStrList.end(), [&kindStrMap](const std::string& kindStr) {
        return kindStrMap.find(kindStr) != kindStrMap.end();
    });
}

bool isUint32(const std::string& s)
{
    if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos) {
        return false;
    }
    try {
        unsigned long long value = std::stoull(s);
        return value <= std::numeric_limits<uint32_t>::max();
    } catch (...) {
        return false;
    }
}

bool isBool(const std::string& s)
{
    std::string lowerS = s;
    std::transform(lowerS.begin(), lowerS.end(), lowerS.begin(), ::tolower);
    return lowerS == "true" || lowerS == "false";
}

bool isValidPath(const std::string& s)
{
    return s.length() <= PATH_MAX;
}

struct Rule {
    bool required = true;
    std::function<bool(const std::string&)> validate;
    std::string description;
};

std::unordered_map<std::string, Rule> rules = {
    {MSPTI_ACTIVITY_KIND_KEY, {true, isValidKind,
        "valid values: Marker, Kernel, API, Hccl, Memory, MemSet, MemCpy, Communication"}},
    {REPORT_INTERVAL_S_KEY, {true, isUint32, "valid values: uint32"}},
    {NPU_MONITOR_START_KEY, {true, isBool, "valid values: true/True, false/False"}},
    {NPU_MONITOR_STOP_KEY, {true, isBool, "valid values: true/True, false/False"}},
    {NPU_MONITOR_SAVE_PATH, {true, isValidPath, "valid path (max length 4096 characters)"}}
};

bool validateArgs(const std::unordered_map<std::string, std::string>& args,
                  const std::unordered_map<std::string, Rule>& rules)
{
    for (const auto& rulePair : rules) {
        const std::string& key = rulePair.first;
        const Rule& rule = rulePair.second;
        auto it = args.find(key);
        if (it == args.end()) {
            if (rule.required) {
                LOG(ERROR) << "Missing required key: " << key << "\n";
                continue;
            }
        }
        if (!rule.validate(it->second)) {
            LOG(ERROR) << "Invalid value for " << key
                      << " = '" << it->second
                      << "' (" << rule.description << ")\n";
            return false;
        }
    }
    return true;
}

std::set<msptiActivityKind> str2Kinds(const std::string& kindStrs)
{
    std::set<msptiActivityKind> res;
    auto kindStrList = split(kindStrs, ',');
    for (auto& kindStr : kindStrList) {
        auto kind = kindStrMap.find(kindStr);
        if (kind == kindStrMap.end()) {
            return {MSPTI_ACTIVITY_KIND_INVALID};
        }
        res.insert(kind->second);
    }
    return res;
}
}

MsptiMonitorCfg InputParser::DynoLogGetOpts(std::unordered_map<std::string, std::string>& cmd)
{
    if (!validateArgs(cmd, rules)) {
        return {{MSPTI_ACTIVITY_KIND_INVALID}, 0, false, false, false, ""};
    }
    auto activityKinds = str2Kinds(cmd[MSPTI_ACTIVITY_KIND_KEY]);
    uint32_t reportTimes = 0;
    Str2Uint32(reportTimes, cmd[REPORT_INTERVAL_S_KEY]);
    bool startSwitch = false; 
    Str2Bool(startSwitch, cmd[NPU_MONITOR_START_KEY]);
    bool endSwitch = false;
    Str2Bool(endSwitch, cmd[NPU_MONITOR_STOP_KEY]);
    return {activityKinds, reportTimes, startSwitch, endSwitch, true, cmd[NPU_MONITOR_SAVE_PATH]};
}
} // namespace ipc_monitor
} // namespace dynolog_npu
