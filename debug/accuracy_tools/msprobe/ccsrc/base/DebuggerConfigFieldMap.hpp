/*
 * Copyright (C) 2024-2024. Huawei Technologies Co., Ltd. All rights reserved.
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

#include <string>
#include <map>

#include "DebuggerConfig.hpp"

namespace MindStudioDebugger {

constexpr const char* kFramework = "framework";
constexpr const char* kFrameworkPyTorch = "PyTorch";
constexpr const char* kFrameworkMindSpore = "MindSpore";

constexpr const char* kTaskStatistics = "statistics";
constexpr const char* kTaskDumpTensor = "tensor";
constexpr const char* kTaskOverflowCheck = "overflow_check";

constexpr const char* kLevel0 = "L0";
constexpr const char* kLevel1 = "L1";
constexpr const char* kLevel2 = "L2";
constexpr const char* kLevelMix = "mix";

constexpr const char* kDirectionForward = "forward";
constexpr const char* kDirectionBackward = "backward";
constexpr const char* kDirectionBoth = "both";
constexpr const char* kInOutInput = "input";
constexpr const char* kInOutOutput = "output";
constexpr const char* kInOutBoth = "both";
constexpr const char* kDataModeAll = "all";

constexpr const char* kFreeBenchmarkHandlerCheck = "check";
constexpr const char* kFreeBenchmarkHandlerFix = "fix";

constexpr const char* kDumpFileFormatBin = "bin";
constexpr const char* kDumpFileFormatNpy = "npy";

constexpr const char* kOpCheckLevelAiCore = "aicore";
constexpr const char* kOpCheckLevelAtomic = "atomic";
constexpr const char* kOpCheckLevelAll = "all";

constexpr const char* kTask = "task";
constexpr const char* kTasks = "tasks";
constexpr const char* kOutputPath = "dump_path";
constexpr const char* kRank = "rank";
constexpr const char* kStep = "step";
constexpr const char* kLevel = "level";
constexpr const char* kSeed = "seed";
constexpr const char* kIsDeterministic = "is_deterministic";
constexpr const char* kEnableDataloader = "enable_dataloader";
constexpr const char* kAclConfig = "acl_config";

constexpr const char* kScope = "scope";
constexpr const char* kList = "list";

constexpr const char* kDataMode = "data_mode";
constexpr const char* kSummaryMode = "summary_mode";
constexpr const char* kFileFormat = "file_format";
constexpr const char* kOverflowNums = "overflow_nums";
constexpr const char* kCheckMode = "check_mode";
constexpr const char* kBackwardInput = "backward_input";

constexpr const char* kStatistics = "statistics";
constexpr const char* kMd5 = "md5";
constexpr const char* kMax = "max";
constexpr const char* kMin = "min";
constexpr const char* kMean = "mean";
constexpr const char* kL2Norm = "l2norm";
constexpr const char* kNanCount = "nan count";
constexpr const char* kNegativeInfCount = "negative inf count";
constexpr const char* kPositiveInfCount = "positive inf count";

const std::map<int32_t, std::string> FrameworkEnum2Name = {
    {static_cast<int32_t>(DebuggerFramework::FRAMEWORK_PYTORCH), kFrameworkPyTorch},
    {static_cast<int32_t>(DebuggerFramework::FRAMEWORK_MINDSPORE), kFrameworkMindSpore},
};

const std::map<int32_t, std::string> TaskTypeEnum2Name = {
    {static_cast<int32_t>(DebuggerTaskType::TASK_DUMP_TENSOR), kTaskDumpTensor},
    {static_cast<int32_t>(DebuggerTaskType::TASK_DUMP_STATISTICS), kTaskStatistics},
    {static_cast<int32_t>(DebuggerTaskType::TASK_OVERFLOW_CHECK), kTaskOverflowCheck},
};

const std::map<int32_t, std::string> DebuggerLevelEnum2Name = {
    {static_cast<int32_t>(DebuggerLevel::L0), kLevel0},
    {static_cast<int32_t>(DebuggerLevel::L1), kLevel1},
    {static_cast<int32_t>(DebuggerLevel::L2), kLevel2},
    {static_cast<int32_t>(DebuggerLevel::MIX), kLevelMix},
};

const std::map<int32_t, std::string> DataDirectionEnum2Name = {
    {static_cast<int32_t>(DebuggerDataDirection::DIRECTION_FORWARD), kDirectionForward},
    {static_cast<int32_t>(DebuggerDataDirection::DIRECTION_BACKWARD), kDirectionBackward},
    {static_cast<int32_t>(DebuggerDataDirection::DIRECTION_BOTH), kDirectionBoth},
};

const std::map<int32_t, std::string> DataInOutEnum2Name = {
    {static_cast<int32_t>(DebuggerDataInOut::INOUT_INPUT), kInOutInput},
    {static_cast<int32_t>(DebuggerDataInOut::INOUT_OUTPUT), kInOutOutput},
    {static_cast<int32_t>(DebuggerDataInOut::INOUT_BOTH), kInOutBoth},
};

const std::map<int32_t, std::string> DumpFileFormatEnum2Name = {
    {static_cast<int32_t>(DebuggerDumpFileFormat::FILE_FORMAT_BIN), kDumpFileFormatBin},
    {static_cast<int32_t>(DebuggerDumpFileFormat::FILE_FORMAT_NPY), kDumpFileFormatNpy},
};

const std::map<int32_t, std::string> OpCheckLevelEnum2Name = {
    {static_cast<int32_t>(DebuggerOpCheckLevel::CHECK_LEVEL_AICORE), kOpCheckLevelAiCore},
    {static_cast<int32_t>(DebuggerOpCheckLevel::CHECK_LEVEL_ATOMIC), kOpCheckLevelAtomic},
    {static_cast<int32_t>(DebuggerOpCheckLevel::CHECK_LEVEL_ALL), kOpCheckLevelAll},
};

const std::map<int32_t, std::string> SummaryOptionEnum2Name = {
    {static_cast<int32_t>(DebuggerSummaryOption::MAX), kMax},
    {static_cast<int32_t>(DebuggerSummaryOption::MIN), kMin},
    {static_cast<int32_t>(DebuggerSummaryOption::MEAN), kMean},
    {static_cast<int32_t>(DebuggerSummaryOption::NAN_CNT), kNanCount},
    {static_cast<int32_t>(DebuggerSummaryOption::NEG_INF_CNT), kNegativeInfCount},
    {static_cast<int32_t>(DebuggerSummaryOption::POS_INF_CNT), kPositiveInfCount},
    {static_cast<int32_t>(DebuggerSummaryOption::L2NORM), kL2Norm},

    {static_cast<int32_t>(DebuggerSummaryOption::MD5), kMd5},
};

inline int32_t GetEnumIdFromName(const std::map<int32_t, std::string>& enum2name, const std::string& name)
{
    for (auto iter = enum2name.begin(); iter != enum2name.end(); iter++) {
        if (iter->second == name) {
            return iter->first;
        }
    }
    return debuggerInvalidEnum;
}

inline std::string GetNameFromEnumId(const std::map<int32_t, std::string>& enum2name, int32_t id)
{
    auto iter = enum2name.find(id);
    if (iter == enum2name.end()) {
        return "UNKNOWN";
    }
    return iter->second;
}

}