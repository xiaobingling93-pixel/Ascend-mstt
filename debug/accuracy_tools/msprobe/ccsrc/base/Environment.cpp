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

#include "utils/CPythonUtils.hpp"
#include "DebuggerConfig.hpp"
#include "Environment.hpp"

namespace MindStudioDebugger {
namespace Environment {

static int32_t GetRankID_PT()
{
    /*  if torch.distributed.is_initialized():
     *     return torch.distributed.get_rank()
     */
    CPythonUtils::PythonObject torch = CPythonUtils::PythonObject::Import("torch");
    if (!torch.IsModule()) {
        return -1;
    }

    CPythonUtils::PythonObject distributed = torch.Get("distributed");
    if (distributed.IsNone()) {
        return -1;
    }

    if (!distributed.Get("is_initialized").Call()) {
        return -1;
    }

    CPythonUtils::PythonObject rank = distributed.Get("get_rank").Call();
    int32_t id;
    if (rank.To(id) != 0) {
        return -1;
    }
    return id;
}

static int32_t GetRankID_MS()
{
    constexpr const char* kRankId = "RANK_ID";
    const char* rankIdEnv =  getenv(kRankId);
    if (rankIdEnv == nullptr) {
        return -1;
    }

    std::string rankId(rankIdEnv);
    std::istringstream iss(rankId);
    int32_t id = -1;
    if (!(iss >> id) || id < 0) {
        return -1;
    }

    return id;
}

int32_t GetRankID()
{
    if (!DebuggerConfig::GetInstance().IsCfgLoaded()) {
        return -1;
    }

    static int32_t id = -1;
    if (id >= 0) {
        return id;
    }

    if (DebuggerConfig::GetInstance().GetFramework() == DebuggerFramework::FRAMEWORK_PYTORCH) {
        id = GetRankID_PT();
    } else if (DebuggerConfig::GetInstance().GetFramework() == DebuggerFramework::FRAMEWORK_MINDSPORE) {
        id = GetRankID_MS();
    }

    return id;
}

}
}