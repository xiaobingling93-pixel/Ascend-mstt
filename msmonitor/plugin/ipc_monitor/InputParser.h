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
#ifndef INPUT_PARSER_H
#define INPUT_PARSER_H

#include <unordered_map>
#include <singleton.h>
#include <set>
#include <mspti.h>

namespace dynolog_npu {
namespace ipc_monitor {

struct MsptiMonitorCfg {
    std::set<msptiActivityKind> enableActivities;
    uint32_t reportIntervals;
    bool monitorStart;
    bool monitorStop;
    bool isMonitor;
};


class InputParser : public dynolog_npu::ipc_monitor::Singleton<InputParser> {
public:
    MsptiMonitorCfg DynoLogGetOpts(std::unordered_map<std::string, std::string>& cmd);
};

} // namespace ipc_monitor
} // namespace dynolog_npu

#endif