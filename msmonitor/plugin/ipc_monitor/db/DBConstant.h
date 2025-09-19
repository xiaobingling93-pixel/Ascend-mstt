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

#ifndef IPC_MONITOR_DB_CONSTANT_H
#define IPC_MONITOR_DB_CONSTANT_H

#include <string>

namespace dynolog_npu {
namespace ipc_monitor {
namespace db {
const std::string SQL_TEXT_TYPE = "TEXT";
const std::string SQL_INT_TYPE = "INTEGER";
const std::string SQL_REAL_TYPE = "REAL";
const std::string SQL_NUMERIC_TYPE = "NUMERIC";

const std::string TABLE_STRING_IDS = "STRING_IDS";
const std::string TABLE_SESSION_TIME_INFO = "SESSION_TIME_INFO";
const std::string TABLE_CANN_API = "CANN_API";
const std::string TABLE_TASK = "TASK";
const std::string TABLE_COMPUTE_TASK_INFO = "COMPUTE_TASK_INFO";
const std::string TABLE_COMMUNICATION_OP = "COMMUNICATION_OP";
const std::string TABLE_MSTX = "MSTX_EVENTS";
const std::string TABLE_MSTX_EVENT_TYPE = "ENUM_MSTX_EVENT_TYPE";
const std::string TABLE_HCCL_DATA_TYPE = "ENUM_HCCL_DATA_TYPE";
const std::string TABLE_API_TYPE = "ENUM_API_TYPE";
const std::string TABLE_HOST_INFO = "HOST_INFO";
const std::string TABLE_NPU_INFO = "NPU_INFO";
const std::string TABLE_RANK_DEVICE_MAP = "RANK_DEVICE_MAP";
const std::string TABLE_META_DATA = "META_DATA";

} // namespace db
} // namespace ipc_monitor
} // namespace dynolog_npu

#endif // IPC_MONITOR_DB_CONSTANT_H
