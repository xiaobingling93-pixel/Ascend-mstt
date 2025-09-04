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

#include "db/DataBase.h"
#include "db/DBConstant.h"

namespace dynolog_npu {
namespace ipc_monitor {
namespace db {
namespace {
const TableColumns STRING_IDS = {
    {"id", SQL_INT_TYPE},
    {"value", SQL_TEXT_TYPE}
};

const TableColumns SESSION_TIME_INFO = {
    {"startTimeNs", SQL_INT_TYPE},
    {"endTimeNs", SQL_INT_TYPE}
};

const TableColumns ENUM_TABLE = {
    {"id", SQL_INT_TYPE, true},
    {"name", SQL_TEXT_TYPE}
};

const TableColumns META_DATA = {
    {"name", SQL_TEXT_TYPE},
    {"value", SQL_TEXT_TYPE}
};

const TableColumns HOST_INFO = {
    {"hostUid", SQL_TEXT_TYPE},
    {"hostName", SQL_TEXT_TYPE}
};

const TableColumns RANK_DEVICE_MAP = {
    {"rankId", SQL_INT_TYPE},
    {"deviceId", SQL_INT_TYPE}
};

const TableColumns CANN_API = {
    {"startNs", SQL_INT_TYPE},
    {"endNs", SQL_INT_TYPE},
    {"type", SQL_INT_TYPE},
    {"globalTid", SQL_INT_TYPE},
    {"connectionId", SQL_INT_TYPE},
    {"name", SQL_INT_TYPE}
};

const TableColumns TASK = {
    {"startNs", SQL_INT_TYPE},
    {"endNs", SQL_INT_TYPE},
    {"deviceId", SQL_INT_TYPE},
    {"connectionId", SQL_INT_TYPE},
    {"globalTaskId", SQL_INT_TYPE},
    {"globalPid", SQL_INT_TYPE},
    {"taskType", SQL_INT_TYPE},
    {"contextId", SQL_INT_TYPE},
    {"streamId", SQL_INT_TYPE},
    {"taskId", SQL_INT_TYPE},
    {"modelId", SQL_INT_TYPE}
};

const TableColumns COMPUTE_TASK_INFO = {
    {"name", SQL_INT_TYPE},
    {"globalTaskId", SQL_INT_TYPE},
    {"blockDim", SQL_INT_TYPE},
    {"mixBlockDim", SQL_INT_TYPE},
    {"taskType", SQL_INT_TYPE},
    {"opType", SQL_INT_TYPE},
    {"inputFormats", SQL_INT_TYPE},
    {"inputDataTypes", SQL_INT_TYPE},
    {"inputShapes", SQL_INT_TYPE},
    {"outputFormats", SQL_INT_TYPE},
    {"outputDataTypes", SQL_INT_TYPE},
    {"outputShapes", SQL_INT_TYPE},
    {"attrInfo", SQL_INT_TYPE},
    {"opState", SQL_INT_TYPE},
    {"hf32Eligible", SQL_INT_TYPE}
};

const TableColumns COMMUNICATION_OP = {
    {"opName", SQL_INT_TYPE},
    {"startNs", SQL_INT_TYPE},
    {"endNs", SQL_INT_TYPE},
    {"connectionId", SQL_INT_TYPE},
    {"groupName", SQL_INT_TYPE},
    {"opId", SQL_INT_TYPE},
    {"relay", SQL_INT_TYPE},
    {"retry", SQL_INT_TYPE},
    {"dataType", SQL_INT_TYPE},
    {"algType", SQL_INT_TYPE},
    {"count", SQL_NUMERIC_TYPE},
    {"opType", SQL_INT_TYPE}
};

const TableColumns MSTX = {
    {"startNs", SQL_INT_TYPE},
    {"endNs", SQL_INT_TYPE},
    {"eventType", SQL_INT_TYPE},
    {"rangeId", SQL_INT_TYPE},
    {"category", SQL_INT_TYPE},
    {"message", SQL_INT_TYPE},
    {"globalTid", SQL_INT_TYPE},
    {"endGlobalTid", SQL_INT_TYPE},
    {"domainId", SQL_INT_TYPE},
    {"connectionId", SQL_INT_TYPE}
};
} // namespace

const TableColumns& Database::GetTableCols(const std::string &tableName)
{
    auto iter = tableColumns_.find(tableName);
    if (iter == tableColumns_.end()) {
        LOG(ERROR) << "Table " << tableName << " is not found";
        return {};
    }
    return iter->second;
}

MsMonitorDB::MsMonitorDB()
{
    dbName_ = "msmonitor.db";
    tableColumns_ = {
        {TABLE_STRING_IDS, STRING_IDS},
        {TABLE_SESSION_TIME_INFO, SESSION_TIME_INFO},
        {TABLE_COMMUNICATION_OP, COMMUNICATION_OP},
        {TABLE_HCCL_DATA_TYPE, ENUM_TABLE},
        {TABLE_MSTX, MSTX},
        {TABLE_MSTX_EVENT_TYPE, ENUM_TABLE},
        {TABLE_API_TYPE, ENUM_TABLE},
        {TABLE_CANN_API, CANN_API},
        {TABLE_TASK, TASK},
        {TABLE_COMPUTE_TASK_INFO, COMPUTE_TASK_INFO},
        {TABLE_META_DATA, META_DATA},
        {TABLE_HOST_INFO, HOST_INFO},
        {TABLE_RANK_DEVICE_MAP, RANK_DEVICE_MAP}
    };
}
} // namespace db
} // namespace ipc_monitor
} // namespace dynolog_npu
