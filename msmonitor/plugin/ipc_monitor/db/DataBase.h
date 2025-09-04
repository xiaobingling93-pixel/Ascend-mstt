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

#ifndef IPC_MONITOR_DB_BASE_H
#define IPC_MONITOR_DB_BASE_H

#include <unordered_map>
#include "db/Connection.h"

namespace dynolog_npu {
namespace ipc_monitor {
namespace db {
using TableColumns = std::vector<TableColumn>;

class Database {
public:
    Database() = default;
    virtual ~Database() = default;
    void SetDBName(std::string dbName) { dbName_ = std::move(dbName); }
    std::string GetDBName() const { return dbName_; }
    const TableColumns& GetTableCols(const std::string &tableName);
protected:
    std::string dbName_;
    std::unordered_map<std::string, TableColumns> tableColumns_;
};

class MsMonitorDB : public Database {
public:
    MsMonitorDB();
};
} // namespace db
} // namespace ipc_monitor
} // namespace dynolog_npu

#endif // IPC_MONITOR_DB_BASE_H
