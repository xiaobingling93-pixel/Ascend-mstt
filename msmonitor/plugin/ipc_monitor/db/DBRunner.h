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

#ifndef IPC_MONITOR_DB_RUNNER_H
#define IPC_MONITOR_DB_RUNNER_H
#include "db/Connection.h"
#include "utils.h"

namespace dynolog_npu {
namespace ipc_monitor {
namespace db {
class DBRunner {
public:
    explicit DBRunner(const std::string &dbPath): path_(dbPath) {};
    ~DBRunner() = default;
    bool CheckTableExists(const std::string &tableName) const;
    bool CreateTable(const std::string &tableName, const std::vector<TableColumn> &cols) const;
    bool CreateIndex(const std::string &tableName, const std::string &indexName,
                     const std::vector<std::string> &colNames) const;
    bool DropTable(const std::string &tableName) const;
    template<typename... Args>
    bool InsertData(const std::string &tableName, const std::vector<std::tuple<Args...>> &data) const;
    bool DeleteData(const std::string &sql) const;
    template<typename... Args>
    bool QueryData(const std::string &sql, std::vector<std::tuple<Args...>> &result) const;
    bool UpdateData(const std::string &sql) const;
    std::vector<TableColumn> GetTableColumns(const std::string &tableName) const;
private:
    std::string path_;
};

template<typename... Args>
bool DBRunner::InsertData(const std::string &tableName, const std::vector<std::tuple<Args...>> &data) const
{
    if (tableName.empty()) {
        LOG(ERROR) << "Table name is empty";
        return false;
    }
    LOG(INFO) << "Start insert data to " << tableName;
    std::shared_ptr<Connection> conn{nullptr};
    MakeSharedPtr(conn, path_);
    if (conn == nullptr || !conn->IsDBOpened()) {
        LOG(ERROR) << "Create connection for " << tableName << " failed";
        return false;
    }
    if (!conn->ExecuteInsert(tableName, data)) {
        LOG(ERROR) << "Insert data to " << tableName << " failed";
        return false;
    }
    LOG(INFO) << "Insert data to " << tableName << " success";
    return true;
}

template<typename... Args>
bool DBRunner::QueryData(const std::string &sql, std::vector<std::tuple<Args...>> &result) const
{
    LOG(INFO) << "Start query data";
    std::shared_ptr<Connection> conn{nullptr};
    MakeSharedPtr(conn, path_);
    if (conn == nullptr || !conn->IsDBOpened()) {
        LOG(ERROR) << "Create connection failed: " << sql;
        return false;
    }
    if (!conn->ExecuteQuery(sql, result)) {
        LOG(ERROR) << "Query data failed: " << sql;
        return false;
    }
    LOG(INFO) << "Query data success: " << sql;
    return true;
}
} // namespace db
} // namespace ipc_monitor
} // namespace dynolog_npu

#endif // IPC_MONITOR_DB_RUNNER_H
