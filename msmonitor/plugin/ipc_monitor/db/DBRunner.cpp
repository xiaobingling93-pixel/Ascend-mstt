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

#include "db/DBRunner.h"
#include <algorithm>

namespace dynolog_npu {
namespace ipc_monitor {
namespace db {
namespace {
std::string GetColumnsString(const std::vector<TableColumn> &columns)
{
    std::vector<std::string> columnStrings(columns.size());
    std::transform(columns.begin(), columns.end(), columnStrings.begin(), [](const TableColumn &column) {
        return column.ToString();
    });
    return join(columnStrings, ",");
}
}

bool DBRunner::CheckTableExists(const std::string &tableName) const
{
    std::shared_ptr<Connection> conn{nullptr};
    MakeSharedPtr(conn, path_);
    if (conn == nullptr) {
        return false;
    }
    return conn->CheckTableExists(tableName);
}

bool DBRunner::CreateTable(const std::string &tableName, const std::vector<TableColumn> &columns) const
{
    if (tableName.empty()) {
        LOG(ERROR) << "Create table failed, table name is empty";
        return false;
    }
    std::shared_ptr<Connection> conn{nullptr};
    MakeSharedPtr(conn, path_);
    if (conn == nullptr) {
        return false;
    }
    LOG(INFO) << "Create table " << tableName;
    std::string columnsString = GetColumnsString(columns);
    std::string sql = "CREATE TABLE IF NOT EXISTS " + tableName + " (" + columnsString + ")";
    if (!conn->ExecuteCreateTable(sql)) {
        LOG(ERROR) << "Create table " << tableName << " failed";
        return false;
    }
    LOG(INFO) << "Create table " << tableName << " success";
    return true;
}

bool DBRunner::CreateIndex(const std::string &tableName, const std::string &indexName,
                           const std::vector<std::string> &colNames) const
{
    if (tableName.empty() || indexName.empty() || colNames.empty()) {
        LOG(ERROR) << "Create index failed, table name or index name or column name is empty";
        return false;
    }
    std::shared_ptr<Connection> conn{nullptr};
    MakeSharedPtr(conn, path_);
    if (conn == nullptr) {
        return false;
    }
    LOG(INFO) << "Create index " << indexName << " on table " << tableName;
    std::string valueStr = join(colNames, ",");
    std::string sql = "CREATE INDEX IF NOT EXISTS " + indexName + " ON " + tableName + " (" + valueStr + ")";
    if (!conn->ExecuteCreateIndex(sql)) {
        LOG(ERROR) << "Create index " << indexName << " on table " << tableName << " failed, sql: " << sql;
        return false;
    }
    LOG(INFO) << "Create index " << indexName << " on table " << tableName << " success";
    return true;
}

bool DBRunner::DropTable(const std::string &tableName) const
{
    if (tableName.empty()) {
        LOG(ERROR) << "Drop table failed, table name is empty";
        return false;
    }
    std::shared_ptr<Connection> conn{nullptr};
    MakeSharedPtr(conn, path_);
    if (conn == nullptr) {
        return false;
    }
    LOG(INFO) << "Drop table " << tableName;
    std::string sql = "DROP TABLE " + tableName;
    if (!conn->ExecuteDropTable(sql)) {
        LOG(ERROR) << "Drop table " << tableName << " failed";
        return false;
    }
    LOG(INFO) << "Drop table " << tableName << " success";
    return true;
}

bool DBRunner::DeleteData(const std::string &sql) const
{
    std::shared_ptr<Connection> conn{nullptr};
    MakeSharedPtr(conn, path_);
    if (conn == nullptr) {
        return false;
    }
    LOG(INFO) << "Delete data, sql: " << sql;
    if (!conn->ExecuteDelete(sql)) {
        LOG(ERROR) << "Delete data failed, sql: " << sql;
        return false;
    }
    LOG(INFO) << "Delete data success, sql: " << sql;
    return true;
}

bool DBRunner::UpdateData(const std::string &sql) const
{
    std::shared_ptr<Connection> conn{nullptr};
    MakeSharedPtr(conn, path_);
    if (conn == nullptr) {
        return false;
    }
    LOG(INFO) << "Update data, sql: " << sql;
    if (!conn->ExecuteUpdate(sql)) {
        LOG(ERROR) << "Update data failed, sql: " << sql;
        return false;
    }
    LOG(INFO) << "Update data success, sql: " << sql;
    return true;
}

std::vector<TableColumn> DBRunner::GetTableColumns(const std::string &tableName) const
{
    std::shared_ptr<Connection> conn{nullptr};
    MakeSharedPtr(conn, path_);
    if (conn == nullptr) {
        return {};
    }
    LOG(INFO) << "Get table columns, table name: " << tableName;
    auto cols = conn->ExecuteGetTableColumns(tableName);
    if (cols.empty()) {
        LOG(ERROR) << "Get table columns failed, table name: " << tableName;
        return cols;
    }
    LOG(INFO) << "Get table columns success, table name: " << tableName;
    return cols;
}
} // namespace db
} // namespace ipc_monitor
} // namespace dynolog_npu
