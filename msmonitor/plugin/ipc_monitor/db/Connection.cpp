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

#include "db/Connection.h"
#include "utils.h"

namespace {
constexpr int32_t TIMEOUT = INT32_MAX;
const std::string CREATE_TABLE = "CREATE TABLE";
const std::string CREATE_INDEX = "CREATE INDEX";
const std::string DROP_TABLE = "DROP TABLE";
const std::string UPDATE = "UPDATE";
const std::string DELETE = "DELETE";
const std::string CHECK = "CHECK";
}

namespace dynolog_npu {
namespace ipc_monitor {
namespace db {
Connection::Connection(const std::string &path)
{
    auto rc = sqlite3_open(path.c_str(), &db_);
    if (rc != SQLITE_OK) {
        LOG(ERROR) << "Open database failed: " << rc << ", msg: " << sqlite3_errmsg(db_);
        sqlite3_close_v2(db_);
        db_ = nullptr;
    } else {
        sqlite3_exec(db_, "PRAGMA synchronous=OFF;", nullptr, nullptr, nullptr);
    }
}

Connection::~Connection()
{
    if (stmt_) {
        sqlite3_finalize(stmt_);
    }
    if (db_) {
        auto rc = sqlite3_close(db_);
        if (rc != SQLITE_OK) {
            LOG(ERROR) << "Close database failed: " << rc << ", msg: " << sqlite3_errmsg(db_);
            sqlite3_close_v2(db_);
        }
        db_ = nullptr;
    }
}

bool Connection::ExecuteSql(const std::string &sql, const std::string &sqlType)
{
    CHAR_PTR errMsg{nullptr};
    sqlite3_busy_timeout(db_, TIMEOUT);
    auto rc = sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &errMsg);
    if (rc != SQLITE_OK) {
        if (sqlType == CHECK) {
            LOG(WARNING) << "Execute sql failed: " << rc << ", type: " << sqlType << ", msg: " << errMsg;
        } else {
            LOG(ERROR) << "Execute sql failed: " << rc << ", type: " << sqlType << ", msg: " << errMsg;
        }
        sqlite3_free(errMsg);
        return false;
    }
    return true;
}

bool Connection::CheckTableExists(const std::string &tableName)
{
    std::string sql{"SELECT 1 FROM sqlite_master WHERE type='table' AND name='" + tableName + "' LIMIT 1"};
    std::vector<std::tuple<int32_t>> result;
    if (ExecuteQuery(sql, result)) {
        return !result.empty();
    }
    return false;
}

bool Connection::ExecuteCreateTable(const std::string &sql)
{
    return ExecuteSql(sql, CREATE_TABLE);
}

bool Connection::ExecuteCreateIndex(const std::string &sql)
{
    return ExecuteSql(sql, CREATE_INDEX);
}

bool Connection::ExecuteDropTable(const std::string &sql)
{
    return ExecuteSql(sql, DROP_TABLE);
}

bool Connection::ExecuteUpdate(const std::string &sql)
{
    return ExecuteSql(sql, UPDATE);
}

bool Connection::ExecuteDelete(const std::string &sql)
{
    return ExecuteSql(sql, DELETE);
}

std::vector<TableColumn> Connection::ExecuteGetTableColumns(const std::string &tableName)
{
    std::vector<TableColumn> columns;
    std::string sql = "PRAGMA table_info(" + tableName + ")";
    sqlite3_busy_timeout(db_, TIMEOUT);
    auto rc = sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt_, nullptr);
    if (rc != SQLITE_OK) {
        LOG(ERROR) << "Execute sql failed: " << rc << ", msg: " << sqlite3_errmsg(db_);
        return columns;
    }
    while (sqlite3_step(stmt_) == SQLITE_ROW) {
        std::string name;
        std::string type;
        GetColumn(name);
        GetColumn(type);
        columns.emplace_back(name, type);
        index_ = 0;
    }
    return columns;
}

bool Connection::InsertCmd(const std::string &tableName, uint32_t colNum)
{
    std::string sql = "INSERT INTO " + tableName + " VALUES (";
    for (uint32_t i = 0; i < colNum; ++i) {
        sql += "?";
        if (i < colNum - 1) {
            sql += ", ";
        }
    }
    sql += ")";
    sqlite3_busy_timeout(db_, TIMEOUT);
    auto rc = sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt_, nullptr);
    if (rc != SQLITE_OK) {
        LOG(ERROR) << "Execute sql failed: " << rc << ", msg: " << sqlite3_errmsg(db_);
        return false;
    }
    return true;
}

bool Connection::QueryCmd(const std::string &sql)
{
    auto rc = sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt_, nullptr);
    if (rc != SQLITE_OK) {
        LOG(ERROR) << "Execute sql failed: " << rc << ", msg: " << sqlite3_errmsg(db_);
        return false;
    }
    return true;
}

bool Connection::BindParameters(int32_t value)
{
    return sqlite3_bind_int(stmt_, ++index_, value) == SQLITE_OK;
}

bool Connection::BindParameters(uint32_t value)
{
    return sqlite3_bind_int64(stmt_, ++index_, value) == SQLITE_OK;
}

bool Connection::BindParameters(int64_t value)
{
    return sqlite3_bind_int64(stmt_, ++index_, value) == SQLITE_OK;
}

bool Connection::BindParameters(uint64_t value)
{
    return sqlite3_bind_int64(stmt_, ++index_, value) == SQLITE_OK;
}

bool Connection::BindParameters(double value)
{
    return sqlite3_bind_double(stmt_, ++index_, value) == SQLITE_OK;
}

bool Connection::BindParameters(std::string value)
{
    return sqlite3_bind_text(stmt_, ++index_, value.c_str(), -1, SQLITE_TRANSIENT) == SQLITE_OK;
}

void Connection::GetColumn(uint16_t &value)
{
    value = static_cast<uint16_t>(sqlite3_column_int(stmt_, ++index_));
}

void Connection::GetColumn(int32_t &value)
{
    value = sqlite3_column_int(stmt_, ++index_);
}

void Connection::GetColumn(uint32_t &value)
{
    value = static_cast<uint32_t>(sqlite3_column_int64(stmt_, ++index_));
}

void Connection::GetColumn(int64_t &value)
{
    value = sqlite3_column_int64(stmt_, ++index_);
}

void Connection::GetColumn(uint64_t &value)
{
    value = static_cast<uint64_t>(sqlite3_column_int64(stmt_, ++index_));
}

void Connection::GetColumn(double &value)
{
    value = sqlite3_column_double(stmt_, ++index_);
}

void Connection::GetColumn(std::string &value)
{
    const unsigned char *text = sqlite3_column_text(stmt_, ++index_);
    if (text == nullptr) {
        value.clear();
    } else {
        value = std::string(ReinterpretConvert<const char*>(text));
    }
}
} // namespace db
} // namespace ipc_monitor
} // namespace dynolog_npu
