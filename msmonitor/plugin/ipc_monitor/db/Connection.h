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

#ifndef IPC_MONITOR_DB_CONNECTION_H
#define IPC_MONITOR_DB_CONNECTION_H

#include <cstdint>
#include <string>
#include <vector>
#include <tuple>

#include <sqlite3.h>
#include <glog/logging.h>

namespace dynolog_npu {
namespace ipc_monitor {
namespace db {
using CHAR_PTR = char*;
struct TableColumn {
    std::string name;
    std::string type;
    bool isPrimaryKey = false;

    TableColumn(const std::string &name, const std::string &type, bool isPrimaryKey = false)
        : name(name), type(type), isPrimaryKey(isPrimaryKey) {}

    std::string ToString() const
    {
        return name + " " + type + (isPrimaryKey ? " PRIMARY KEY" : "");
    }

    bool operator==(const TableColumn &other) const
    {
        return (name == other.name) && (type == other.type);
    }
};

template<size_t... S>
struct IndexSequence {};

template<size_t N, size_t... S>
struct IndexSequenceMaker : IndexSequenceMaker<N - 1, N - 1, S...> {};

template<size_t... S>
struct IndexSequenceMaker<0, S...> {
    using type = IndexSequence<S...>;
};

template<size_t N>
using MakeIndexSequence = typename IndexSequenceMaker<N>::type;

class Connection {
public:
    explicit Connection(const std::string &path);
    ~Connection();
    bool IsDBOpened() const { return db_ != nullptr; }
    bool CheckTableExists(const std::string &tableName);
    bool ExecuteSql(const std::string &sql, const std::string &sqlType);
    bool ExecuteCreateTable(const std::string &sql);
    bool ExecuteCreateIndex(const std::string &sql);
    bool ExecuteDropTable(const std::string &sql);
    template<typename... Args>
    bool ExecuteInsert(const std::string &tableName, const std::vector<std::tuple<Args...>> &data);
    template<typename... Args>
    bool ExecuteQuery(const std::string &sql, std::vector<std::tuple<Args...>> &result);
    bool ExecuteUpdate(const std::string &sql);
    bool ExecuteDelete(const std::string &sql);
    std::vector<TableColumn> ExecuteGetTableColumns(const std::string &tableName);

private:
    bool InsertCmd(const std::string &tableName, uint32_t colNum);
    void FinalizeStmt();
    bool BindParameters(int32_t value);
    bool BindParameters(uint32_t value);
    bool BindParameters(int64_t value);
    bool BindParameters(uint64_t value);
    bool BindParameters(double value);
    bool BindParameters(std::string value);
    template<typename T, size_t... S>
    void ExecuteInsertHelper(T &row, IndexSequence<S...>);
    template<typename T>
    int ExecuteInsertHelperHerlper(T t);
    template<typename T>
    void InsertRow(T &row);

    bool QueryCmd(const std::string &sql);
    void GetColumn(uint16_t &value);
    void GetColumn(int32_t &value);
    void GetColumn(uint32_t &value);
    void GetColumn(int64_t &value);
    void GetColumn(uint64_t &value);
    void GetColumn(double &value);
    void GetColumn(std::string &value);
    template<typename T, size_t... S>
    void ExecuteQueryHelper(T &row, IndexSequence<S...>);
    template<typename T>
    int ExecuteQueryHelperHelper(T &t);
    template<typename T>
    void GetRow(T &row);

private:
    int index_{0};
    sqlite3 *db_{nullptr};
    sqlite3_stmt *stmt_{nullptr};
};

template<typename T>
int Connection::ExecuteInsertHelperHerlper(T t)
{
    return BindParameters(t) ? 0 : -1;
}

template<typename T, size_t... S>
void Connection::ExecuteInsertHelper(T &row, IndexSequence<S...>)
{
    std::initializer_list<int> {(ExecuteInsertHelperHerlper(std::get<S>(row)), 0)...};
}

template<typename T>
void Connection::InsertRow(T &row)
{
    using TupleType = typename std::decay<T>::type;
    ExecuteInsertHelper(row, MakeIndexSequence<std::tuple_size<TupleType>::value>{});
}

template<typename T>
int Connection::ExecuteQueryHelperHelper(T &t)
{
    GetColumn(t);
    return 0;
}

template<typename T, size_t... S>
void Connection::ExecuteQueryHelper(T &row, IndexSequence<S...>)
{
    std::initializer_list<int> {(ExecuteQueryHelperHelper(std::get<S>(row)), 0)...};
}

template<typename T>
void Connection::GetRow(T &row)
{
    using TupleType = typename std::decay<T>::type;
    ExecuteQueryHelper(row, MakeIndexSequence<std::tuple_size<TupleType>::value>{});
}

template<typename... Args>
bool Connection::ExecuteInsert(const std::string &tableName, const std::vector<std::tuple<Args...>> &data)
{
    uint32_t colNum = sizeof...(Args);
    sqlite3_exec(db_, "BEGIN", nullptr, nullptr, nullptr);
    if (!InsertCmd(tableName, colNum)) {
        return false;
    }
    for (const auto &row : data) {
        index_ = 0;
        sqlite3_reset(stmt_);
        InsertRow(row);
        auto rc = sqlite3_step(stmt_);
        if (rc != SQLITE_DONE) {
            LOG(ERROR) << "ExecuteInsert failed: " << rc << ", msg: " << sqlite3_errmsg(db_) << ", insert failed";
            if (sqlite3_exec(db_, "ROLLBACK", nullptr, nullptr, nullptr) != SQLITE_OK) {
                LOG(ERROR) << "ExecuteInsert failed: " << rc << ", rollback failed";
            }
            return false;
        }
    }
    sqlite3_exec(db_, "COMMIT", nullptr, nullptr, nullptr);
    return true;
}

template<typename... Args>
bool Connection::ExecuteQuery(const std::string &sql, std::vector<std::tuple<Args...>> &result)
{
    if (!QueryCmd(sql)) {
        return false;
    }
    while (true) {
        auto rc = sqlite3_step(stmt_);
        if (rc != SQLITE_ROW) {
            if (rc != SQLITE_DONE) {
                LOG(ERROR) << "ExecuteQuery failed: " << rc << ", msg: " << sqlite3_errmsg(db_) << ", query failed";
                return false;
            }
            break;
        }
        index_ = -1;
        std::tuple<Args...> row;
        GetRow(row);
        result.emplace_back(row);
    }
    return true;
}
} // namespace db
} // namespace ipc_monitor
} // namespace dynolog_npu

#endif // IPC_MONITOR_DB_CONNECTION_H
