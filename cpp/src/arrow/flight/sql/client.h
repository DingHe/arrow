// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "arrow/flight/client.h"
#include "arrow/flight/sql/types.h"
#include "arrow/flight/sql/visibility.h"
#include "arrow/flight/types.h"
#include "arrow/result.h"
#include "arrow/status.h"

namespace arrow {
namespace flight {
namespace sql {

class PreparedStatement;
class Transaction;
class Savepoint;

/// \brief A default transaction to use when the default behavior
///   (auto-commit) is desired.
ARROW_FLIGHT_SQL_EXPORT
const Transaction& no_transaction();

/// \brief Flight client with Flight SQL semantics.
///
/// Wraps a Flight client to provide the Flight SQL RPC calls.
// FlightSqlClient 是 Apache Arrow Flight SQL 的 C++ 客户端核心类。它封装了通用的 FlightClient，并在此基础之上实现了 Flight SQL RPC 语义。
// 统一的 SQL 接口：向支持 Flight SQL 协议的数据库或服务端发送 SQL 查询、更新（Update）、批量数据导入（Bulk Ingestion）以及 Substrait 执行计划。
// 标准元数据查询：提供了一套完整的方法来获取数据库的元数据信息，如目录（Catalogs）、数据库模式（DB Schemas）、表（Tables）、主键/外键（Primary/Foreign Keys）、表类型、数据类型（XDBC Type Info）及 SQL 支持信息等。
// 事务与保存点管理：支持事务的开启（Begin）、提交（Commit）和回滚（Rollback），以及在事务中创建和管理保存点（Savepoints）。
// 预编译语句（PreparedStatement）：支持创建和执行参数化的预编译语句，提高查询执行效率并防止 SQL 注入。
// 会话与底层控制：管理会话选项（Session Options）、取消查询、关闭连接以及直接调用底层的 Flight RPC 操作。
class ARROW_FLIGHT_SQL_EXPORT FlightSqlClient {
  friend class PreparedStatement;

 private:
 // 底层的 Apache Arrow FlightClient 智能指针实例
  // FlightSqlClient 的绝大多数高级 SQL 操作最终都会委托或转换成对该底层通用 Flight 客户端的 RPC 调用。
  std::shared_ptr<FlightClient> impl_;

 public:
  // 接收一个已经初始化好的通用 FlightClient 智能指针，将其包装并初始化 FlightSqlClient。
  explicit FlightSqlClient(std::shared_ptr<FlightClient> client);

  virtual ~FlightSqlClient() = default;

  /// \brief Execute a SQL query on the server.
  /// \param[in] options      RPC-layer hints for this call.
  /// \param[in] query        The UTF8-encoded SQL query to be executed.
  /// \param[in] transaction  A transaction to associate this query with.
  /// \return The FlightInfo describing where to access the dataset.
  // 在服务器端执行一条标准的 UTF-8 编码的 SQL 查询。
  // 参数：options（RPC 调用配置/超时等）、query（SQL 查询语句）、transaction（关联的事务，默认无事务/自动提交）。
  arrow::Result<std::unique_ptr<FlightInfo>> Execute(
      const FlightCallOptions& options, const std::string& query,
      const Transaction& transaction = no_transaction());

  /// \brief Execute a Substrait plan that returns a result set on the server.
  /// \param[in] options      RPC-layer hints for this call.
  /// \param[in] plan         The plan to be executed.
  /// \param[in] transaction  A transaction to associate this query with.
  /// \return The FlightInfo describing where to access the dataset.
  // 在服务器端执行一个返回结果集的 Substrait（跨语言可移植的序列化查询计划）计划。
  // 参数：options、plan（Substrait 计划对象）、transaction（关联事务）。
  arrow::Result<std::unique_ptr<FlightInfo>> ExecuteSubstrait(
      const FlightCallOptions& options, const SubstraitPlan& plan,
      const Transaction& transaction = no_transaction());

  /// \brief Get the result set schema from the server.
  // 向服务器请求获取指定 SQL 查询语句返回结果集的 Schema（数据结构定义），而不实际传输数据。
  // 包含结果集结构信息的 SchemaResult
  arrow::Result<std::unique_ptr<SchemaResult>> GetExecuteSchema(
      const FlightCallOptions& options, const std::string& query,
      const Transaction& transaction = no_transaction());

  /// \brief Get the result set schema from the server.
  // 向服务器请求获取指定 Substrait 计划返回结果集的 Schema。
  // 结构信息 SchemaResult。
  arrow::Result<std::unique_ptr<SchemaResult>> GetExecuteSubstraitSchema(
      const FlightCallOptions& options, const SubstraitPlan& plan,
      const Transaction& transaction = no_transaction());

  /// \brief Execute an update query on the server.
  /// \param[in] options      RPC-layer hints for this call.
  /// \param[in] query        The UTF8-encoded SQL query to be executed.
  /// \param[in] transaction  A transaction to associate this query with.
  /// \return The quantity of rows affected by the operation.
  // 在服务器端执行一条不返回结果集的 SQL 更新、插入或删除语句（DML/DDL）。
  // 返回受该操作影响的行数 (int64_t)。
  arrow::Result<int64_t> ExecuteUpdate(const FlightCallOptions& options,
                                       const std::string& query,
                                       const Transaction& transaction = no_transaction());

  /// \brief Execute a Substrait plan that does not return a result set on the server.
  /// \param[in] options      RPC-layer hints for this call.
  /// \param[in] plan         The plan to be executed.
  /// \param[in] transaction  A transaction to associate this query with.
  /// \return The FlightInfo describing where to access the dataset.
  // 执行一个不返回结果集的 Substrait 更新计划。
  // 返回值：受影响的行数 (int64_t)。
  arrow::Result<int64_t> ExecuteSubstraitUpdate(
      const FlightCallOptions& options, const SubstraitPlan& plan,
      const Transaction& transaction = no_transaction());

  /// \brief Execute a bulk ingestion to the server.
  /// \param[in] options                   RPC-layer hints for this call.
  /// \param[in] reader                    The records to ingest.
  /// \param[in] table_definition_options  The behavior for handling the table definition.
  /// \param[in] table                     The destination table to load into.
  /// \param[in] schema                    The DB schema of the destination table.
  /// \param[in] catalog                   The catalog of the destination table.
  /// \param[in] temporary                 Use a temporary table.
  /// \param[in] transaction               Ingest as part of this transaction.
  /// \param[in] ingest_options            Additional, backend-specific options.
  /// \return The number of rows ingested to the server.
  // 向服务器端执行批量数据导入（Bulk Ingestion）。
  // 包含数据的 reader、表定义处理行为选项、目标表名、数据库模式、目录、是否为临时表、事务及后端特定的扩展选项。
  arrow::Result<int64_t> ExecuteIngest(
      const FlightCallOptions& options, const std::shared_ptr<RecordBatchReader>& reader,
      const TableDefinitionOptions& table_definition_options, const std::string& table,
      const std::optional<std::string>& schema, const std::optional<std::string>& catalog,
      const bool temporary, const Transaction& transaction = no_transaction(),
      const std::unordered_map<std::string, std::string>& ingest_options = {});

  /// \brief Request a list of catalogs.
  /// \param[in] options      RPC-layer hints for this call.
  /// \return The FlightInfo describing where to access the dataset.
  // 请求获取数据库中所有 Catalog（目录）的列表。
  arrow::Result<std::unique_ptr<FlightInfo>> GetCatalogs(
      const FlightCallOptions& options);

  /// \brief Get the catalogs schema from the server (should be
  ///   identical to SqlSchema::GetCatalogsSchema).
  // 获取 Catalog 列表对应的结果集 Schema。
  arrow::Result<std::unique_ptr<SchemaResult>> GetCatalogsSchema(
      const FlightCallOptions& options);

  /// \brief Request a list of database schemas.
  /// \param[in] options                   RPC-layer hints for this call.
  /// \param[in] catalog                   The catalog.
  /// \param[in] db_schema_filter_pattern  The schema filter pattern.
  /// \return The FlightInfo describing where to access the dataset.
  // 请求获取数据库 Schema（模式）的列表，支持通过 Catalog 和模式名称过滤模式。
  arrow::Result<std::unique_ptr<FlightInfo>> GetDbSchemas(
      const FlightCallOptions& options, const std::string* catalog,
      const std::string* db_schema_filter_pattern);

  /// \brief Get the database schemas schema from the server (should be
  ///   identical to SqlSchema::GetDbSchemasSchema).
  // 获取数据库 Schema 列表对应的 Schema 定义。
  arrow::Result<std::unique_ptr<SchemaResult>> GetDbSchemasSchema(
      const FlightCallOptions& options);

  /// \brief Given a flight ticket and schema, request to be sent the
  /// stream. Returns record batch stream reader
  /// \param[in] options Per-RPC options
  /// \param[in] ticket The flight ticket to use
  /// \return The returned RecordBatchReader
  // 根据传入的 Flight 票据（Ticket）和 Schema，向服务器请求具体的数据流。
  // 返回用于读取 Record Batch 流的 FlightStreamReader。
  virtual arrow::Result<std::unique_ptr<FlightStreamReader>> DoGet(
      const FlightCallOptions& options, const Ticket& ticket);

  /// \brief Request a list of tables.
  /// \param[in] options                   RPC-layer hints for this call.
  /// \param[in] catalog                   The catalog.
  /// \param[in] db_schema_filter_pattern  The schema filter pattern.
  /// \param[in] table_filter_pattern      The table filter pattern.
  /// \param[in] include_schema            True to include the schema upon return,
  ///                                      false to not include the schema.
  /// \param[in] table_types               The table types to include.
  /// \return The FlightInfo describing where to access the dataset.
  // 请求获取数据库中的表列表，支持按 Catalog、Schema、表名模式、表类型过滤，并可选择是否在返回结果中包含表的 Schema。
  arrow::Result<std::unique_ptr<FlightInfo>> GetTables(
      const FlightCallOptions& options, const std::string* catalog,
      const std::string* db_schema_filter_pattern,
      const std::string* table_filter_pattern, bool include_schema,
      const std::vector<std::string>* table_types);

  /// \brief Get the tables schema from the server (should be
  ///   identical to SqlSchema::GetTablesSchema).
  // 获取表列表查询对应的 Schema 定义。
  arrow::Result<std::unique_ptr<SchemaResult>> GetTablesSchema(
      const FlightCallOptions& options, bool include_schema);

  /// \brief Request the primary keys for a table.
  /// \param[in] options          RPC-layer hints for this call.
  /// \param[in] table_ref        The table reference.
  /// \return The FlightInfo describing where to access the dataset.
  // 请求获取指定表的主键列信息。
  arrow::Result<std::unique_ptr<FlightInfo>> GetPrimaryKeys(
      const FlightCallOptions& options, const TableRef& table_ref);

  /// \brief Get the primary keys schema from the server (should be
  ///   identical to SqlSchema::GetPrimaryKeysSchema).
  // 获取主键信息对应的 Schema 定义。
  arrow::Result<std::unique_ptr<SchemaResult>> GetPrimaryKeysSchema(
      const FlightCallOptions& options);

  /// \brief Retrieves a description about the foreign key columns that reference the
  /// primary key columns of the given table.
  /// \param[in] options          RPC-layer hints for this call.
  /// \param[in] table_ref        The table reference.
  /// \return The FlightInfo describing where to access the dataset.
  // 获取引用给定表的主键的外键列信息（被导出的外键）。
  arrow::Result<std::unique_ptr<FlightInfo>> GetExportedKeys(
      const FlightCallOptions& options, const TableRef& table_ref);

  /// \brief Get the exported keys schema from the server (should be
  ///   identical to SqlSchema::GetExportedKeysSchema).
  // 获取导出键信息对应的 Schema 定义。
  arrow::Result<std::unique_ptr<SchemaResult>> GetExportedKeysSchema(
      const FlightCallOptions& options);

  /// \brief Retrieves the foreign key columns for the given table.
  /// \param[in] options          RPC-layer hints for this call.
  /// \param[in] table_ref        The table reference.
  /// \return The FlightInfo describing where to access the dataset.
  // 获取给定表所导入的外键列信息（引用其他表主键的外键）。
  arrow::Result<std::unique_ptr<FlightInfo>> GetImportedKeys(
      const FlightCallOptions& options, const TableRef& table_ref);

  /// \brief Get the imported keys schema from the server (should be
  ///   identical to SqlSchema::GetImportedKeysSchema).
  // 获取导入键信息对应的 Schema 定义。
  arrow::Result<std::unique_ptr<SchemaResult>> GetImportedKeysSchema(
      const FlightCallOptions& options);

  /// \brief Retrieves a description of the foreign key columns in the given foreign key
  ///        table that reference the primary key or the columns representing a unique
  ///        constraint of the parent table (could be the same or a different table).
  /// \param[in] options        RPC-layer hints for this call.
  /// \param[in] pk_table_ref   The table reference that exports the key.
  /// \param[in] fk_table_ref   The table reference that imports the key.
  /// \return The FlightInfo describing where to access the dataset.
  // 获取两个表之间的外键交叉引用信息（外键表中的列如何引用主键表的约束）。
  arrow::Result<std::unique_ptr<FlightInfo>> GetCrossReference(
      const FlightCallOptions& options, const TableRef& pk_table_ref,
      const TableRef& fk_table_ref);

  /// \brief Get the cross reference schema from the server (should be
  ///   identical to SqlSchema::GetCrossReferenceSchema).
  // 获取交叉引用对应的 Schema 定义。
  arrow::Result<std::unique_ptr<SchemaResult>> GetCrossReferenceSchema(
      const FlightCallOptions& options);

  /// \brief Request a list of table types.
  /// \param[in] options          RPC-layer hints for this call.
  /// \return The FlightInfo describing where to access the dataset.
  // 请求获取数据库支持的表类型列表（如 TABLE、VIEW 等）。
  arrow::Result<std::unique_ptr<FlightInfo>> GetTableTypes(
      const FlightCallOptions& options);

  /// \brief Get the table types schema from the server (should be
  ///   identical to SqlSchema::GetTableTypesSchema).
  // 获取表类型列表对应的 Schema 定义。
  arrow::Result<std::unique_ptr<SchemaResult>> GetTableTypesSchema(
      const FlightCallOptions& options);

  /// \brief Request the information about all the data types supported.
  /// \param[in] options          RPC-layer hints for this call.
  /// \return The FlightInfo describing where to access the dataset.
  // 请求获取数据库支持的全部 XDBC 数据类型信息。
  arrow::Result<std::unique_ptr<FlightInfo>> GetXdbcTypeInfo(
      const FlightCallOptions& options);

  /// \brief Request the information about all the data types supported.
  /// \param[in] options          RPC-layer hints for this call.
  /// \param[in] data_type        The data type to search for as filtering.
  /// \return The FlightInfo describing where to access the dataset.
  // 请求获取指定特定数据类型（data_type）的 XDBC 类型信息。
  arrow::Result<std::unique_ptr<FlightInfo>> GetXdbcTypeInfo(
      const FlightCallOptions& options, int data_type);

  /// \brief Get the type info schema from the server (should be
  ///   identical to SqlSchema::GetXdbcTypeInfoSchema).
 // 获取 XDBC 类型信息对应的 Schema 定义。
  arrow::Result<std::unique_ptr<SchemaResult>> GetXdbcTypeInfoSchema(
      const FlightCallOptions& options);

  /// \brief Request a list of SQL information.
  /// \param[in] options RPC-layer hints for this call.
  /// \param[in] sql_info the SQL info required.
  /// \return The FlightInfo describing where to access the dataset.
  // 请求获取一组特定的 SQL 服务端元数据/功能支持信息（通过传入整数标识符向量）。
  arrow::Result<std::unique_ptr<FlightInfo>> GetSqlInfo(const FlightCallOptions& options,
                                                        const std::vector<int>& sql_info);

  /// \brief Get the SQL information schema from the server (should be
  ///   identical to SqlSchema::GetSqlInfoSchema).
  // 获取 SQL 信息对应的 Schema 定义。
  arrow::Result<std::unique_ptr<SchemaResult>> GetSqlInfoSchema(
      const FlightCallOptions& options);

  /// \brief Create a prepared statement object.
  /// \param[in] options              RPC-layer hints for this call.
  /// \param[in] query                The query that will be executed.
  /// \param[in] transaction          A transaction to associate this query with.
  /// \return The created prepared statement.
  // 在服务器端准备一条参数化 SQL 查询，创建一个预编译语句对象。
  arrow::Result<std::shared_ptr<PreparedStatement>> Prepare(
      const FlightCallOptions& options, const std::string& query,
      const Transaction& transaction = no_transaction());

  /// \brief Create a prepared statement object.
  /// \param[in] options              RPC-layer hints for this call.
  /// \param[in] plan                 The Substrait plan that will be executed.
  /// \param[in] transaction          A transaction to associate this query with.
  /// \return The created prepared statement.
  // 基于 Substrait 计划创建预编译语句对象。
  arrow::Result<std::shared_ptr<PreparedStatement>> PrepareSubstrait(
      const FlightCallOptions& options, const SubstraitPlan& plan,
      const Transaction& transaction = no_transaction());

  /// \brief Call the underlying Flight client's GetFlightInfo.
  // 直接调用底层 FlightClient 的 GetFlightInfo。
  virtual arrow::Result<std::unique_ptr<FlightInfo>> GetFlightInfo(
      const FlightCallOptions& options, const FlightDescriptor& descriptor) {
    return impl_->GetFlightInfo(options, descriptor);
  }

  /// \brief Call the underlying Flight client's GetSchema.
  // 直接调用底层 FlightClient 的 GetSchema。
  virtual arrow::Result<std::unique_ptr<SchemaResult>> GetSchema(
      const FlightCallOptions& options, const FlightDescriptor& descriptor) {
    return impl_->GetSchema(options, descriptor);
  }

  /// \brief Begin a new transaction.
  // 在服务器上开启一个新的事务。返回一个 Transaction 对象。
  ::arrow::Result<Transaction> BeginTransaction(const FlightCallOptions& options);

  /// \brief Create a new savepoint within a transaction.
  /// \param[in] options      RPC-layer hints for this call.
  /// \param[in] transaction  The parent transaction.
  /// \param[in] name         A friendly name for the savepoint.
  // 在指定父事务中创建一个新的保存点（Savepoint）。
  ::arrow::Result<Savepoint> BeginSavepoint(const FlightCallOptions& options,
                                            const Transaction& transaction,
                                            const std::string& name);

  /// \brief Commit a transaction.
  ///
  /// After this, the transaction and all associated savepoints will
  /// be invalidated.
  ///
  /// \param[in] options      RPC-layer hints for this call.
  /// \param[in] transaction  The transaction.
  // 提交指定的事务。提交后该事务及其所有关联的保存点将失效。
  Status Commit(const FlightCallOptions& options, const Transaction& transaction);

  /// \brief Release a savepoint.
  ///
  /// After this, the savepoint (and all savepoints created after it) will be invalidated.
  ///
  /// \param[in] options      RPC-layer hints for this call.
  /// \param[in] savepoint    The savepoint.
  // 释放指定的保存点。释放后该保存点（及其之后创建的保存点）将失效。
  Status Release(const FlightCallOptions& options, const Savepoint& savepoint);

  /// \brief Rollback a transaction.
  ///
  /// After this, the transaction and all associated savepoints will be invalidated.
  ///
  /// \param[in] options      RPC-layer hints for this call.
  /// \param[in] transaction  The transaction.
  // 回滚指定的事务。回滚后该事务及所有关联的保存点将失效。
  Status Rollback(const FlightCallOptions& options, const Transaction& transaction);

  /// \brief Rollback a savepoint.
  ///
  /// After this, the savepoint will still be valid, but all
  /// savepoints created after it will be invalidated.
  ///
  /// \param[in] options      RPC-layer hints for this call.
  /// \param[in] savepoint    The savepoint.
  // 回滚到指定的保存点。回滚后该保存点本身保持有效，但在其之后创建的保存点会失效。
  Status Rollback(const FlightCallOptions& options, const Savepoint& savepoint);

  /// \brief Explicitly cancel a FlightInfo.
  ///
  /// \param[in] options      RPC-layer hints for this call.
  /// \param[in] request      The CancelFlightInfoRequest.
  /// \return Arrow result with a canceled result.
  // 显式取消一个正在执行的 FlightInfo 请求。
  ::arrow::Result<CancelFlightInfoResult> CancelFlightInfo(
      const FlightCallOptions& options, const CancelFlightInfoRequest& request) {
    return impl_->CancelFlightInfo(options, request);
  }

  /// \brief Explicitly cancel a query.
  ///
  /// \param[in] options      RPC-layer hints for this call.
  /// \param[in] info         The FlightInfo of the query to cancel.
  ///
  /// \deprecated Deprecated since 13.0.0. Use CancelFlightInfo()
  /// instead. If you can assume that a server requires 13.0.0 or
  /// later, you can always use CancelFlightInfo(). Otherwise, you may
  /// need to use CancelQuery() and/or CancelFlightInfo().
  ARROW_DEPRECATED(
      "Deprecated in 13.0.0. Use CancelFlightInfo() instead. "
      "If you can assume that a server requires 13.0.0 or later, "
      "you can always use CancelFLightInfo(). Otherwise, you "
      "may need to use CancelQuery() and/or CancelFlightInfo()")
  // 显式取消一个查询（自 13.0.0 版本起已废弃，建议使用 CancelFlightInfo）。
  ::arrow::Result<CancelResult> CancelQuery(const FlightCallOptions& options,
                                            const FlightInfo& info);

  /// \brief Sets session options.
  ///
  /// \param[in] options            RPC-layer hints for this call.
  /// \param[in] request            The session options to set.
  // 设置当前连接的会话选项。
  ::arrow::Result<SetSessionOptionsResult> SetSessionOptions(
      const FlightCallOptions& options, const SetSessionOptionsRequest& request) {
    return impl_->SetSessionOptions(options, request);
  }

  /// \brief Gets current session options.
  ///
  /// \param[in] options            RPC-layer hints for this call.
  /// \param[in] request            The (empty) GetSessionOptions request object.
  // 获取当前的会话选项。
  ::arrow::Result<GetSessionOptionsResult> GetSessionOptions(
      const FlightCallOptions& options, const GetSessionOptionsRequest& request) {
    return impl_->GetSessionOptions(options, request);
  }

  /// \brief Explicitly closes the session if applicable.
  ///
  /// \param[in] options      RPC-layer hints for this call.
  /// \param[in] request      The (empty) CloseSession request object.
  // 显式关闭当前的数据库会话。
  ::arrow::Result<CloseSessionResult> CloseSession(const FlightCallOptions& options,
                                                   const CloseSessionRequest& request) {
    return impl_->CloseSession(options, request);
  }

  /// \brief Extends the expiration of a FlightEndpoint.
  ///
  /// \param[in] options      RPC-layer hints for this call.
  /// \param[in] request      The RenewFlightEndpointRequest.
  /// \return Arrow result with a renewed FlightEndpoint
  // 延长某个 FlightEndpoint 的过期时间。
  ::arrow::Result<FlightEndpoint> RenewFlightEndpoint(
      const FlightCallOptions& options, const RenewFlightEndpointRequest& request) {
    return impl_->RenewFlightEndpoint(options, request);
  }

  /// \brief Explicitly shut down and clean up the client.
  // 显式关闭客户端连接并清理相关资源。
  Status Close();

  /// \brief Wrapper around FlightClient::DoGet.
  ///
  /// \internal
  /// Don't call this directly.
  /// \endinternal
  // 对底层 FlightClient::DoPut 的包装，用于向服务器发送数据流。
  virtual ::arrow::Result<FlightClient::DoPutResult> DoPut(
      const FlightCallOptions& options, const FlightDescriptor& descriptor,
      const std::shared_ptr<Schema>& schema) {
    return impl_->DoPut(options, descriptor, schema);
  }

  /// \brief Wrapper around FlightClient::DoPut. Don't call this directly.
  ///
  /// \internal
  /// Don't call this directly.
  /// \endinternal
  // 对底层 FlightClient::DoAction 的包装，用于执行自定义 RPC 动作。
  virtual ::arrow::Result<std::unique_ptr<ResultStream>> DoAction(
      const FlightCallOptions& options, const Action& action) {
    return impl_->DoAction(options, action);
  }
};

/// \brief A prepared statement that can be executed.
// 代表一个服务器端的预编译 SQL 语句或计划。
// 客户端通过它来绑定参数、执行查询或更新、获取参数及结果集的 Schema。
class ARROW_FLIGHT_SQL_EXPORT PreparedStatement {
 public:
  /// \brief Create a new prepared statement. However, applications
  /// should generally use FlightSqlClient::Prepare.
  ///
  /// \param[in] client                Client object used to make the RPC requests.
  /// \param[in] handle                Handle for this prepared statement.
  /// \param[in] dataset_schema        Schema of the resulting dataset.
  /// \param[in] parameter_schema      Schema of the parameters (if any).
  PreparedStatement(FlightSqlClient* client, std::string handle,
                    std::shared_ptr<Schema> dataset_schema,
                    std::shared_ptr<Schema> parameter_schema);

  /// \brief Default destructor for the PreparedStatement class.
  /// The destructor will call the Close method from the class in order,
  /// to send a request to close the PreparedStatement.
  /// NOTE: It is best to explicitly close the PreparedStatement, otherwise
  /// errors can't be caught.
  ~PreparedStatement();

  /// \brief Create a PreparedStatement by parsing the server response.
  static arrow::Result<std::shared_ptr<PreparedStatement>> ParseResponse(
      FlightSqlClient* client, std::unique_ptr<ResultStream> results);

  /// \brief Executes the prepared statement query on the server.
  /// \return A FlightInfo object representing the stream(s) to fetch.
  arrow::Result<std::unique_ptr<FlightInfo>> Execute(
      const FlightCallOptions& options = {});

  /// \brief Executes the prepared statement update query on the server.
  /// \return The number of rows affected.
  arrow::Result<int64_t> ExecuteUpdate(const FlightCallOptions& options = {});

  /// \brief Retrieve the parameter schema from the query.
  /// \return The parameter schema from the query.
  const std::shared_ptr<Schema>& parameter_schema() const;

  /// \brief Retrieve the ResultSet schema from the query.
  /// \return The ResultSet schema from the query.
  const std::shared_ptr<Schema>& dataset_schema() const;

  /// \brief Set a RecordBatch that contains the parameters that will be bound.
  Status SetParameters(std::shared_ptr<RecordBatch> parameter_binding);

  /// \brief Set a RecordBatchReader that contains the parameters that will be bound.
  Status SetParameters(std::shared_ptr<RecordBatchReader> parameter_binding);

  /// \brief Re-request the result set schema from the server (should
  ///   be identical to dataset_schema).
  arrow::Result<std::unique_ptr<SchemaResult>> GetSchema(
      const FlightCallOptions& options = {});

  /// \brief Close the prepared statement so the server can free up any resources.
  ///
  /// After this, the prepared statement may not be used anymore.
  Status Close(const FlightCallOptions& options = {});

  /// \brief Check if the prepared statement is closed.
  /// \return The state of the prepared statement.
  bool IsClosed() const;

 private:
  FlightSqlClient* client_;
  std::string handle_;
  std::shared_ptr<Schema> dataset_schema_;
  std::shared_ptr<Schema> parameter_schema_;
  std::shared_ptr<RecordBatchReader> parameter_binding_;
  bool is_closed_;
};

/// \brief A handle for a server-side savepoint.
// 封装了服务器端事务保存点的句柄。
class ARROW_FLIGHT_SQL_EXPORT Savepoint {
 public:
  explicit Savepoint(std::string savepoint_id) : savepoint_id_(std::move(savepoint_id)) {}
  const std::string& savepoint_id() const { return savepoint_id_; }
  bool is_valid() const { return !savepoint_id_.empty(); }

 private:
  // 保存点的唯一字符串标识 ID。
  std::string savepoint_id_;
};

/// \brief A handle for a server-side transaction.
// 封装了服务器端事务的句柄。
class ARROW_FLIGHT_SQL_EXPORT Transaction {
 public:
  explicit Transaction(std::string transaction_id)
      : transaction_id_(std::move(transaction_id)) {}
  const std::string& transaction_id() const { return transaction_id_; }
  bool is_valid() const { return !transaction_id_.empty(); }

 private:
  // 事务的唯一字符串标识 ID。
  std::string transaction_id_;
};

}  // namespace sql
}  // namespace flight
}  // namespace arrow
