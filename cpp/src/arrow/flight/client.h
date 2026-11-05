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

/// \brief Implementation of Flight RPC client.

#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "arrow/ipc/options.h"
#include "arrow/ipc/reader.h"
#include "arrow/ipc/writer.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/util/cancel.h"

#include "arrow/flight/type_fwd.h"
#include "arrow/flight/types.h"  // IWYU pragma: keep
#include "arrow/flight/visibility.h"

namespace arrow {

class RecordBatch;
class Schema;

namespace flight {

/// \brief A duration type for Flight call timeouts.
// 定义 Flight 调用超时的持续时间类型
typedef std::chrono::duration<double, std::chrono::seconds::period> TimeoutDuration;

/// \brief Hints to the underlying RPC layer for Arrow Flight calls.
// RPC 调用选项
class ARROW_FLIGHT_EXPORT FlightCallOptions {
 public:
  /// Create a default set of call options.
  FlightCallOptions();

  /// \brief An optional timeout for this call. Negative durations
  /// mean an implementation-defined default behavior will be used
  /// instead. This is the default value.
  // 调用的可选超时时间。负数表示使用实现定义的默认行为。
  TimeoutDuration timeout;

  /// \brief IPC reader options, if applicable for the call.
  // IPC 读取器选项。
  ipc::IpcReadOptions read_options;

  /// \brief IPC writer options, if applicable for the call.
  // IPC 写入器选项。
  ipc::IpcWriteOptions write_options;

  /// \brief Headers for client to add to context.
  // 客户端需要添加到 RPC 上下文中的自定义 HTTP 标头键值对列表。
  std::vector<std::pair<std::string, std::string>> headers;

  /// \brief A token to enable interactive user cancellation of long-running requests.
  // 用于支持用户交互式取消长时间运行请求的令牌。
  StopToken stop_token;

  /// \brief An optional memory manager to control where to allocate incoming data.
  // 可选的内存管理器，用于控制传入数据的分配位置。
  std::shared_ptr<MemoryManager> memory_manager;
};

/// \brief Indicate that the client attempted to write a message
///     larger than the soft limit set via write_size_limit_bytes.
// 写大小超限状态详情
// 当客户端尝试写入超过 write_size_limit_bytes 软限制的单批消息时触发的状态详情。
class ARROW_FLIGHT_EXPORT FlightWriteSizeStatusDetail : public arrow::StatusDetail {
 public:
  explicit FlightWriteSizeStatusDetail(int64_t limit, int64_t actual)
      : limit_(limit), actual_(actual) {}
  const char* type_id() const override;
  std::string ToString() const override;
  int64_t limit() const { return limit_; }
  int64_t actual() const { return actual_; }

  /// \brief Extract this status detail from a status, or return
  ///     nullptr if the status doesn't contain this status detail.
  static std::shared_ptr<FlightWriteSizeStatusDetail> UnwrapStatus(
      const arrow::Status& status);

 private:
  int64_t limit_;
  int64_t actual_;
};
// 客户端配置选项
struct ARROW_FLIGHT_EXPORT FlightClientOptions {
  /// \brief Root certificates to use for validating server
  /// certificates.
  // 用于验证服务器证书的根证书。
  std::string tls_root_certs;
  /// \brief Override the hostname checked by TLS. Use with caution.
  // 覆盖 TLS 检查的主机名。
  std::string override_hostname;
  /// \brief The client certificate to use if using Mutual TLS
  // 用于双向 TLS (Mutual TLS) 的客户端证书与私钥。
  std::string cert_chain;
  /// \brief The private key associated with the client certificate for Mutual TLS
   // 用于双向 TLS (Mutual TLS) 的客户端证书与私钥。
  std::string private_key;
  /// \brief A list of client middleware to apply.
  // 客户端中间件工厂列表。
  std::vector<std::shared_ptr<ClientMiddlewareFactory>> middleware;
  /// \brief A soft limit on the number of bytes to write in a single
  ///     batch when sending Arrow data to a server.
  ///
  /// Used to help limit server memory consumption. Only enabled if
  /// positive. When enabled, FlightStreamWriter.Write* may yield a
  /// IOError with error detail FlightWriteSizeStatusDetail.
  // 向服务器发送 Arrow 数据时单批写入的字节数软限制（默认 0，表示不限制）。
  int64_t write_size_limit_bytes = 0;

  /// \brief Generic connection options, passed to the underlying
  ///     transport; interpretation is implementation-dependent.
  // 传递给底层传输层的通用连接选项。
  std::vector<std::pair<std::string, std::variant<int, std::string>>> generic_options;

  /// \brief Use TLS without validating the server certificate. Use with caution.
  // 是否禁用服务器证书验证（谨慎使用）。
  bool disable_server_verification = false;

  /// \brief Get default options.
  // 获取默认配置选项
  static FlightClientOptions Defaults();
};

/// \brief A RecordBatchReader exposing Flight metadata and cancel
/// operations.

// Flight 流读取器
// Cancel()（尝试取消调用）、ToRecordBatches(stop_token)（将整个流消费为 Record Batch 向量）、ToTable(stop_token)（将整个流消费为 Table）、stats()（返回当前的读取统计信息）。
class ARROW_FLIGHT_EXPORT FlightStreamReader : public MetadataRecordBatchReader {
 public:
  /// \brief Try to cancel the call.
  virtual void Cancel() = 0;

  using MetadataRecordBatchReader::ToRecordBatches;
  /// \brief Consume entire stream as a vector of record batches
  virtual arrow::Result<std::vector<std::shared_ptr<RecordBatch>>> ToRecordBatches(
      const StopToken& stop_token) = 0;

  using MetadataRecordBatchReader::ToTable;
  /// \brief Consume entire stream as a Table
  arrow::Result<std::shared_ptr<Table>> ToTable(const StopToken& stop_token);

  using MetadataRecordBatchReader::stats;
  /// \brief Return current read statistics
  virtual arrow::ipc::ReadStats stats() const = 0;
};

// Silence warning
// "non dll-interface class RecordBatchReader used as base for dll-interface class"
#ifdef _MSC_VER
#  pragma warning(push)
#  pragma warning(disable : 4275)
#endif

/// \brief A RecordBatchWriter that also allows sending
/// application-defined metadata via the Flight protocol.
// Flight 流写入器
// DoneWriting()（指示应用程序已完成对该流的写入，区别于关闭整个流）。
class ARROW_FLIGHT_EXPORT FlightStreamWriter : public MetadataRecordBatchWriter {
 public:
  /// \brief Indicate that the application is done writing to this stream.
  ///
  /// The application may not write to this stream after calling
  /// this. This differs from closing the stream because this writer
  /// may represent only one half of a readable and writable stream.
  virtual Status DoneWriting() = 0;
};

#ifdef _MSC_VER
#  pragma warning(pop)
#endif

/// \brief A reader for application-specific metadata sent back to the
/// client during an upload.
// 元数据读取器
// 用于在上传过程中读取服务器发回的特定应用元数据。
// ReadMetadata(Buffer** out)（从服务器读取一条元数据消息）。
class ARROW_FLIGHT_EXPORT FlightMetadataReader {
 public:
  virtual ~FlightMetadataReader();
  /// \brief Read a message from the server.
  virtual Status ReadMetadata(std::shared_ptr<Buffer>* out) = 0;
};

/// \brief Client class for Arrow Flight RPC services.
// FlightClient 是 Apache Arrow Flight 服务端的 C++ 客户端核心类。它封装了基于 gRPC 和 Arrow Flight 协议的网络通信细节，负责与 Arrow Flight 服务端进行交互。
// 主要作用包括：
// 连接与认证：支持连接到指定的 Flight 服务端 URI（Location），并提供了灵活的 TLS 配置、客户端证书、基础认证（Basic Token）以及自定义的身份验证处理器（ClientAuthHandler）。
// 数据集元数据查询：允许客户端获取数据集的访问计划（GetFlightInfo、异步获取、长轮询 PollFlightInfo）、获取数据架构（GetSchema）以及列出服务器上可用的所有航班/数据集（ListFlights）。
// 高速数据传输：
// DoGet：从服务器下载/流式读取 Arrow 数据（以 FlightStreamReader 形式返回）。
// DoPut：向服务器上传/写入 Arrow 数据（返回写端 FlightStreamWriter 和元数据读端 FlightMetadataReader）。
// DoExchange：实现双向流式数据交换。
// 服务端动作与会话管理：支持列出和执行服务器自定义动作（ListActions、DoAction）、取消正在运行的查询（CancelFlightInfo）、延长端点有效期（RenewFlightEndpoint）以及管理服务端会话（Set/Get/Close Session Options）。

class ARROW_FLIGHT_EXPORT FlightClient {
 public:
  ~FlightClient();

  /// \brief Connect to an unauthenticated flight service
  /// \param[in] location the URI
  /// \return Arrow result with the created FlightClient, OK status may not indicate that
  /// the connection was successful
  // 通过指定的 URI（location）和可选的配置（options）连接到未认证的 Flight 服务端，返回创建的 FlightClient 实例。
  static arrow::Result<std::unique_ptr<FlightClient>> Connect(const Location& location);

  /// \brief Connect to an unauthenticated flight service
  /// \param[in] location the URI
  /// \param[in] options Other options for setting up the client
  /// \return Arrow result with the created FlightClient, OK status may not indicate that
  /// the connection was successful
  static arrow::Result<std::unique_ptr<FlightClient>> Connect(
      const Location& location, const FlightClientOptions& options);

  /// \brief Authenticate to the server using the given handler.
  /// \param[in] options Per-RPC options
  /// \param[in] auth_handler The authentication mechanism to use
  /// \return Status OK if the client authenticated successfully
  // 使用传入的自定义认证处理器（auth_handler）向服务器进行身份验证。
  Status Authenticate(const FlightCallOptions& options,
                      std::unique_ptr<ClientAuthHandler> auth_handler);

  /// \brief Authenticate to the server using basic HTTP style authentication.
  /// \param[in] options Per-RPC options
  /// \param[in] username Username to use
  /// \param[in] password Password to use
  /// \return Arrow result with bearer token and status OK if client authenticated
  /// successfully
  // 使用 HTTP 风格的基本认证（用户名和密码）向服务器认证，成功后返回 Bearer Token 和状态。
  arrow::Result<std::pair<std::string, std::string>> AuthenticateBasicToken(
      const FlightCallOptions& options, const std::string& username,
      const std::string& password);

  /// \brief Perform the indicated action, returning an iterator to the stream
  /// of results, if any
  /// \param[in] options Per-RPC options
  /// \param[in] action the action to be performed
  /// \return Arrow result with an iterator object for reading the returned results
 // 执行指定的服务器端自定义动作（Action），返回包含结果流的迭代器（ResultStream）。
  arrow::Result<std::unique_ptr<ResultStream>> DoAction(const FlightCallOptions& options,
                                                        const Action& action);
  arrow::Result<std::unique_ptr<ResultStream>> DoAction(const Action& action) {
    return DoAction({}, action);
  }

  /// \brief Perform the CancelFlightInfo action, returning a
  /// CancelFlightInfoResult
  ///
  /// \param[in] options Per-RPC options
  /// \param[in] request The CancelFlightInfoRequest
  /// \return Arrow result with a CancelFlightInfoResult
  // 执行 CancelFlightInfo 动作，请求取消指定的任务，返回取消结果。
  arrow::Result<CancelFlightInfoResult> CancelFlightInfo(
      const FlightCallOptions& options, const CancelFlightInfoRequest& request);
  arrow::Result<CancelFlightInfoResult> CancelFlightInfo(
      const CancelFlightInfoRequest& request) {
    return CancelFlightInfo({}, request);
  }

  /// \brief Perform the RenewFlightEndpoint action, returning a renewed
  /// FlightEndpoint
  ///
  /// \param[in] options Per-RPC options
  /// \param[in] request The RenewFlightEndpointRequest
  /// \return Arrow result with a renewed FlightEndpoint
  // 执行 RenewFlightEndpoint 动作，延长某个 FlightEndpoint 的有效期限并返回更新后的端点。
  arrow::Result<FlightEndpoint> RenewFlightEndpoint(
      const FlightCallOptions& options, const RenewFlightEndpointRequest& request);
  arrow::Result<FlightEndpoint> RenewFlightEndpoint(
      const RenewFlightEndpointRequest& request) {
    return RenewFlightEndpoint({}, request);
  }

  /// \brief Retrieve a list of available Action types
  /// \param[in] options Per-RPC options
  /// \return Arrow result with the available actions
  // 请求并返回服务器支持的所有可用 Action 类型列表。
  arrow::Result<std::vector<ActionType>> ListActions(const FlightCallOptions& options);
  arrow::Result<std::vector<ActionType>> ListActions() {
    return ListActions(FlightCallOptions());
  }

  /// \brief Request access plan for a single flight, which may be an existing
  /// dataset or a command to be executed
  /// \param[in] options Per-RPC options
  /// \param[in] descriptor the dataset request, whether a named dataset or
  /// command
  /// \return Arrow result with the FlightInfo describing where to access the dataset
  // 根据数据集描述符（descriptor，可以是命名数据集或命令）向服务器请求访问计划，返回描述数据访问位置的 FlightInfo。
  arrow::Result<std::unique_ptr<FlightInfo>> GetFlightInfo(
      const FlightCallOptions& options, const FlightDescriptor& descriptor);
  arrow::Result<std::unique_ptr<FlightInfo>> GetFlightInfo(
      const FlightDescriptor& descriptor) {
    return GetFlightInfo({}, descriptor);
  }

  /// \brief Asynchronous GetFlightInfo.
  /// \param[in] options Per-RPC options
  /// \param[in] descriptor the dataset request
  /// \param[in] listener Callbacks for response and RPC completion
  // 异步获取 FlightInfo，通过传入的监听器（listener）接收响应和 RPC 完成回调。
  void GetFlightInfoAsync(const FlightCallOptions& options,
                          const FlightDescriptor& descriptor,
                          std::shared_ptr<AsyncListener<FlightInfo>> listener);
  void GetFlightInfoAsync(const FlightDescriptor& descriptor,
                          std::shared_ptr<AsyncListener<FlightInfo>> listener) {
    return GetFlightInfoAsync({}, descriptor, std::move(listener));
  }

  /// \brief Asynchronous GetFlightInfo returning a Future.
  /// \param[in] options Per-RPC options
  /// \param[in] descriptor the dataset request
  // 异步获取 FlightInfo 并返回一个 arrow::Future<FlightInfo> 异步未来对象。
  arrow::Future<FlightInfo> GetFlightInfoAsync(const FlightCallOptions& options,
                                               const FlightDescriptor& descriptor);
  arrow::Future<FlightInfo> GetFlightInfoAsync(const FlightDescriptor& descriptor) {
    return GetFlightInfoAsync({}, descriptor);
  }

  /// \brief Request and poll a long running query
  /// \param[in] options Per-RPC options
  /// \param[in] descriptor the dataset request or a descriptor returned by a
  /// prior PollFlightInfo call
  /// \return Arrow result with the PollInfo describing the status of
  /// the requested query
  // 请求并轮询一个长时间运行的查询，返回描述查询状态的 PollInfo。
  arrow::Result<std::unique_ptr<PollInfo>> PollFlightInfo(
      const FlightCallOptions& options, const FlightDescriptor& descriptor);
  arrow::Result<std::unique_ptr<PollInfo>> PollFlightInfo(
      const FlightDescriptor& descriptor) {
    return PollFlightInfo({}, descriptor);
  }

  /// \brief Request schema for a single flight, which may be an existing
  /// dataset or a command to be executed
  /// \param[in] options Per-RPC options
  /// \param[in] descriptor the dataset request, whether a named dataset or
  /// command
  /// \return Arrow result with the SchemaResult describing the dataset schema
  // 请求指定数据集的 Arrow 结构定义（Schema），返回 SchemaResult。
  arrow::Result<std::unique_ptr<SchemaResult>> GetSchema(
      const FlightCallOptions& options, const FlightDescriptor& descriptor);

  arrow::Result<std::unique_ptr<SchemaResult>> GetSchema(
      const FlightDescriptor& descriptor) {
    return GetSchema({}, descriptor);
  }

  /// \brief List all available flights known to the server
  /// \return Arrow result with an iterator that returns a FlightInfo for each flight
  // 列出服务器已知的全部或符合过滤条件（criteria）的航班/数据集，返回一个可迭代的 FlightListing。
  arrow::Result<std::unique_ptr<FlightListing>> ListFlights();

  /// \brief List available flights given indicated filter criteria
  /// \param[in] options Per-RPC options
  /// \param[in] criteria the filter criteria (opaque)
  /// \return Arrow result with an iterator that returns a FlightInfo for each flight
  arrow::Result<std::unique_ptr<FlightListing>> ListFlights(
      const FlightCallOptions& options, const Criteria& criteria);

  /// \brief Given a flight ticket and schema, request to be sent the
  /// stream. Returns record batch stream reader
  /// \param[in] options Per-RPC options
  /// \param[in] ticket The flight ticket to use
  /// \return Arrow result with the returned RecordBatchReader
  // 根据传入的票据（Ticket）向服务器请求数据流，返回用于读取 Record Batch 的 FlightStreamReader。
  arrow::Result<std::unique_ptr<FlightStreamReader>> DoGet(
      const FlightCallOptions& options, const Ticket& ticket);
  arrow::Result<std::unique_ptr<FlightStreamReader>> DoGet(const Ticket& ticket) {
    return DoGet({}, ticket);
  }

  /// \brief DoPut return value
  struct DoPutResult {
    /// \brief a writer to write record batches to
    std::unique_ptr<FlightStreamWriter> writer;
    /// \brief a reader for application metadata from the server
    std::unique_ptr<FlightMetadataReader> reader;
  };
  /// \brief Upload data to a Flight described by the given
  /// descriptor. The caller must call Close() on the returned stream
  /// once they are done writing.
  ///
  /// The reader and writer are linked; closing the writer will also
  /// close the reader. Use \a DoneWriting to only close the write
  /// side of the channel.
  ///
  /// \param[in] options Per-RPC options
  /// \param[in] descriptor the descriptor of the stream
  /// \param[in] schema the schema for the data to upload
  /// \return Arrow result with a DoPutResult struct holding a reader and a writer
  // 向由 descriptor 描述的 Flight 服务端上传数据。返回一个包含写端 (writer) 和元数据读端 (reader) 的 DoPutResult 结构体。
  arrow::Result<DoPutResult> DoPut(const FlightCallOptions& options,
                                   const FlightDescriptor& descriptor,
                                   const std::shared_ptr<Schema>& schema);

  arrow::Result<DoPutResult> DoPut(const FlightDescriptor& descriptor,
                                   const std::shared_ptr<Schema>& schema) {
    return DoPut({}, descriptor, schema);
  }

  struct DoExchangeResult {
    std::unique_ptr<FlightStreamWriter> writer;
    std::unique_ptr<FlightStreamReader> reader;
  };
  // 执行双向流式数据交换（DoExchange），返回包含写端 (writer) 和读端 (reader) 的 DoExchangeResult。
  arrow::Result<DoExchangeResult> DoExchange(const FlightCallOptions& options,
                                             const FlightDescriptor& descriptor);
  arrow::Result<DoExchangeResult> DoExchange(const FlightDescriptor& descriptor) {
    return DoExchange({}, descriptor);
  }

  /// \brief Set server session option(s) by name/value. Sessions are generally
  /// persisted via HTTP cookies.
  /// \param[in] options Per-RPC options
  /// \param[in] request The server session options to set
  // 按名/值设置服务器会话选项（通常通过 HTTP Cookie 持久化）。
  ::arrow::Result<SetSessionOptionsResult> SetSessionOptions(
      const FlightCallOptions& options, const SetSessionOptionsRequest& request);

  /// \brief Get the current server session options. The session is generally
  /// accessed via an HTTP cookie.
  /// \param[in] options Per-RPC options
  /// \param[in] request The (empty) GetSessionOptions request object.
  // 获取当前的服务器会话选项。
  ::arrow::Result<GetSessionOptionsResult> GetSessionOptions(
      const FlightCallOptions& options, const GetSessionOptionsRequest& request);

  /// \brief Close/invalidate the current server session. The session is generally
  /// accessed via an HTTP cookie.
  /// \param[in] options Per-RPC options
  /// \param[in] request The (empty) CloseSession request object.
  // 显式关闭或使当前的服务器会话失效。
  ::arrow::Result<CloseSessionResult> CloseSession(const FlightCallOptions& options,
                                                   const CloseSessionRequest& request);

  /// \brief Explicitly shut down and clean up the client.
  ///
  /// For backwards compatibility, this will be implicitly called by
  /// the destructor if not already called, but this gives the
  /// application no chance to handle errors, so it is recommended to
  /// explicitly close the client.
  ///
  /// \since 8.0.0
  // 显式关闭并清理客户端连接及相关资源，推荐显式调用以捕获潜在错误。
  Status Close();

  /// \brief Whether this client supports asynchronous methods.
  // 检查该客户端当前是否支持异步方法。
  bool supports_async() const;

  /// \brief Check whether this client supports asynchronous methods.
  ///
  /// This is like supports_async(), except that a detailed error message
  /// is returned if async support is not available.  If async support is
  /// available, this function returns successfully.
  // 检查异步支持情况；如果不支持，返回包含详细错误信息的 Status，支持则成功返回。
  Status CheckAsyncSupport() const;

 private:
  FlightClient();
  Status CheckOpen() const;
  std::unique_ptr<internal::ClientTransport> transport_;
  bool closed_;
  int64_t write_size_limit_bytes_;
};

}  // namespace flight
}  // namespace arrow
