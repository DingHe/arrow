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

// Read Arrow files and streams

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "arrow/io/caching.h"
#include "arrow/io/type_fwd.h"
#include "arrow/ipc/message.h"
#include "arrow/ipc/options.h"
#include "arrow/record_batch.h"
#include "arrow/result.h"
#include "arrow/type_fwd.h"
#include "arrow/util/async_generator.h"
#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"

namespace arrow {
namespace ipc {

class DictionaryMemo;
struct IpcPayload;

using RecordBatchReader = ::arrow::RecordBatchReader;

// ReadStats 是一个简单的结构体（POD），专门用于记录在处理 Arrow IPC（进程间通信）流或文件读取过程中的统计信息。
// ReadStats 的主要作用是监控和审计 IPC 解码过程。
// 由于 Arrow 数据不仅包含普通的“数据记录批次（Record Batches）”，还可能包含“字典批次（Dictionary Batches）”，尤其是在处理带有字典编码（Dictionary Encoding）的数据流时。ReadStats 提供了对流中不同类型消息计数的透明度。
struct ReadStats {
  /// Number of IPC messages read.
  // 读取到的 IPC 消息总数。
  // 在 Arrow IPC 协议中，所有的内容（Schema、Dictionary Batch、Record Batch）都被包装在“消息（Message）”中。这个指标是所有读取到的消息的累加，是衡量流活跃度的基本指标。
  int64_t num_messages = 0;
  /// Number of record batches read.
  // 读取到的记录批次（Record Batch）数量。
  // 这是用户最关心的数据指标，代表了包含实际业务数据的批次数量。通常 RecordBatchReader 每成功产生一个 RecordBatch 对象，该值就会加 1。
  int64_t num_record_batches = 0;
  /// Number of dictionary batches read.
  ///
  /// Note: num_dictionary_batches >= num_dictionary_deltas + num_replaced_dictionaries
  // 读取到的字典批次（Dictionary Batch）总数
  // 当某一列使用字典编码时，流中会包含专门定义或更新字典的消息。
  int64_t num_dictionary_batches = 0;

  /// Number of dictionary deltas read.
  // 读取到的字典增量（Dictionary Deltas）数量。
  // 在长流传输中，为了节省空间，系统可能只发送字典中新增的部分（Delta）。该指标记录了这种“增量更新”发生的次数。
  int64_t num_dictionary_deltas = 0;
  /// Number of replaced dictionaries (i.e. where a dictionary batch replaces
  /// an existing dictionary with an unrelated new dictionary).
  // 读取到的字典替换（Replaced Dictionaries）数量。
  // 当一个字典批次发送了全新的数据，并要求丢弃旧字典改用新字典时，记为一次替换。这通常发生在数据分布发生剧烈变化，导致旧字典不再适用时。
  int64_t num_replaced_dictionaries = 0;
};

/// \brief Synchronous batch stream reader that reads from io::InputStream
///
/// This class reads the schema (plus any dictionaries) as the first messages
/// in the stream, followed by record batches. For more granular zero-copy
/// reads see the ReadRecordBatch functions
// 在 Apache Arrow 项目中，RecordBatchStreamReader 是处理 Arrow IPC 流格式（Streaming Format） 的核心类。它继承自 RecordBatchReader，专门用于从连续的字节流中同步读取数据。
// 该类的主要作用是解析 Arrow 流式协议。
// Arrow 的 IPC 协议分为两种主要格式：文件格式（File Format）和流格式（Stream Format）。RecordBatchStreamReader 负责处理后者。
// 顺序解析：流格式是专为单次遍历设计的。它假设数据是顺序到达的（例如通过 TCP 网络连接或管道）。
// 自描述性：它首先从流中读取 Schema（模式）和 DictionaryBatches（字典块），建立元数据上下文，然后按顺序读取一个或多个 RecordBatches。
// 同步读取：它提供同步接口，调用者在请求下一个批次时会阻塞，直到流中有足够的数据被解码。
class ARROW_EXPORT RecordBatchStreamReader : public RecordBatchReader {
 public:
  /// Create batch reader from generic MessageReader.
  /// This will take ownership of the given MessageReader.
  ///
  /// \param[in] message_reader a MessageReader implementation
  /// \param[in] options any IPC reading options (optional)
  /// \return the created batch reader
  // Open 方法有三个重载版本，用于从不同的数据源初始化读取器：
  // 从一个底层的 MessageReader 创建读取器。
  // 这是最底层的构造方式。
  // MessageReader 负责将原始字节封装成 Arrow 的 Message 对象（如 Schema 消息或 RecordBatch 消息）。
  // 该方法会接管 message_reader 的所有权。
  static Result<std::shared_ptr<RecordBatchStreamReader>> Open(
      std::unique_ptr<MessageReader> message_reader,
      const IpcReadOptions& options = IpcReadOptions::Defaults());

  /// \brief Record batch stream reader from InputStream
  ///
  /// \param[in] stream an input stream instance. Must stay alive throughout
  /// lifetime of stream reader
  /// \param[in] options any IPC reading options (optional)
  /// \return the created batch reader
  // 直接从一个原始的输入流指针创建读取器。
  // 适用于流的生命周期由外部管理的情况。注意：调用者必须确保 stream 在 RecordBatchStreamReader 存活期间不被销毁。
  static Result<std::shared_ptr<RecordBatchStreamReader>> Open(
      io::InputStream* stream,
      const IpcReadOptions& options = IpcReadOptions::Defaults());

  /// \brief Open stream and retain ownership of stream object
  /// \param[in] stream the input stream
  /// \param[in] options any IPC reading options (optional)
  /// \return the created batch reader
  // 从共享指针形式的输入流创建读取器。
  // 这是最常用的版本。它会增加 stream 的引用计数，从而确保底层流对象的生命周期与读取器绑定。
  static Result<std::shared_ptr<RecordBatchStreamReader>> Open(
      const std::shared_ptr<io::InputStream>& stream,
      const IpcReadOptions& options = IpcReadOptions::Defaults());

  /// \brief Return current read statistics
  virtual ReadStats stats() const = 0;
};

/// \brief Reads the record batch file format
// 在 Apache Arrow 项目中，RecordBatchFileReader 是处理 Arrow IPC 文件格式（File Format / Random Access Format） 的核心类。
// 与流式格式不同，文件格式支持随机访问，允许用户直接跳到文件的任何部分读取特定的数据块。
// RecordBatchFileReader 的设计目标是高效地管理和读取存储在磁盘或内存映射文件中的 Arrow 数据。
// 随机访问（Random Access）：这是它与 RecordBatchStreamReader 最大的区别。通过读取位于文件末尾的“页脚”（Footer），它能获取所有数据块的偏移量，从而实现对任意一个 RecordBatch 的 $O(1)$ 级别定位。
// 零拷贝读取（Zero-copy）：如果底层使用的是内存映射文件（Memory Mapped File），该类可以直接返回指向磁盘缓冲区的指针，而无需将数据拷贝到用户态内存。
// 自包含元数据：它负责解析文件的 Schema、字典信息以及自定义的键值对元数据。
// 异步与并发支持：提供了异步打开和预缓冲（Pre-buffering）功能，显著提升在高延迟存储（如云存储）上的性能。
// 文件格式逻辑布局示意
//Header: ARROW1 幻数。
//Data Blocks: 连续存储的多个 RecordBatch 和 DictionaryBatch。
// Footer: 包含所有数据块的偏移量和长度。包含 Schema。
// Footer Length: 4 字节，记录 Footer 的大小。
// Magic Number: ARROW1（6 字节）。

class ARROW_EXPORT RecordBatchFileReader
    : public std::enable_shared_from_this<RecordBatchFileReader> {
 public:
  virtual ~RecordBatchFileReader() = default;

  /// \brief Open a RecordBatchFileReader
  ///
  /// Open a file-like object that is assumed to be self-contained; i.e., the
  /// end of the file interface is the end of the Arrow file. Note that there
  /// can be any amount of data preceding the Arrow-formatted data, because we
  /// need only locate the end of the Arrow file stream to discover the metadata
  /// and then proceed to read the data into memory.
  static Result<std::shared_ptr<RecordBatchFileReader>> Open(
      io::RandomAccessFile* file,
      const IpcReadOptions& options = IpcReadOptions::Defaults());

  /// \brief Open a RecordBatchFileReader
  /// If the file is embedded within some larger file or memory region, you can
  /// pass the absolute memory offset to the end of the file (which contains the
  /// metadata footer). The metadata must have been written with memory offsets
  /// relative to the start of the containing file
  ///
  /// \param[in] file the data source
  /// \param[in] footer_offset the position of the end of the Arrow file
  /// \param[in] options options for IPC reading
  /// \return the returned reader
  static Result<std::shared_ptr<RecordBatchFileReader>> Open(
      io::RandomAccessFile* file, int64_t footer_offset,
      const IpcReadOptions& options = IpcReadOptions::Defaults());

  /// \brief Version of Open that retains ownership of file
  ///
  /// \param[in] file the data source
  /// \param[in] options options for IPC reading
  /// \return the returned reader
  static Result<std::shared_ptr<RecordBatchFileReader>> Open(
      const std::shared_ptr<io::RandomAccessFile>& file,
      const IpcReadOptions& options = IpcReadOptions::Defaults());

  /// \brief Version of Open that retains ownership of file
  ///
  /// \param[in] file the data source
  /// \param[in] footer_offset the position of the end of the Arrow file
  /// \param[in] options options for IPC reading
  /// \return the returned reader
  static Result<std::shared_ptr<RecordBatchFileReader>> Open(
      const std::shared_ptr<io::RandomAccessFile>& file, int64_t footer_offset,
      const IpcReadOptions& options = IpcReadOptions::Defaults());

  /// \brief Open a file asynchronously (owns the file).
  static Future<std::shared_ptr<RecordBatchFileReader>> OpenAsync(
      const std::shared_ptr<io::RandomAccessFile>& file,
      const IpcReadOptions& options = IpcReadOptions::Defaults());

  /// \brief Open a file asynchronously (borrows the file).
  static Future<std::shared_ptr<RecordBatchFileReader>> OpenAsync(
      io::RandomAccessFile* file,
      const IpcReadOptions& options = IpcReadOptions::Defaults());

  /// \brief Open a file asynchronously (owns the file).
  static Future<std::shared_ptr<RecordBatchFileReader>> OpenAsync(
      const std::shared_ptr<io::RandomAccessFile>& file, int64_t footer_offset,
      const IpcReadOptions& options = IpcReadOptions::Defaults());

  /// \brief Open a file asynchronously (borrows the file).
  static Future<std::shared_ptr<RecordBatchFileReader>> OpenAsync(
      io::RandomAccessFile* file, int64_t footer_offset,
      const IpcReadOptions& options = IpcReadOptions::Defaults());

  /// \brief The schema read from the file
  virtual std::shared_ptr<Schema> schema() const = 0;

  /// \brief Returns the number of record batches in the file
  virtual int num_record_batches() const = 0;

  /// \brief Return the metadata version from the file metadata
  virtual MetadataVersion version() const = 0;

  /// \brief Return the contents of the custom_metadata field from the file's
  /// Footer
  virtual std::shared_ptr<const KeyValueMetadata> metadata() const = 0;

  /// \brief Read a particular record batch from the file. Does not copy memory
  /// if the input source supports zero-copy.
  ///
  /// \param[in] i the index of the record batch to return
  /// \return the read batch
  virtual Result<std::shared_ptr<RecordBatch>> ReadRecordBatch(int i) = 0;

  /// \brief Read a particular record batch along with its custom metadata from the file.
  /// Does not copy memory if the input source supports zero-copy.
  ///
  /// \param[in] i the index of the record batch to return
  /// \return a struct containing the read batch and its custom metadata
  virtual Result<RecordBatchWithMetadata> ReadRecordBatchWithCustomMetadata(int i) = 0;

  /// \brief Return current read statistics
  virtual ReadStats stats() const = 0;

  /// \brief Computes the total number of rows in the file.
  virtual Result<int64_t> CountRows() = 0;

  /// \brief Begin loading metadata for the desired batches into memory.
  ///
  /// This method will also begin loading all dictionaries messages into memory.
  ///
  /// For a regular file this will immediately begin disk I/O in the background on a
  /// thread on the IOContext's thread pool.  If the file is memory mapped this will
  /// ensure the memory needed for the metadata is paged from disk into memory
  ///
  /// \param indices Indices of the batches to prefetch
  ///                If empty then all batches will be prefetched.
  virtual Status PreBufferMetadata(const std::vector<int>& indices) = 0;

  /// \brief Get a reentrant generator of record batches.
  ///
  /// \param[in] coalesce If true, enable I/O coalescing.
  /// \param[in] io_context The IOContext to use (controls which thread pool
  ///     is used for I/O).
  /// \param[in] cache_options Options for coalescing (if enabled).
  /// \param[in] executor Optionally, an executor to use for decoding record
  ///     batches. This is generally only a benefit for very wide and/or
  ///     compressed batches.
  virtual Result<AsyncGenerator<std::shared_ptr<RecordBatch>>> GetRecordBatchGenerator(
      const bool coalesce = false,
      const io::IOContext& io_context = io::default_io_context(),
      const io::CacheOptions cache_options = io::CacheOptions::LazyDefaults(),
      arrow::internal::Executor* executor = NULLPTR) = 0;

  /// \brief Collect all batches as a vector of record batches
  Result<RecordBatchVector> ToRecordBatches();

  /// \brief Collect all batches and concatenate as arrow::Table
  Result<std::shared_ptr<Table>> ToTable();
};

/// \brief A general listener class to receive events.
///
/// You must implement callback methods for interested events.
///
/// This API is EXPERIMENTAL.
///
/// \since 0.17.0
// 在 Apache Arrow 项目中，Listener 类是用于异步流式解析的一个核心接口。
// 它通常与 StreamDecoder（流式解码器）配合使用，采用的是一种典型的回调（Callback）机制。
// Listener 的核心作用是作为事件接收器，处理在解析 Arrow IPC（进程间通信）流过程中产生的各种对象。
// 异步解耦：与传统的 RecordBatchStreamReader（主动拉取数据）不同，Listener 允许你在数据到达并解析完成后，让解码器“通知”你。这在反应式编程或基于事件的网络通信中非常有用。
// 按序处理：当底层的 StreamDecoder 接收到字节流并识别出元数据或数据块时，它会自动调用 Listener 中相应的虚函数。
// 扩展性：用户通过继承 Listener 并重写（Override）感兴趣的方法，可以自定义数据到达后的逻辑（例如：直接存入数据库、进行实时计算或更新 UI）。
class ARROW_EXPORT Listener {
 public:
  virtual ~Listener() = default;

  /// \brief Called when end-of-stream is received.
  ///
  /// The default implementation just returns arrow::Status::OK().
  ///
  /// \return Status
  ///
  /// \see StreamDecoder
  // 当接收到“流结束”（End-of-Stream, EOS）信号时被调用。
  virtual Status OnEOS();

  /// \brief Called when a record batch is decoded and
  /// OnRecordBatchWithMetadataDecoded() isn't overridden.
  ///
  /// The default implementation just returns
  /// arrow::Status::NotImplemented().
  ///
  /// \param[in] record_batch a record batch decoded
  /// \return Status
  ///
  /// \see StreamDecoder
  // 当一个通用的数据批次解码完成时被调用。
  virtual Status OnRecordBatchDecoded(std::shared_ptr<RecordBatch> record_batch);

  /// \brief Called when a record batch with custom metadata is decoded.
  ///
  /// The default implementation just calls OnRecordBatchDecoded()
  /// without custom metadata.
  ///
  /// \param[in] record_batch_with_metadata a record batch with custom
  /// metadata decoded
  /// \return Status
  ///
  /// \see StreamDecoder
  ///
  /// \since 13.0.0
  // 处理带有自定义元数据（Custom Metadata）的数据批次。
  virtual Status OnRecordBatchWithMetadataDecoded(
      RecordBatchWithMetadata record_batch_with_metadata);

  /// \brief Called when a schema is decoded.
  ///
  /// The default implementation just returns arrow::Status::OK().
  ///
  /// \param[in] schema a schema decoded
  /// \return Status
  ///
  /// \see StreamDecoder
  // 当流中的模式信息（表结构）解析完成时被调用。
  virtual Status OnSchemaDecoded(std::shared_ptr<Schema> schema);

  /// \brief Called when a schema is decoded.
  ///
  /// The default implementation just calls OnSchemaDecoded(schema)
  /// (without filtered_schema) to keep backward compatibility.
  ///
  /// \param[in] schema a schema decoded
  /// \param[in] filtered_schema a filtered schema that only has read fields
  /// \return Status
  ///
  /// \see StreamDecoder
  ///
  /// \since 13.0.0
  // 处理包含过滤后的模式的情况。
  virtual Status OnSchemaDecoded(std::shared_ptr<Schema> schema,
                                 std::shared_ptr<Schema> filtered_schema);
};

/// \brief Collect schema and record batches decoded by StreamDecoder.
///
/// This API is EXPERIMENTAL.
///
/// \since 0.17.0
class ARROW_EXPORT CollectListener : public Listener {
 public:
  CollectListener() : schema_(), filtered_schema_(), record_batches_(), metadatas_() {}
  virtual ~CollectListener() = default;

  Status OnSchemaDecoded(std::shared_ptr<Schema> schema,
                         std::shared_ptr<Schema> filtered_schema) override {
    schema_ = std::move(schema);
    filtered_schema_ = std::move(filtered_schema);
    return Status::OK();
  }

  Status OnRecordBatchWithMetadataDecoded(
      RecordBatchWithMetadata record_batch_with_metadata) override {
    record_batches_.push_back(std::move(record_batch_with_metadata.batch));
    metadatas_.push_back(std::move(record_batch_with_metadata.custom_metadata));
    return Status::OK();
  }

  /// \return the decoded schema
  std::shared_ptr<Schema> schema() const { return schema_; }

  /// \return the filtered schema
  std::shared_ptr<Schema> filtered_schema() const { return filtered_schema_; }

  /// \return the all decoded record batches
  const std::vector<std::shared_ptr<RecordBatch>>& record_batches() const {
    return record_batches_;
  }

  /// \return the all decoded metadatas
  const std::vector<std::shared_ptr<KeyValueMetadata>>& metadatas() const {
    return metadatas_;
  }

  /// \return the number of collected record batches
  int64_t num_record_batches() const { return record_batches_.size(); }

  /// \return the last decoded record batch and remove it from
  /// record_batches
  std::shared_ptr<RecordBatch> PopRecordBatch() {
    auto record_batch_with_metadata = PopRecordBatchWithMetadata();
    return std::move(record_batch_with_metadata.batch);
  }

  /// \return the last decoded record batch with custom metadata and
  /// remove it from record_batches
  RecordBatchWithMetadata PopRecordBatchWithMetadata() {
    RecordBatchWithMetadata record_batch_with_metadata;
    if (record_batches_.empty()) {
      return record_batch_with_metadata;
    }
    record_batch_with_metadata.batch = std::move(record_batches_.back());
    record_batch_with_metadata.custom_metadata = std::move(metadatas_.back());
    record_batches_.pop_back();
    metadatas_.pop_back();
    return record_batch_with_metadata;
  }

 private:
  std::shared_ptr<Schema> schema_;
  std::shared_ptr<Schema> filtered_schema_;
  std::vector<std::shared_ptr<RecordBatch>> record_batches_;
  std::vector<std::shared_ptr<KeyValueMetadata>> metadatas_;
};

/// \brief Push style stream decoder that receives data from user.
///
/// This class decodes the Apache Arrow IPC streaming format data.
///
/// This API is EXPERIMENTAL.
///
/// \see https://arrow.apache.org/docs/format/Columnar.html#ipc-streaming-format
///
/// \since 0.17.0
class ARROW_EXPORT StreamDecoder {
 public:
  /// \brief Construct a stream decoder.
  ///
  /// \param[in] listener a Listener that must implement
  /// Listener::OnRecordBatchDecoded() to receive decoded record batches
  /// \param[in] options any IPC reading options (optional)
  StreamDecoder(std::shared_ptr<Listener> listener,
                IpcReadOptions options = IpcReadOptions::Defaults());

  virtual ~StreamDecoder();

  /// \brief Feed data to the decoder as a raw data.
  ///
  /// If the decoder can read one or more record batches by the data,
  /// the decoder calls listener->OnRecordBatchDecoded() with a
  /// decoded record batch multiple times.
  ///
  /// \param[in] data a raw data to be processed. This data isn't
  /// copied. The passed memory must be kept alive through record
  /// batch processing.
  /// \param[in] size raw data size.
  /// \return Status
  Status Consume(const uint8_t* data, int64_t size);

  /// \brief Feed data to the decoder as a Buffer.
  ///
  /// If the decoder can read one or more record batches by the
  /// Buffer, the decoder calls listener->RecordBatchReceived() with a
  /// decoded record batch multiple times.
  ///
  /// \param[in] buffer a Buffer to be processed.
  /// \return Status
  Status Consume(std::shared_ptr<Buffer> buffer);

  /// \brief Reset the internal status.
  ///
  /// You can reuse this decoder for new stream after calling
  /// this.
  ///
  /// \return Status
  Status Reset();

  /// \return the shared schema of the record batches in the stream
  std::shared_ptr<Schema> schema() const;

  /// \brief Return the number of bytes needed to advance the state of
  /// the decoder.
  ///
  /// This method is provided for users who want to optimize performance.
  /// Normal users don't need to use this method.
  ///
  /// Here is an example usage for normal users:
  ///
  /// ~~~{.cpp}
  /// decoder.Consume(buffer1);
  /// decoder.Consume(buffer2);
  /// decoder.Consume(buffer3);
  /// ~~~
  ///
  /// Decoder has internal buffer. If consumed data isn't enough to
  /// advance the state of the decoder, consumed data is buffered to
  /// the internal buffer. It causes performance overhead.
  ///
  /// If you pass next_required_size() size data to each Consume()
  /// call, the decoder doesn't use its internal buffer. It improves
  /// performance.
  ///
  /// Here is an example usage to avoid using internal buffer:
  ///
  /// ~~~{.cpp}
  /// buffer1 = get_data(decoder.next_required_size());
  /// decoder.Consume(buffer1);
  /// buffer2 = get_data(decoder.next_required_size());
  /// decoder.Consume(buffer2);
  /// ~~~
  ///
  /// Users can use this method to avoid creating small chunks. Record
  /// batch data must be contiguous data. If users pass small chunks
  /// to the decoder, the decoder needs concatenate small chunks
  /// internally. It causes performance overhead.
  ///
  /// Here is an example usage to reduce small chunks:
  ///
  /// ~~~{.cpp}
  /// buffer = AllocateResizableBuffer();
  /// while ((small_chunk = get_data(&small_chunk_size))) {
  ///   auto current_buffer_size = buffer->size();
  ///   buffer->Resize(current_buffer_size + small_chunk_size);
  ///   memcpy(buffer->mutable_data() + current_buffer_size,
  ///          small_chunk,
  ///          small_chunk_size);
  ///   if (buffer->size() < decoder.next_required_size()) {
  ///     continue;
  ///   }
  ///   std::shared_ptr<arrow::Buffer> chunk(buffer.release());
  ///   decoder.Consume(chunk);
  ///   buffer = AllocateResizableBuffer();
  /// }
  /// if (buffer->size() > 0) {
  ///   std::shared_ptr<arrow::Buffer> chunk(buffer.release());
  ///   decoder.Consume(chunk);
  /// }
  /// ~~~
  ///
  /// \return the number of bytes needed to advance the state of the
  /// decoder
  int64_t next_required_size() const;

  /// \brief Return current read statistics
  ReadStats stats() const;

 private:
  class StreamDecoderImpl;
  std::unique_ptr<StreamDecoderImpl> impl_;

  ARROW_DISALLOW_COPY_AND_ASSIGN(StreamDecoder);
};

// Generic read functions; does not copy data if the input supports zero copy reads

/// \brief Read Schema from stream serialized as a single IPC message
/// and populate any dictionary-encoded fields into a DictionaryMemo
///
/// \param[in] stream an InputStream
/// \param[in] dictionary_memo for recording dictionary-encoded fields
/// \return the output Schema
///
/// If record batches follow the schema, it is better to use
/// RecordBatchStreamReader
ARROW_EXPORT
Result<std::shared_ptr<Schema>> ReadSchema(io::InputStream* stream,
                                           DictionaryMemo* dictionary_memo);

/// \brief Read Schema from encapsulated Message
///
/// \param[in] message the message containing the Schema IPC metadata
/// \param[in] dictionary_memo DictionaryMemo for recording dictionary-encoded
/// fields. Can be nullptr if you are sure there are no
/// dictionary-encoded fields
/// \return the resulting Schema
ARROW_EXPORT
Result<std::shared_ptr<Schema>> ReadSchema(const Message& message,
                                           DictionaryMemo* dictionary_memo);

/// Read record batch as encapsulated IPC message with metadata size prefix and
/// header
///
/// \param[in] schema the record batch schema
/// \param[in] dictionary_memo DictionaryMemo which has any
/// dictionaries. Can be nullptr if you are sure there are no
/// dictionary-encoded fields
/// \param[in] options IPC options for reading
/// \param[in] stream the file where the batch is located
/// \return the read record batch
ARROW_EXPORT
Result<std::shared_ptr<RecordBatch>> ReadRecordBatch(
    const std::shared_ptr<Schema>& schema, const DictionaryMemo* dictionary_memo,
    const IpcReadOptions& options, io::InputStream* stream);

/// \brief Read record batch from message
///
/// \param[in] message a Message containing the record batch metadata
/// \param[in] schema the record batch schema
/// \param[in] dictionary_memo DictionaryMemo which has any
/// dictionaries. Can be nullptr if you are sure there are no
/// dictionary-encoded fields
/// \param[in] options IPC options for reading
/// \return the read record batch
ARROW_EXPORT
Result<std::shared_ptr<RecordBatch>> ReadRecordBatch(
    const Message& message, const std::shared_ptr<Schema>& schema,
    const DictionaryMemo* dictionary_memo, const IpcReadOptions& options);

/// Read record batch from file given metadata and schema
///
/// \param[in] metadata a Message containing the record batch metadata
/// \param[in] schema the record batch schema
/// \param[in] dictionary_memo DictionaryMemo which has any
/// dictionaries. Can be nullptr if you are sure there are no
/// dictionary-encoded fields
/// \param[in] file a random access file
/// \param[in] options options for deserialization
/// \return the read record batch
ARROW_EXPORT
Result<std::shared_ptr<RecordBatch>> ReadRecordBatch(
    const Buffer& metadata, const std::shared_ptr<Schema>& schema,
    const DictionaryMemo* dictionary_memo, const IpcReadOptions& options,
    io::RandomAccessFile* file);

/// \brief Read arrow::Tensor as encapsulated IPC message in file
///
/// \param[in] file an InputStream pointed at the start of the message
/// \return the read tensor
ARROW_EXPORT
Result<std::shared_ptr<Tensor>> ReadTensor(io::InputStream* file);

/// \brief EXPERIMENTAL: Read arrow::Tensor from IPC message
///
/// \param[in] message a Message containing the tensor metadata and body
/// \return the read tensor
ARROW_EXPORT
Result<std::shared_ptr<Tensor>> ReadTensor(const Message& message);

/// \brief EXPERIMENTAL: Read arrow::SparseTensor as encapsulated IPC message in file
///
/// \param[in] file an InputStream pointed at the start of the message
/// \return the read sparse tensor
ARROW_EXPORT
Result<std::shared_ptr<SparseTensor>> ReadSparseTensor(io::InputStream* file);

/// \brief EXPERIMENTAL: Read arrow::SparseTensor from IPC message
///
/// \param[in] message a Message containing the tensor metadata and body
/// \return the read sparse tensor
ARROW_EXPORT
Result<std::shared_ptr<SparseTensor>> ReadSparseTensor(const Message& message);

namespace internal {

// These internal APIs may change without warning or deprecation

/// \brief EXPERIMENTAL: Read arrow::SparseTensorFormat::type from a metadata
/// \param[in] metadata a Buffer containing the sparse tensor metadata
/// \return the count of the body buffers
ARROW_EXPORT
Result<size_t> ReadSparseTensorBodyBufferCount(const Buffer& metadata);

/// \brief EXPERIMENTAL: Read arrow::SparseTensor from an IpcPayload
/// \param[in] payload a IpcPayload contains a serialized SparseTensor
/// \return the read sparse tensor
ARROW_EXPORT
Result<std::shared_ptr<SparseTensor>> ReadSparseTensorPayload(const IpcPayload& payload);

// For fuzzing targets
ARROW_EXPORT
Status FuzzIpcStream(const uint8_t* data, int64_t size);
ARROW_EXPORT
Status FuzzIpcTensorStream(const uint8_t* data, int64_t size);
ARROW_EXPORT
Status FuzzIpcFile(const uint8_t* data, int64_t size);

}  // namespace internal

}  // namespace ipc
}  // namespace arrow
