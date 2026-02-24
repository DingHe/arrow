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
#include <vector>

#include "arrow/compare.h"
#include "arrow/device.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/type_fwd.h"
#include "arrow/util/iterator.h"
#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"

namespace arrow {

/// \class RecordBatch
/// \brief Collection of equal-length arrays matching a particular Schema
///
/// A record batch is table-like data structure that is semantically a sequence
/// of fields, each a contiguous Arrow array
// 在 Apache Arrow 项目中，RecordBatch（数据记录批次）是一个极其核心的类。它是 Arrow 实现列式存储和高性能计算的基本单位。
// RecordBatch 是一个二维表结构的内存抽象。你可以把它想象成数据库中的一个“数据页”或 Pandas 中的一个 DataFrame。
// 列式组织：它由一系列等长的 Array（列）组成。
// 模式绑定：每个 RecordBatch 都绑定了一个 Schema，定义了每一列的名字、类型和元数据。
// 不可变性：在 Arrow 中，RecordBatch 一旦创建，其内容（数据）通常是不可变的。
// 传输单元：它是 Arrow IPC（进程间通信）和 Flight 协议中传输的基本数据包。
class ARROW_EXPORT RecordBatch {
 public:
  virtual ~RecordBatch() = default;

  /// \param[in] schema The record batch schema
  /// \param[in] num_rows length of fields in the record batch. Each array
  /// should have the same length as num_rows
  /// \param[in] columns the record batch fields as vector of arrays
  /// \param[in] sync_event optional synchronization event for non-CPU device
  /// memory used by buffers
  // 接受 Array 对象的 vector。适用于已经构建好的列。
  static std::shared_ptr<RecordBatch> Make(
      std::shared_ptr<Schema> schema, int64_t num_rows,
      std::vector<std::shared_ptr<Array>> columns,
      std::shared_ptr<Device::SyncEvent> sync_event = NULLPTR);

  /// \brief Construct record batch from vector of internal data structures
  /// \since 0.5.0
  ///
  /// This class is intended for internal use, or advanced users.
  ///
  /// \param schema the record batch schema
  /// \param num_rows the number of semantic rows in the record batch. This
  /// should be equal to the length of each field
  /// \param columns the data for the batch's columns
  /// \param device_type the type of the device that the Arrow columns are
  /// allocated on
  /// \param sync_event optional synchronization event for non-CPU device
  /// memory used by buffers
  // 接受 ArrayData 对象的 vector（更底层）。支持指定 DeviceAllocationType（如 CPU 或 GPU）和同步事件（SyncEvent）
  static std::shared_ptr<RecordBatch> Make(
      std::shared_ptr<Schema> schema, int64_t num_rows,
      std::vector<std::shared_ptr<ArrayData>> columns,
      DeviceAllocationType device_type = DeviceAllocationType::kCPU,
      std::shared_ptr<Device::SyncEvent> sync_event = NULLPTR);

  /// \brief Create an empty RecordBatch of a given schema
  ///
  /// The output RecordBatch will be created with DataTypes from
  /// the given schema.
  ///
  /// \param[in] schema the schema of the empty RecordBatch
  /// \param[in] pool the memory pool to allocate memory from
  /// \return the resulting RecordBatch
  // 根据给定的 Schema 创建一个行数为 0 的空批次，常用于初始化流或测试。
  static Result<std::shared_ptr<RecordBatch>> MakeEmpty(
      std::shared_ptr<Schema> schema, MemoryPool* pool = default_memory_pool());

  /// \brief Convert record batch to struct array
  ///
  /// Create a struct array whose child arrays are the record batch's columns.
  /// Note that the record batch's top-level field metadata cannot be reflected
  /// in the resulting struct array.
  // 将整个批次转换为一个单一的 StructArray。
  Result<std::shared_ptr<StructArray>> ToStructArray() const;

  /// \brief Convert record batch with one data type to Tensor
  ///
  /// Create a Tensor object with shape (number of rows, number of columns) and
  /// strides (type size in bytes, type size in bytes * number of rows).
  /// Generated Tensor will have column-major layout.
  ///
  /// \param[in] null_to_nan if true, convert nulls to NaN
  /// \param[in] row_major if true, create row-major Tensor else column-major Tensor
  /// \param[in] pool the memory pool to allocate the tensor buffer
  /// \return the resulting Tensor
  // 如果批次中所有列的类型相同，可以将其转换为一个 Tensor（张量），支持行优先或列优先布局。
  Result<std::shared_ptr<Tensor>> ToTensor(
      bool null_to_nan = false, bool row_major = true,
      MemoryPool* pool = default_memory_pool()) const;

  /// \brief Construct record batch from struct array
  ///
  /// This constructs a record batch using the child arrays of the given
  /// array, which must be a struct array.
  ///
  /// \param[in] array the source array, must be a StructArray
  /// \param[in] pool the memory pool to allocate new validity bitmaps
  ///
  /// This operation will usually be zero-copy.  However, if the struct array has an
  /// offset or a validity bitmap then these will need to be pushed into the child arrays.
  /// Pushing the offset is zero-copy but pushing the validity bitmap is not.
  // 将一个 StructArray（结构体数组）解包成一个 RecordBatch。这通常是零拷贝操作，除非存在复杂的位图合并。
  static Result<std::shared_ptr<RecordBatch>> FromStructArray(
      const std::shared_ptr<Array>& array, MemoryPool* pool = default_memory_pool());

  /// \brief Determine if two record batches are equal
  ///
  /// \param[in] other the RecordBatch to compare with
  /// \param[in] check_metadata if true, the schema metadata will be compared,
  ///            regardless of the value set in \ref EqualOptions::use_metadata
  /// \param[in] opts the options for equality comparisons
  /// \return true if batches are equal
  bool Equals(const RecordBatch& other, bool check_metadata = false,
              const EqualOptions& opts = EqualOptions::Defaults()) const;

  /// \brief Determine if two record batches are equal
  ///
  /// \param[in] other the RecordBatch to compare with
  /// \param[in] opts the options for equality comparisons
  /// \return true if batches are equal
  bool Equals(const RecordBatch& other, const EqualOptions& opts) const;

  /// \brief Determine if two record batches are approximately equal
  ///
  /// \param[in] other the RecordBatch to compare with
  /// \param[in] opts the options for equality comparisons
  /// \return true if batches are approximately equal
  bool ApproxEquals(const RecordBatch& other,
                    const EqualOptions& opts = EqualOptions::Defaults()) const {
    return Equals(other, opts.use_schema(false).use_atol(true));
  }

  /// \return the record batch's schema
  // 获取当前的 Schema。
  const std::shared_ptr<Schema>& schema() const { return schema_; }

  /// \brief Replace the schema with another schema with the same types, but potentially
  /// different field names and/or metadata.
  // 替换 Schema（必须保证物理类型兼容，仅修改字段名或元数据）
  Result<std::shared_ptr<RecordBatch>> ReplaceSchema(
      std::shared_ptr<Schema> schema) const;

  /// \brief Retrieve all columns at once
  virtual const std::vector<std::shared_ptr<Array>>& columns() const = 0;

  /// \brief Retrieve an array from the record batch
  /// \param[in] i field index, does not boundscheck
  /// \return an Array object
  // 获取第 $i$ 列的 Array 对象。
  virtual std::shared_ptr<Array> column(int i) const = 0;

  /// \brief Retrieve an array from the record batch
  /// \param[in] name field name
  /// \return an Array or null if no field was found
  // 通过列名查找列。
  std::shared_ptr<Array> GetColumnByName(const std::string& name) const;

  /// \brief Retrieve an array's internal data from the record batch
  /// \param[in] i field index, does not boundscheck
  /// \return an internal ArrayData object
  // 获取第 $i$ 列更底层的 ArrayData（用于内核计算）。
  virtual std::shared_ptr<ArrayData> column_data(int i) const = 0;

  /// \brief Retrieve all arrays' internal data from the record batch.
  virtual const ArrayDataVector& column_data() const = 0;

  /// \brief Add column to the record batch, producing a new RecordBatch
  ///
  /// \param[in] i field index, which will be boundschecked
  /// \param[in] field field to be added
  /// \param[in] column column to be added
  virtual Result<std::shared_ptr<RecordBatch>> AddColumn(
      int i, const std::shared_ptr<Field>& field,
      const std::shared_ptr<Array>& column) const = 0;

  /// \brief Add new nullable column to the record batch, producing a new
  /// RecordBatch.
  ///
  /// For non-nullable columns, use the Field-based version of this method.
  ///
  /// \param[in] i field index, which will be boundschecked
  /// \param[in] field_name name of field to be added
  /// \param[in] column column to be added
  // 在指定位置插入一列。
  virtual Result<std::shared_ptr<RecordBatch>> AddColumn(
      int i, std::string field_name, const std::shared_ptr<Array>& column) const;

  /// \brief Replace a column in the record batch, producing a new RecordBatch
  ///
  /// \param[in] i field index, does boundscheck
  /// \param[in] field field to be replaced
  /// \param[in] column column to be replaced
  // 替换指定位置的列。
  virtual Result<std::shared_ptr<RecordBatch>> SetColumn(
      int i, const std::shared_ptr<Field>& field,
      const std::shared_ptr<Array>& column) const = 0;

  /// \brief Remove column from the record batch, producing a new RecordBatch
  ///
  /// \param[in] i field index, does boundscheck
  // 删除指定位置的列。
  virtual Result<std::shared_ptr<RecordBatch>> RemoveColumn(int i) const = 0;

  virtual std::shared_ptr<RecordBatch> ReplaceSchemaMetadata(
      const std::shared_ptr<const KeyValueMetadata>& metadata) const = 0;

  /// \brief Name in i-th column
  // 获取第 $i$ 列的名字。
  const std::string& column_name(int i) const;

  /// \return the number of columns in the table
  // 获取总列数。
  int num_columns() const;

  /// \return the number of rows (the corresponding length of each column)
  // 获取总行数。
  int64_t num_rows() const { return num_rows_; }

  /// \brief Copy the entire RecordBatch to destination MemoryManager
  ///
  /// This uses Array::CopyTo on each column of the record batch to create
  /// a new record batch where all underlying buffers for the columns have
  /// been copied to the destination MemoryManager. This uses
  /// MemoryManager::CopyBuffer under the hood.
  // 将整个批次的数据拷贝到目标设备（例如从 CPU 拷贝到 GPU 显存）。
  Result<std::shared_ptr<RecordBatch>> CopyTo(
      const std::shared_ptr<MemoryManager>& to) const;

  /// \brief View or Copy the entire RecordBatch to destination MemoryManager
  ///
  /// This uses Array::ViewOrCopyTo on each column of the record batch to create
  /// a new record batch where all underlying buffers for the columns have
  /// been zero-copy viewed on the destination MemoryManager, falling back
  /// to performing a copy if it can't be viewed as a zero-copy buffer. This uses
  /// Buffer::ViewOrCopy under the hood.
  // 优先尝试零拷贝视图，如果不跨物理设备则直接引用，否则执行拷贝。
  Result<std::shared_ptr<RecordBatch>> ViewOrCopyTo(
      const std::shared_ptr<MemoryManager>& to) const;

  /// \brief Slice each of the arrays in the record batch
  /// \param[in] offset the starting offset to slice, through end of batch
  /// \return new record batch
  // 极其重要。
  // 创建一个子视图，不发生数据拷贝，只是修改偏移量和长度。这是实现高性能分页和过滤的关键。
  virtual std::shared_ptr<RecordBatch> Slice(int64_t offset) const;

  /// \brief Slice each of the arrays in the record batch
  /// \param[in] offset the starting offset to slice
  /// \param[in] length the number of elements to slice from offset
  /// \return new record batch
  // 极其重要。
  // 创建一个子视图，不发生数据拷贝，只是修改偏移量和长度。这是实现高性能分页和过滤的关键。
  virtual std::shared_ptr<RecordBatch> Slice(int64_t offset, int64_t length) const = 0;

  /// \return PrettyPrint representation suitable for debugging
  // 返回批次的文本表示，方便调试。
  std::string ToString() const;

  /// \brief Return names of all columns
  std::vector<std::string> ColumnNames() const;

  /// \brief Rename columns with provided names
  // 批量修改列名。
  Result<std::shared_ptr<RecordBatch>> RenameColumns(
      const std::vector<std::string>& names) const;

  /// \brief Return new record batch with specified columns
  // 根据索引集合投影（Project）出若干列。
  Result<std::shared_ptr<RecordBatch>> SelectColumns(
      const std::vector<int>& indices) const;

  /// \brief Perform cheap validation checks to determine obvious inconsistencies
  /// within the record batch's schema and internal data.
  ///
  /// This is O(k) where k is the total number of fields and array descendents.
  ///
  /// \return Status
  // 快速检查。确认各列长度是否一致，是否符合 Schema。
  virtual Status Validate() const;

  /// \brief Perform extensive validation checks to determine inconsistencies
  /// within the record batch's schema and internal data.
  ///
  /// This is potentially O(k*n) where n is the number of rows.
  ///
  /// \return Status
  // 深度检查。会遍历数据以确保位图、偏移量等内部逻辑完全正确（$O(N)$ 复杂度）。
  virtual Status ValidateFull() const;

  /// \brief EXPERIMENTAL: Return a top-level sync event object for this record batch
  ///
  /// If all of the data for this record batch is in CPU memory, then this
  /// will return null. If the data for this batch is
  /// on a device, then if synchronization is needed before accessing the
  /// data the returned sync event will allow for it.
  ///
  /// \return null or a Device::SyncEvent
  // 实验性功能。用于异步计算，返回用于等待非 CPU 设备数据准备就绪的同步信号。
  virtual const std::shared_ptr<Device::SyncEvent>& GetSyncEvent() const = 0;

  virtual DeviceAllocationType device_type() const = 0;

  /// \brief Create a statistics array of this record batch
  ///
  /// The created array follows the C data interface statistics
  /// specification. See
  /// https://arrow.apache.org/docs/format/StatisticsSchema.html
  /// for details.
  ///
  /// \param[in] pool the memory pool to allocate memory from
  /// \return the statistics array of this record batch
  // 为此批次生成符合 Arrow 规范的统计信息数组（如最大值、最小值、空值计数等）。
  Result<std::shared_ptr<Array>> MakeStatisticsArray(
      MemoryPool* pool = default_memory_pool()) const;

 protected:
  RecordBatch(std::shared_ptr<Schema> schema, int64_t num_rows);
  // 记录批次的“表头”，描述了数据的逻辑结构。
  std::shared_ptr<Schema> schema_;
  // 记录批次的行数。由于是等长数组，所有列的长度必须等于此值。
  int64_t num_rows_;

 private:
  ARROW_DISALLOW_COPY_AND_ASSIGN(RecordBatch);
};

struct ARROW_EXPORT RecordBatchWithMetadata {
  std::shared_ptr<RecordBatch> batch;
  std::shared_ptr<KeyValueMetadata> custom_metadata;
};

template <>
struct IterationTraits<RecordBatchWithMetadata> {
  static RecordBatchWithMetadata End() { return {NULLPTR, NULLPTR}; }
  static bool IsEnd(const RecordBatchWithMetadata& val) { return val.batch == NULLPTR; }
};

/// \brief Abstract interface for reading stream of record batches
// RecordBatchReader 是一个至关重要的流式处理接口。它定义了如何以迭代的方式读取一系列具有相同结构的 RecordBatch。
// RecordBatchReader 的核心目标是解决内存限制问题。
// 在处理海量数据（如数 GB 甚至数 TB 的文件）时，将所有数据一次性加载到内存中的 Table 对象是不现实的。RecordBatchReader 提供了一种生产者-消费者模型：
// 延迟加载：数据只有在调用读取方法时才会从磁盘、网络或计算任务中拉取。
// 流式传输：它是 Arrow IPC（进程间通信）和 Arrow Flight 协议传输数据的基本形式。
// 接口抽象：它是一个抽象基类。不同的子类可以实现从 CSV 文件、Parquet 文件、Socket 流或扫描数据库结果集中读取数据。
class ARROW_EXPORT RecordBatchReader {
 public:
  // 定义了读取器产出的数据类型，即 std::shared_ptr<RecordBatch>
  using ValueType = std::shared_ptr<RecordBatch>;

  virtual ~RecordBatchReader();

  /// \return the shared schema of the record batches in the stream
  // 返回流中所有记录批次共有的逻辑结构（Schema）
  // 重要性：消费者在读取实际数据前，通常需要通过 Schema 了解字段名和数据类型。
  virtual std::shared_ptr<Schema> schema() const = 0;

  /// \brief Read the next record batch in the stream. Return null for batch
  /// when reaching end of stream
  ///
  /// Example:
  ///
  /// ```
  /// while (true) {
  ///   std::shared_ptr<RecordBatch> batch;
  ///   ARROW_RETURN_NOT_OK(reader->ReadNext(&batch));
  ///   if (!batch) {
  ///     break;
  ///   }
  ///   // handling the `batch`, the `batch->num_rows()`
  ///   // might be 0.
  /// }
  /// ```
  ///
  /// \param[out] batch the next loaded batch, null at end of stream. Returning
  /// an empty batch doesn't mean the end of stream because it is valid data.
  /// \return Status
  // 流式读取的核心。
  // 尝试读取下一个批次。
  // 通过输出参数返回 batch。如果到达流末尾，返回 null。
  virtual Status ReadNext(std::shared_ptr<RecordBatch>* batch) = 0;
  // 带元数据（Metadata）的读取版本。默认返回“未实现”，允许特定格式提供批次级别的额外信息。
  virtual Result<RecordBatchWithMetadata> ReadNext() {
    return Status::NotImplemented("ReadNext with custom metadata");
  }

  /// \brief Iterator interface
  // 更符合 C++ 习惯的快捷方法，返回 Result<shared_ptr<RecordBatch>>。
  Result<std::shared_ptr<RecordBatch>> Next() {
    std::shared_ptr<RecordBatch> batch;
    ARROW_RETURN_NOT_OK(ReadNext(&batch));
    return batch;
  }

  /// \brief finalize reader
  // 显式终止读取器。用于提前释放底层资源（如网络连接）。
  virtual Status Close() { return Status::OK(); }

  /// \brief EXPERIMENTAL: Get the device type for record batches this reader produces
  ///
  /// default implementation is to return DeviceAllocationType::kCPU
  // 指示该读取器产生的批次位于什么设备上（如 CPU 或 GPU）。默认返回 kCPU。
  virtual DeviceAllocationType device_type() const { return DeviceAllocationType::kCPU; }
  // 实现了 C++ 标准库风格的 Input Iterator
  // 允许使用 for (auto batch : *reader) 这样的语法。
  class RecordBatchReaderIterator {
   public:
    using iterator_category = std::input_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = std::shared_ptr<RecordBatch>;
    using pointer = const value_type*;
    using reference = const value_type&;

    RecordBatchReaderIterator() : batch_(RecordBatchEnd()), reader_(NULLPTR) {}

    explicit RecordBatchReaderIterator(RecordBatchReader* reader)
        : batch_(RecordBatchEnd()), reader_(reader) {
      Next();
    }

    bool operator==(const RecordBatchReaderIterator& other) const {
      return batch_ == other.batch_;
    }

    bool operator!=(const RecordBatchReaderIterator& other) const {
      return !(*this == other);
    }

    Result<std::shared_ptr<RecordBatch>> operator*() {
      ARROW_RETURN_NOT_OK(batch_);

      return batch_;
    }

    RecordBatchReaderIterator& operator++() {
      Next();
      return *this;
    }

    RecordBatchReaderIterator operator++(int) {
      RecordBatchReaderIterator tmp(*this);
      Next();
      return tmp;
    }

   private:
    std::shared_ptr<RecordBatch> RecordBatchEnd() {
      return std::shared_ptr<RecordBatch>(NULLPTR);
    }

    void Next() {
      if (reader_ == NULLPTR) {
        batch_ = RecordBatchEnd();
        return;
      }
      batch_ = reader_->Next();
    }

    Result<std::shared_ptr<RecordBatch>> batch_;
    RecordBatchReader* reader_;
  };
  /// \brief Return an iterator to the first record batch in the stream
  RecordBatchReaderIterator begin() { return RecordBatchReaderIterator(this); }

  /// \brief Return an iterator to the end of the stream
  RecordBatchReaderIterator end() { return RecordBatchReaderIterator(); }

  /// \brief Consume entire stream as a vector of record batches
  // 将流中的所有批次读取到一个 vector 中
  Result<RecordBatchVector> ToRecordBatches();

  /// \brief Read all batches and concatenate as arrow::Table
  // 读取所有批次并将它们合并（Concatenate）成一个单一的 arrow::Table 对象。
  Result<std::shared_ptr<Table>> ToTable();

  /// \brief Create a RecordBatchReader from a vector of RecordBatch.
  ///
  /// \param[in] batches the vector of RecordBatch to read from
  /// \param[in] schema schema to conform to. Will be inferred from the first
  ///            element if not provided.
  /// \param[in] device_type the type of device that the batches are allocated on
  // 将内存中已有的 vector<RecordBatch> 包装成一个读取器，方便统一接口。
  static Result<std::shared_ptr<RecordBatchReader>> Make(
      RecordBatchVector batches, std::shared_ptr<Schema> schema = NULLPTR,
      DeviceAllocationType device_type = DeviceAllocationType::kCPU);

  /// \brief Create a RecordBatchReader from an Iterator of RecordBatch.
  ///
  /// \param[in] batches an iterator of RecordBatch to read from.
  /// \param[in] schema schema that each record batch in iterator will conform to.
  /// \param[in] device_type the type of device that the batches are allocated on
  // 将一个通用的迭代器包装成读取器，常用于将复杂的计算算子链包装成流
  static Result<std::shared_ptr<RecordBatchReader>> MakeFromIterator(
      Iterator<std::shared_ptr<RecordBatch>> batches, std::shared_ptr<Schema> schema,
      DeviceAllocationType device_type = DeviceAllocationType::kCPU);
};

/// \brief Concatenate record batches
///
/// The columns of the new batch are formed by concatenate the same columns of each input
/// batch. Concatenate multiple batches into a new batch requires that the schema must be
/// consistent. It supports merging batches without columns (only length, scenarios such
/// as count(*)).
///
/// \param[in] batches a vector of record batches to be concatenated
/// \param[in] pool memory to store the result will be allocated from this memory pool
/// \return the concatenated record batch
// 将一个 vector 中的多个 RecordBatch 物理合并为一个大的 RecordBatch。
// 它会把每一列对应的多个 Array 拼接在一起。这涉及到内存分配，因此需要传入 MemoryPool。
ARROW_EXPORT
Result<std::shared_ptr<RecordBatch>> ConcatenateRecordBatches(
    const RecordBatchVector& batches, MemoryPool* pool = default_memory_pool());

}  // namespace arrow
