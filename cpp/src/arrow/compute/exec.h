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

// NOTE: API is EXPERIMENTAL and will change without going through a
// deprecation cycle

#pragma once

#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "arrow/array/data.h"
#include "arrow/compute/expression.h"
#include "arrow/compute/type_fwd.h"
#include "arrow/datum.h"
#include "arrow/result.h"
#include "arrow/type_fwd.h"
#include "arrow/util/macros.h"
#include "arrow/util/type_fwd.h"
#include "arrow/util/visibility.h"

namespace arrow {
namespace compute {

// It seems like 64K might be a good default chunksize to use for execution
// based on the experience of other query processing systems. The current
// default is not to chunk contiguous arrays, though, but this may change in
// the future once parallel execution is implemented
static constexpr int64_t kDefaultExecChunksize = UINT16_MAX;

/// \brief Context for expression-global variables and options used by
/// function evaluation
// 在 Apache Arrow 的计算层中，ExecContext 是一个全局资源配置管理器。如果把 KernelContext（内核上下文）比作一个工人的工作台，那么 ExecContext 就是整个工厂的车间配置。
// ExecContext 的主要作用是统一管理执行环境的共享资源和行为策略：
// 资源提供：为计算内核提供物理资源，包括内存分配器（MemoryPool）和计算算力（Executor）。
// 规则定义：决定大型数据如何分块执行（exec_chunksize），以及是否进行预分配优化（preallocate_contiguous）。
// 服务发现：通过关联的 FunctionRegistry，让计算流程知道去哪里查找可用的算子。
// 跨内核共享：它通常在一次复杂的查询规划（ExecPlan）中被创建一次，并被传递给该规划下的所有算子使用，确保行为一致。
class ARROW_EXPORT ExecContext {
 public:
  // If no function registry passed, the default is used.
  // pool: 默认为全局默认内存池。
  // executor: 可选的线程池（Executor）。如果为空，通常在单线程下运行。
  // func_registry: 函数注册表。如果为空，自动关联全局注册表（GetFunctionRegistry()）。
  explicit ExecContext(MemoryPool* pool = default_memory_pool(),
                       ::arrow::internal::Executor* executor = NULLPTR,
                       FunctionRegistry* func_registry = NULLPTR);

  /// \brief The MemoryPool used for allocations, default is
  /// default_memory_pool().
  // 获取内存池指针。
  MemoryPool* memory_pool() const { return pool_; }
  // 获取当前运行环境的 CPU 信息。
  // 用于检测硬件特性（如是否支持 AVX-512），配合我们之前提到的 SimdLevel 来选择最优内核。
  const ::arrow::internal::CpuInfo* cpu_info() const;

  /// \brief An Executor which may be used to parallelize execution.
  // 获取执行器（Executor/线程池）指针。
  ::arrow::internal::Executor* executor() const { return executor_; }

  /// \brief The FunctionRegistry for looking up functions by name and
  /// selecting kernels for execution. Defaults to the library-global function
  /// registry provided by GetFunctionRegistry.
  // 获取当前的函数查找目录。
  FunctionRegistry* func_registry() const { return func_registry_; }

  // \brief Set maximum length unit of work for kernel execution. Larger
  // contiguous array inputs will be split into smaller chunks, and, if
  // possible and enabled, processed in parallel. The default chunksize is
  // INT64_MAX, so contiguous arrays are not split.
  // 设置/获取执行时的分块大小。
  // 这是性能调优的关键。如果输入是一个包含 1 亿行数据的巨大连续数组，Arrow 会根据这个值将其切分为多个小 Batch。
  void set_exec_chunksize(int64_t chunksize) { exec_chunksize_ = chunksize; }

  // \brief Maximum length for ExecBatch data chunks processed by
  // kernels. Contiguous array inputs with longer length will be split into
  // smaller chunks.
  int64_t exec_chunksize() const { return exec_chunksize_; }

  /// \brief Set whether to use multiple threads for function execution. This
  /// is not yet used.
  // 控制是否启用多线程加速。
  void set_use_threads(bool use_threads = true) { use_threads_ = use_threads; }

  /// \brief If true, then utilize multiple threads where relevant for function
  /// execution. This is not yet used.
  bool use_threads() const { return use_threads_; }

  // Set the preallocation strategy for kernel execution as it relates to
  // chunked execution. For chunked execution, whether via ChunkedArray inputs
  // or splitting larger Array arguments into smaller pieces, contiguous
  // allocation (if permitted by the kernel) will allocate one large array to
  // write output into yielding it to the caller at the end. If this option is
  // set to off, then preallocations will be performed independently for each
  // chunk of execution
  //
  // TODO: At some point we might want the limit the size of contiguous
  // preallocations. For example, even if the exec_chunksize is 64K or less, we
  // might limit contiguous allocations to 1M records, say.
  // 设置连续内存预分配策略。
  // 如果为 true：框架会先一次性分配 100 万行的连续大内存，每个分块任务写到这个大内存的切片中。这种方式产出的结果是连续的 Array。
  void set_preallocate_contiguous(bool preallocate) {
    preallocate_contiguous_ = preallocate;
  }

  /// \brief If contiguous preallocations should be used when doing chunked
  /// execution as specified by exec_chunksize(). See
  /// set_preallocate_contiguous() for more information.
  bool preallocate_contiguous() const { return preallocate_contiguous_; }

 private:
  MemoryPool* pool_;
  ::arrow::internal::Executor* executor_;
  FunctionRegistry* func_registry_;
  int64_t exec_chunksize_ = std::numeric_limits<int64_t>::max();
  bool preallocate_contiguous_ = true;
  bool use_threads_ = true;
};

// TODO: Consider standardizing on uint16 selection vectors and only use them
// when we can ensure that each value is 64K length or smaller

/// \brief Container for an array of value selection indices that were
/// materialized from a filter.
///
/// Columnar query engines (see e.g. [1]) have found that rather than
/// materializing filtered data, the filter can instead be converted to an
/// array of the "on" indices and then "fusing" these indices in operator
/// implementations. This is especially relevant for aggregations but also
/// applies to scalar operations.
///
/// We are not yet using this so this is mostly a placeholder for now.
///
/// [1]: http://cidrdb.org/cidr2005/papers/P19.pdf
// 在 Apache Arrow 的计算引擎中，SelectionVector 是一个用于延迟物化（Lazy Materialization）和性能优化的机制。
// 虽然代码注释中提到它在某些版本中仍处于“占位（Placeholder）”或初步开发阶段，但它代表了向量化执行引擎中一种非常成熟的优化策略。
// SelectionVector 的核心作用是存储过滤结果的索引，而不是过滤后的数据本身。
// 传统做法 vs 索引过滤
// 传统做法（物化过滤）：如果你有一个包含 100 万行的数组，过滤后剩下 100 行，引擎会分配新的内存并把这 100 行复制过去。
// 索引过滤（Selection Vector）：引擎不复制数据，而是创建一个小的整数数组，记录下这 100 行在原数组中的位置（索引）。后续的计算算子（如加法、聚合）直接根据这些索引去原数组中读取数据。
// 减少内存拷贝：对于大数据块，拷贝成本极高。
// 算子融合（Operator Fusion）：可以将“过滤”与“聚合”融合在一起，只在最后一步读取数据。
// 缓存友好：索引数组通常很小，可以放入 CPU 缓存。
class ARROW_EXPORT SelectionVector {
 public:
  explicit SelectionVector(std::shared_ptr<ArrayData> data);

  explicit SelectionVector(const Array& arr);

  /// \brief Create SelectionVector from boolean mask
  static Result<std::shared_ptr<SelectionVector>> FromMask(const BooleanArray& arr);

  const int32_t* indices() const { return indices_; }
  int32_t length() const;

 private:
  // 持有索引数据的生命周期。
  std::shared_ptr<ArrayData> data_;
  // 缓存指向索引数据的原始指针，以便在 indices() 方法中快速返回，避免重复的指针解引用开销。
  const int32_t* indices_;
};

/// An index to represent that a batch does not belong to an ordered stream
constexpr int64_t kUnsequencedIndex = -1;

/// \brief A unit of work for kernel execution. It contains a collection of
/// Array and Scalar values and an optional SelectionVector indicating that
/// there is an unmaterialized filter that either must be materialized, or (if
/// the kernel supports it) pushed down into the kernel implementation.
///
/// ExecBatch is semantically similar to RecordBatch in that in a SQL context
/// it represents a collection of records, but constant "columns" are
/// represented by Scalar values rather than having to be converted into arrays
/// with repeated values.
///
/// TODO: Datum uses arrow/util/variant.h which may be a bit heavier-weight
/// than is desirable for this class. Microbenchmarks would help determine for
/// sure. See ARROW-8928.

/// \addtogroup acero-internals
/// @{
// 在 Apache Arrow 的计算引擎中，ExecBatch 是物理执行层（Physical Execution Layer）处理数据的核心单位。它与 RecordBatch 类似，但针对计算任务进行了高度优化。
// ExecBatch 充当了计算内核（Kernel）的输入数据容器。
// 异构数据承载：它能同时持有数组（Array）和标量（Scalar）。在计算 col_a + 10 时，ExecBatch 的 values 会包含一个数组和一个标量。
// 计算状态描述：除了数据，它还携带了关于这批数据的额外元数据，如是否应用了过滤（SelectionVector）、数据的逻辑保证（Guarantee）等。
// 高性能分流：它摒弃了 RecordBatch 中较重的 Schema 校验，通过 Datum 向量直接对内存进行操作，适合在执行算子流水线中快速传递。
struct ARROW_EXPORT ExecBatch {
  ExecBatch() = default;
  ExecBatch(std::vector<Datum> values, int64_t length)
      : values(std::move(values)), length(length) {}

  explicit ExecBatch(const RecordBatch& batch);

  /// \brief Infer the ExecBatch length from values.
  // 遍历所有 Datum 并推断出它们共有的行数。
  static Result<int64_t> InferLength(const std::vector<Datum>& values);

  /// Creates an ExecBatch with length-validation.
  ///
  /// If any value is given, then all values must have a common length. If the given
  /// length is negative, then the length of the ExecBatch is set to this common length,
  /// or to 1 if no values are given. Otherwise, the given length must equal the common
  /// length, if any value is given.
  static Result<ExecBatch> Make(std::vector<Datum> values, int64_t length = -1);
  // 将 ExecBatch 回传给标准的 Arrow 格式。
  Result<std::shared_ptr<RecordBatch>> ToRecordBatch(
      std::shared_ptr<Schema> schema, MemoryPool* pool = default_memory_pool()) const;

  /// The values representing positional arguments to be passed to a kernel's
  /// exec function for processing.
  // 存储该批次的所有列或参数。
  // 每一项是一个 Datum，可以是 ArrayData、ChunkedArray 或 Scalar。这使得 Kernel 可以统一处理“列与列”或“列与常数”的操作。
  std::vector<Datum> values;

  /// A deferred filter represented as an array of indices into the values.
  ///
  /// For example, the filter [true, true, false, true] would be represented as
  /// the selection vector [0, 1, 3]. When the selection vector is set,
  /// ExecBatch::length is equal to the length of this array.
  // 存储一个“延迟过滤器”。
  // 说明当前 Batch 的有效数据只是 values 中被索引指向的那一部分。这避免了在过滤（Filter）操作后立即复制数据。
  std::shared_ptr<SelectionVector> selection_vector;

  /// A predicate Expression guaranteed to evaluate to true for all rows in this batch.
  // 关于该批次数据的谓词保证。
  // 例如，如果数据来自一个分区列且已知 year = 2024，则此信息可用于简化复杂的计算表达式。
  Expression guarantee = literal(true);

  /// The semantic length of the ExecBatch. When the values are all scalars,
  /// the length should be set to 1 for non-aggregate kernels, otherwise the
  /// length is taken from the array values, except when there is a selection
  /// vector. When there is a selection vector set, the length of the batch is
  /// the length of the selection. Aggregate kernels can have an ExecBatch
  /// formed by projecting just the partition columns from a batch in which
  /// case, it would have scalar rows with length greater than 1.
  ///
  /// If the array values are of length 0 then the length is 0 regardless of
  /// whether any values are Scalar.
  // 该批次的逻辑行数。
  // 如果有数组，长度通常为数组长度。
  // 如果只有标量，对于普通计算长度通常为 1；但对于聚合或分组操作，标量行可以代表一整个分组，长度可能大于 1。
  int64_t length = 0;

  /// \brief index of this batch in a sorted stream of batches
  ///
  /// This index must be strictly monotonic starting at 0 without gaps or
  /// it can be set to kUnsequencedIndex if there is no meaningful order
  // 在有序流中的序列号。
  // 当多个 Batch 在多线程中并行处理时，这个序号可以保证最后合并或输出时的顺序。
  int64_t index = kUnsequencedIndex;

  /// \brief The sum of bytes in each buffer referenced by the batch
  ///
  /// Note: Scalars are not counted
  /// Note: Some values may referenced only part of a buffer, for
  ///       example, an array with an offset.  The actual data
  ///       visible to this batch will be smaller than the total
  ///       buffer size in this case.
  int64_t TotalBufferSize() const;

  /// \brief Return the value at the i-th index
  template <typename index_type>
  inline const Datum& operator[](index_type i) const {
    return values[i];
  }

  bool Equals(const ExecBatch& other) const;

  /// \brief A convenience for the number of values / arguments.
  int num_values() const { return static_cast<int>(values.size()); }

  ExecBatch Slice(int64_t offset, int64_t length) const;

  Result<ExecBatch> SelectValues(const std::vector<int>& ids) const;

  /// \brief A convenience for returning the types from the batch.
  std::vector<TypeHolder> GetTypes() const {
    std::vector<TypeHolder> result;
    for (const auto& value : this->values) {
      result.emplace_back(value.type());
    }
    return result;
  }

  std::string ToString() const;
};

inline bool operator==(const ExecBatch& l, const ExecBatch& r) { return l.Equals(r); }
inline bool operator!=(const ExecBatch& l, const ExecBatch& r) { return !l.Equals(r); }

ARROW_EXPORT void PrintTo(const ExecBatch&, std::ostream*);

/// @}

/// \defgroup compute-internals Utilities for calling functions, useful for those
/// extending the function registry
///
/// @{
// 在 Apache Arrow 的计算引擎中，ExecValue 是一个极量级的数据包装器。如果说 Datum 是为了通用性设计的，那么 ExecValue 就是为了极致的执行性能而设计的。
// ExecValue 的核心作用是在内核（Kernel）执行期间，提供一种低开销、非持有型的方式来访问数据。
// 消除智能指针开销：Datum 内部使用 std::shared_ptr 管理 ArrayData，在循环调用中会有引用计数开销。ExecValue 则直接持有原始指针或轻量级的 ArraySpan。
// 统一标量与数组：它是一个联合容器，要么指向一个 ArraySpan（数组的切片视图），要么指向一个 Scalar。这让内核代码可以编写一套逻辑来处理这两种形态。
// 临时视图：它通常只在函数执行的生命周期内存在，不负责内存所有权，仅作为参数传递给底层的计算函数。
struct ExecValue {
  //代表一个数组的视图。
  ArraySpan array = {};
  // 指向一个标量对象的原始指针。
  const Scalar* scalar = NULLPTR;

  ExecValue(const Scalar* scalar)  // NOLINT implicit conversion
      : scalar(scalar) {}

  ExecValue(ArraySpan array)  // NOLINT implicit conversion
      : array(std::move(array)) {}

  ExecValue(const ArrayData& array) {  // NOLINT implicit conversion
    this->array.SetMembers(array);
  }

  ExecValue() = default;
  ExecValue(const ExecValue& other) = default;
  ExecValue& operator=(const ExecValue& other) = default;
  ExecValue(ExecValue&& other) = default;
  ExecValue& operator=(ExecValue&& other) = default;

  int64_t length() const { return this->is_array() ? this->array.length : 1; }

  bool is_array() const { return this->scalar == NULLPTR; }
  bool is_scalar() const { return !this->is_array(); }

  void SetArray(const ArrayData& array) {
    this->array.SetMembers(array);
    this->scalar = NULLPTR;
  }

  void SetScalar(const Scalar* scalar) { this->scalar = scalar; }

  template <typename ExactType>
  const ExactType& scalar_as() const {
    return ::arrow::internal::checked_cast<const ExactType&>(*this->scalar);
  }

  /// XXX: here temporarily for compatibility with datum, see
  /// e.g. MakeStructExec in scalar_nested.cc
  int64_t null_count() const {
    if (this->is_array()) {
      return this->array.GetNullCount();
    } else {
      return this->scalar->is_valid ? 0 : 1;
    }
  }

  const DataType* type() const {
    if (this->is_array()) {
      return array.type;
    } else {
      return scalar->type.get();
    }
  }
};

struct ARROW_EXPORT ExecResult {
  // The default value of the variant is ArraySpan
  std::variant<ArraySpan, std::shared_ptr<ArrayData>> value;

  int64_t length() const {
    if (this->is_array_span()) {
      return this->array_span()->length;
    } else {
      return this->array_data()->length;
    }
  }

  const DataType* type() const {
    if (this->is_array_span()) {
      return this->array_span()->type;
    } else {
      return this->array_data()->type.get();
    }
  }

  const ArraySpan* array_span() const { return &std::get<ArraySpan>(this->value); }
  ArraySpan* array_span_mutable() { return &std::get<ArraySpan>(this->value); }

  bool is_array_span() const { return this->value.index() == 0; }

  const std::shared_ptr<ArrayData>& array_data() const {
    return std::get<std::shared_ptr<ArrayData>>(this->value);
  }
  ArrayData* array_data_mutable() {
    return std::get<std::shared_ptr<ArrayData>>(this->value).get();
  }

  bool is_array_data() const { return this->value.index() == 1; }
};

/// \brief A "lightweight" column batch object which contains no
/// std::shared_ptr objects and does not have any memory ownership
/// semantics. Can represent a view onto an "owning" ExecBatch.
// ExecSpan 是为了极致性能而设计的“轻量级数据视图”。如果说 ExecBatch 是计算引擎中传输的“标准包裹”，那么 ExecSpan 就是为了让内核（Kernel）能够以最快速度读取数据而拆掉外壳的“裸露零件”。
// ExecSpan 的核心设计目标是：在执行热点路径中消除所有的内存管理开销。
// 无所有权语义：它不包含任何 std::shared_ptr。这意味着创建、拷贝或销毁 ExecSpan 不会触发任何原子引用计数的增减。
// 计算内核的直接输入：在 ArrayKernelExec（内核执行函数）中，输入参数通常是以 ExecSpan 形式存在的。它提供了对数据的随机访问，且数据布局极其紧凑。
// 高性能视图：它是对 ExecBatch 的一种观察（View）。当计算任务开始时，引擎会将 ExecBatch 转换成 ExecSpan 供内核使用，计算完成后再销毁。
struct ARROW_EXPORT ExecSpan {
  ExecSpan() = default;
  ExecSpan(const ExecSpan& other) = default;
  ExecSpan& operator=(const ExecSpan& other) = default;
  ExecSpan(ExecSpan&& other) = default;
  ExecSpan& operator=(ExecSpan&& other) = default;

  explicit ExecSpan(std::vector<ExecValue> values, int64_t length)
      : length(length), values(std::move(values)) {}

  explicit ExecSpan(const ExecBatch& batch) {
    this->length = batch.length;
    this->values.resize(batch.values.size());
    for (size_t i = 0; i < batch.values.size(); ++i) {
      const Datum& in_value = batch[i];
      ExecValue* out_value = &this->values[i];
      if (in_value.is_array()) {
        out_value->SetArray(*in_value.array());
      } else {
        out_value->SetScalar(in_value.scalar().get());
      }
    }
  }

  /// \brief Return the value at the i-th index
  template <typename index_type>
  inline const ExecValue& operator[](index_type i) const {
    return values[i];
  }

  /// \brief A convenience for the number of values / arguments.
  int num_values() const { return static_cast<int>(values.size()); }

  std::vector<TypeHolder> GetTypes() const {
    std::vector<TypeHolder> result;
    for (const auto& value : this->values) {
      result.emplace_back(value.type());
    }
    return result;
  }

  ExecBatch ToExecBatch() const {
    ExecBatch result;
    result.length = this->length;
    for (const ExecValue& value : this->values) {
      if (value.is_array()) {
        result.values.push_back(value.array.ToArrayData());
      } else {
        result.values.push_back(value.scalar->GetSharedPtr());
      }
    }
    return result;
  }

  int64_t length = 0;
  std::vector<ExecValue> values;
};

/// \defgroup compute-call-function One-shot calls to compute functions
///
/// @{

/// \brief One-shot invoker for all types of functions.
///
/// Does kernel dispatch, argument checking, iteration of ChunkedArray inputs,
/// and wrapping of outputs.
ARROW_EXPORT
Result<Datum> CallFunction(const std::string& func_name, const std::vector<Datum>& args,
                           const FunctionOptions* options, ExecContext* ctx = NULLPTR);

/// \brief Variant of CallFunction which uses a function's default options.
///
/// NB: Some functions require FunctionOptions be provided.
ARROW_EXPORT
Result<Datum> CallFunction(const std::string& func_name, const std::vector<Datum>& args,
                           ExecContext* ctx = NULLPTR);

/// \brief One-shot invoker for all types of functions.
///
/// Does kernel dispatch, argument checking, iteration of ChunkedArray inputs,
/// and wrapping of outputs.
ARROW_EXPORT
Result<Datum> CallFunction(const std::string& func_name, const ExecBatch& batch,
                           const FunctionOptions* options, ExecContext* ctx = NULLPTR);

/// \brief Variant of CallFunction which uses a function's default options.
///
/// NB: Some functions require FunctionOptions be provided.
ARROW_EXPORT
Result<Datum> CallFunction(const std::string& func_name, const ExecBatch& batch,
                           ExecContext* ctx = NULLPTR);

/// @}

/// \defgroup compute-function-executor One-shot calls to obtain function executors
///
/// @{

/// \brief One-shot executor provider for all types of functions.
///
/// This function creates and initializes a `FunctionExecutor` appropriate
/// for the given function name, input types and function options.
ARROW_EXPORT
Result<std::shared_ptr<FunctionExecutor>> GetFunctionExecutor(
    const std::string& func_name, std::vector<TypeHolder> in_types,
    const FunctionOptions* options = NULLPTR, FunctionRegistry* func_registry = NULLPTR);

/// \brief One-shot executor provider for all types of functions.
///
/// This function creates and initializes a `FunctionExecutor` appropriate
/// for the given function name, input types (taken from the Datum arguments)
/// and function options.
ARROW_EXPORT
Result<std::shared_ptr<FunctionExecutor>> GetFunctionExecutor(
    const std::string& func_name, const std::vector<Datum>& args,
    const FunctionOptions* options = NULLPTR, FunctionRegistry* func_registry = NULLPTR);

/// @}

}  // namespace compute
}  // namespace arrow
