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

#include <atomic>  // IWYU pragma: export
#include <cassert>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "arrow/array/statistics.h"
#include "arrow/buffer.h"
#include "arrow/result.h"
#include "arrow/type.h"
#include "arrow/type_fwd.h"
#include "arrow/util/bit_util.h"
#include "arrow/util/macros.h"
#include "arrow/util/span.h"
#include "arrow/util/visibility.h"

namespace arrow {

namespace internal {
// ----------------------------------------------------------------------
// Null handling for types without a validity bitmap and the dictionary type

ARROW_EXPORT bool IsNullSparseUnion(const ArrayData& data, int64_t i);
ARROW_EXPORT bool IsNullDenseUnion(const ArrayData& data, int64_t i);
ARROW_EXPORT bool IsNullRunEndEncoded(const ArrayData& data, int64_t i);

ARROW_EXPORT bool UnionMayHaveLogicalNulls(const ArrayData& data);
ARROW_EXPORT bool RunEndEncodedMayHaveLogicalNulls(const ArrayData& data);
ARROW_EXPORT bool DictionaryMayHaveLogicalNulls(const ArrayData& data);

}  // namespace internal

// When slicing, we do not know the null count of the sliced range without
// doing some computation. To avoid doing this eagerly, we set the null count
// to -1 (any negative number will do). When Array::null_count is called the
// first time, the null count will be computed. See ARROW-33
constexpr int64_t kUnknownNullCount = -1;

// ----------------------------------------------------------------------
// Generic array data container

/// \class ArrayData
/// \brief Mutable container for generic Arrow array data
///
/// This data structure is a self-contained representation of the memory and
/// metadata inside an Arrow array data structure (called vectors in Java). The
/// Array class and its concrete subclasses provide strongly-typed accessors
/// with support for the visitor pattern and other affordances.
///
/// This class is designed for easy internal data manipulation, analytical data
/// processing, and data transport to and from IPC messages.
///
/// This class is also useful in an analytics setting where memory may be
/// efficiently reused. For example, computing the Abs of a numeric array
/// should return null iff the input is null: therefore, an Abs function can
/// reuse the validity bitmap (a Buffer) of its input as the validity bitmap
/// of its output.
///
/// This class is meant mostly for immutable data access. Any mutable access
/// (either to ArrayData members or to the contents of its Buffers) should take
/// into account the fact that ArrayData instances are typically wrapped in a
/// shared_ptr and can therefore have multiple owners at any given time.
/// Therefore, mutable access is discouraged except when initially populating
/// the ArrayData.
// 如果说 Array 类是提供给用户使用的“精装房”（提供了丰富的类型安全接口），那么 ArrayData 就是“毛坯房”或“建筑结构”。
// 解耦逻辑与物理：它只负责持有内存缓冲区（Buffers）和基本的元数据（长度、偏移），不提供具体的类型操作接口。
// 高效的数据操作：由于它结构简单且不具备复杂的继承体系，非常适合在分析引擎内部进行数据操作、IPC（进程间通信）传输以及内存重用。
// 内存重用：例如，计算一个数组的绝对值时，由于结果的 Null 值位置与输入完全一致，ArrayData 可以直接共享输入的有效性位图（Bitmap）缓冲区，从而避免内存拷贝。
// 不可变性约定：虽然它本身是 struct 且成员公有，但文档明确指出，一旦被 Array 包装，通常应视为不可变。
struct ARROW_EXPORT ArrayData {
  ArrayData() = default;

  ArrayData(std::shared_ptr<DataType> type, int64_t length,
            int64_t null_count = kUnknownNullCount, int64_t offset = 0)
      : type(std::move(type)), length(length), null_count(null_count), offset(offset) {}

  ArrayData(std::shared_ptr<DataType> type, int64_t length,
            std::vector<std::shared_ptr<Buffer>> buffers,
            int64_t null_count = kUnknownNullCount, int64_t offset = 0)
      : ArrayData(std::move(type), length, null_count, offset) {
    this->buffers = std::move(buffers);
#ifndef NDEBUG
    // in debug mode, call the `device_type` function to trigger
    // the DCHECKs that validate all the buffers are on the same device
    ARROW_UNUSED(this->device_type());
#endif
  }

  ArrayData(std::shared_ptr<DataType> type, int64_t length,
            std::vector<std::shared_ptr<Buffer>> buffers,
            std::vector<std::shared_ptr<ArrayData>> child_data,
            int64_t null_count = kUnknownNullCount, int64_t offset = 0)
      : ArrayData(std::move(type), length, null_count, offset) {
    this->buffers = std::move(buffers);
    this->child_data = std::move(child_data);
#ifndef NDEBUG
    // in debug mode, call the `device_type` function to trigger
    // the DCHECKs that validate all the buffers (including children)
    // are on the same device
    ARROW_UNUSED(this->device_type());
#endif
  }

  static std::shared_ptr<ArrayData> Make(std::shared_ptr<DataType> type, int64_t length,
                                         std::vector<std::shared_ptr<Buffer>> buffers,
                                         int64_t null_count = kUnknownNullCount,
                                         int64_t offset = 0);

  static std::shared_ptr<ArrayData> Make(
      std::shared_ptr<DataType> type, int64_t length,
      std::vector<std::shared_ptr<Buffer>> buffers,
      std::vector<std::shared_ptr<ArrayData>> child_data,
      int64_t null_count = kUnknownNullCount, int64_t offset = 0);

  static std::shared_ptr<ArrayData> Make(
      std::shared_ptr<DataType> type, int64_t length,
      std::vector<std::shared_ptr<Buffer>> buffers,
      std::vector<std::shared_ptr<ArrayData>> child_data,
      std::shared_ptr<ArrayData> dictionary, int64_t null_count = kUnknownNullCount,
      int64_t offset = 0);

  static std::shared_ptr<ArrayData> Make(std::shared_ptr<DataType> type, int64_t length,
                                         int64_t null_count = kUnknownNullCount,
                                         int64_t offset = 0);

  // Move constructor
  ArrayData(ArrayData&& other) noexcept
      : type(std::move(other.type)),
        length(other.length),
        null_count(other.null_count.load()),
        offset(other.offset),
        buffers(std::move(other.buffers)),
        child_data(std::move(other.child_data)),
        dictionary(std::move(other.dictionary)),
        statistics(std::move(other.statistics)) {}

  // Copy constructor
  ArrayData(const ArrayData& other) noexcept
      : type(other.type),
        length(other.length),
        null_count(other.null_count.load()),
        offset(other.offset),
        buffers(other.buffers),
        child_data(other.child_data),
        dictionary(other.dictionary),
        statistics(other.statistics) {}

  // Move assignment
  ArrayData& operator=(ArrayData&& other) {
    type = std::move(other.type);
    length = other.length;
    SetNullCount(other.null_count);
    offset = other.offset;
    buffers = std::move(other.buffers);
    child_data = std::move(other.child_data);
    dictionary = std::move(other.dictionary);
    statistics = std::move(other.statistics);
    return *this;
  }

  // Copy assignment
  ArrayData& operator=(const ArrayData& other) {
    type = other.type;
    length = other.length;
    SetNullCount(other.null_count);
    offset = other.offset;
    buffers = other.buffers;
    child_data = other.child_data;
    dictionary = other.dictionary;
    statistics = other.statistics;
    return *this;
  }

  /// \brief Return a shallow copy of this ArrayData
  // Copy() 是浅拷贝（共享底层 Buffer）；
  // CopyTo() 是深拷贝（将 Buffer 移动到指定的内存管理器/设备，如 GPU）
  // ViewOrCopyTo() 则优先尝试零拷贝视图。
  std::shared_ptr<ArrayData> Copy() const { return std::make_shared<ArrayData>(*this); }

  /// \brief Deep copy this ArrayData to destination memory manager
  ///
  /// Returns a new ArrayData object with buffers and all child buffers
  /// copied to the destination memory manager. This includes dictionaries
  /// if applicable.
  Result<std::shared_ptr<ArrayData>> CopyTo(
      const std::shared_ptr<MemoryManager>& to) const;

  /// \brief View or copy this ArrayData to destination memory manager
  ///
  /// Tries to view the buffer contents on the given memory manager's device
  /// if possible (to avoid a copy) but falls back to copying if a no-copy view
  /// isn't supported.
  Result<std::shared_ptr<ArrayData>> ViewOrCopyTo(
      const std::shared_ptr<MemoryManager>& to) const;

  /// \brief Return the null-ness of a given array element
  ///
  /// Calling `IsNull(i)` is the same as `!IsValid(i)`.
  // 检查第 i 个元素是否为空。它会根据类型（Union, REE, 普通类型）自动判断是查位图还是查子节点。
  bool IsNull(int64_t i) const { return !IsValid(i); }

  /// \brief Return the validity of a given array element
  ///
  /// For most data types, this will simply query the validity bitmap.
  /// For union and run-end-encoded arrays, the underlying child data is
  /// queried instead.
  /// For dictionary arrays, this reflects the validity of the dictionary
  /// index, but the corresponding dictionary value might still be null.
  /// For null arrays, this always returns false.
  bool IsValid(int64_t i) const {
    if (buffers[0] != NULLPTR) {
      return bit_util::GetBit(buffers[0]->data(), i + offset);
    }
    const auto type = this->type->id();
    if (type == Type::SPARSE_UNION) {
      return !internal::IsNullSparseUnion(*this, i);
    }
    if (type == Type::DENSE_UNION) {
      return !internal::IsNullDenseUnion(*this, i);
    }
    if (type == Type::RUN_END_ENCODED) {
      return !internal::IsNullRunEndEncoded(*this, i);
    }
    return null_count.load() != length;
  }

  /// \brief Access a buffer's data as a typed C pointer
  ///
  /// \param i the buffer index
  /// \param absolute_offset the offset into the buffer
  ///
  /// If `absolute_offset` is non-zero, the type `T` must match the
  /// layout of buffer number `i` for the array's data type; otherwise
  /// offset computation would be incorrect.
  ///
  /// If the given buffer is bit-packed (such as a validity bitmap, or
  /// the data buffer of a boolean array), then `absolute_offset` must be
  /// zero for correct results, and any bit offset must be applied manually
  /// by the caller.
  // 将第 i 个 Buffer 转换为 T 类型的原始指针，并应用绝对偏移。
  template <typename T>
  inline const T* GetValues(int i, int64_t absolute_offset) const {
    if (buffers[i]) {
      return reinterpret_cast<const T*>(buffers[i]->data()) + absolute_offset;
    } else {
      return NULLPTR;
    }
  }

  /// \brief Access a buffer's data as a typed C pointer
  ///
  /// \param i the buffer index
  ///
  /// This method uses the array's offset to index into buffer number `i`.
  ///
  /// Calling this method on a bit-packed buffer (such as a validity bitmap, or
  /// the data buffer of a boolean array) will lead to incorrect results.
  /// You should instead call `GetValues(i, 0)` and apply the bit offset manually.
  // 快捷方式，自动应用 ArrayData 自身的 offset。
  template <typename T>
  inline const T* GetValues(int i) const {
    return GetValues<T>(i, offset);
  }

  /// \brief Access a buffer's data as a typed C pointer
  ///
  /// \param i the buffer index
  /// \param absolute_offset the offset into the buffer
  ///
  /// Like `GetValues(i, absolute_offset)`, but returns nullptr if the given buffer
  /// is not a CPU buffer.
  // 更安全，如果 Buffer 不在 CPU 上则返回空指针。
  template <typename T>
  inline const T* GetValuesSafe(int i, int64_t absolute_offset) const {
    if (buffers[i] && buffers[i]->is_cpu()) {
      return reinterpret_cast<const T*>(buffers[i]->data()) + absolute_offset;
    } else {
      return NULLPTR;
    }
  }

  /// \brief Access a buffer's data as a typed C pointer
  ///
  /// \param i the buffer index
  ///
  /// Like `GetValues(i)`, but returns nullptr if the given buffer is not a CPU buffer.
  template <typename T>
  inline const T* GetValuesSafe(int i) const {
    return GetValuesSafe<T>(i, offset);
  }

  /// \brief Access a buffer's data as a mutable typed C pointer
  ///
  /// \param i the buffer index
  /// \param absolute_offset the offset into the buffer
  ///
  /// Like `GetValues(i, absolute_offset)`, but allows mutating buffer contents.
  /// This should only be used when initially populating the ArrayData, before
  /// it is attached to a Array instance.
  // 返回可写指针。注意：仅应在构建 ArrayData 初期使用。
  template <typename T>
  inline T* GetMutableValues(int i, int64_t absolute_offset) {
    if (buffers[i]) {
      return reinterpret_cast<T*>(buffers[i]->mutable_data()) + absolute_offset;
    } else {
      return NULLPTR;
    }
  }

  /// \brief Access a buffer's data as a mutable typed C pointer
  ///
  /// \param i the buffer index
  ///
  /// Like `GetValues(i)`, but allows mutating buffer contents.
  /// This should only be used when initially populating the ArrayData, before
  /// it is attached to a Array instance.
  template <typename T>
  inline T* GetMutableValues(int i) {
    return GetMutableValues<T>(i, offset);
  }

  /// \brief Construct a zero-copy slice of the data with the given offset and length
  ///
  /// This method applies the given slice to this ArrayData, taking into account
  /// its existing offset and length.
  /// If the given `length` is too large, the slice length is clamped so as not
  /// to go past the offset end.
  /// If the given `often` is too large, or if either `offset` or `length` is negative,
  /// behavior is undefined.
  ///
  /// The associated ArrayStatistics is always discarded in a sliced
  /// ArrayData, even if the slice is trivially equal to the original ArrayData.
  /// If you want to reuse the statistics from the original ArrayData, you must
  /// explicitly reattach them.
  // 创建一个新的 ArrayData 指向同一份 Buffer，但通过调整 offset 和 length 改变逻辑视图。这是零拷贝的。
  std::shared_ptr<ArrayData> Slice(int64_t offset, int64_t length) const;

  /// \brief Construct a zero-copy slice of the data with the given offset and length
  ///
  /// Like `Slice(offset, length)`, but returns an error if the requested slice
  /// falls out of bounds.
  /// Unlike Slice, `length` isn't clamped to the available buffer size.
  Result<std::shared_ptr<ArrayData>> SliceSafe(int64_t offset, int64_t length) const;

  /// \brief Set the cached physical null count
  ///
  /// \param v the number of nulls in the ArrayData
  ///
  /// This should only be used when initially populating the ArrayData, if
  /// it possible to compute the null count without visiting the entire validity
  /// bitmap. In most cases, relying on `GetNullCount` is sufficient.
  // 获取物理 Null 计数。如果未知，则会扫描位图进行计算。
  void SetNullCount(int64_t v) { null_count.store(v); }

  /// \brief Return the physical null count
  ///
  /// This method returns the number of array elements for which `IsValid` would
  /// return false.
  ///
  /// A cached value is returned if already available, otherwise it is first
  /// computed and stored.
  /// How it is is computed depends on the data type, see `IsValid` for details.
  ///
  /// Note that this method is typically much faster than calling `IsValid`
  /// for all elements. Therefore, it helps avoid per-element validity bitmap
  /// lookups in the common cases where the array contains zero or only nulls.
  // 获取物理 Null 计数。如果未知，则会扫描位图进行计算。
  int64_t GetNullCount() const;

  /// \brief Return true if the array may have nulls in its validity bitmap
  ///
  /// This method returns true if the data has a validity bitmap, and the physical
  /// null count is either known to be non-zero or not yet known.
  ///
  /// Unlike `MayHaveLogicalNulls`, this does not check for the presence of nulls
  /// in child data for data types such as unions and run-end encoded types.
  ///
  /// \see HasValidityBitmap
  /// \see MayHaveLogicalNulls
  // 检查是否有顶级位图且计数非零；
  bool MayHaveNulls() const {
    // If an ArrayData is slightly malformed it may have kUnknownNullCount set
    // but no buffer
    return null_count.load() != 0 && buffers[0] != NULLPTR;
  }

  /// \brief Return true if the array has a validity bitmap
  // 检查是否有顶级位图且计数非零；
  bool HasValidityBitmap() const { return buffers[0] != NULLPTR; }

  /// \brief Return true if the array may have logical nulls
  ///
  /// Unlike `MayHaveNulls`, this method checks for null child values
  /// for types without a validity bitmap, such as unions and run-end encoded
  /// types, and for null dictionary values for dictionary types.
  ///
  /// This implies that `MayHaveLogicalNulls` may return true for arrays that
  /// don't have a top-level validity bitmap. It is therefore necessary
  /// to call `HasValidityBitmap` before accessing a top-level validity bitmap.
  ///
  /// Code that previously used MayHaveNulls and then dealt with the validity
  /// bitmap directly can be fixed to handle all types correctly without
  /// performance degradation when handling most types by adopting
  /// HasValidityBitmap and MayHaveLogicalNulls.
  ///
  /// Before:
  ///
  ///     uint8_t* validity = array.MayHaveNulls() ? array.buffers[0].data : NULLPTR;
  ///     for (int64_t i = 0; i < array.length; ++i) {
  ///       if (validity && !bit_util::GetBit(validity, i)) {
  ///         continue;  // skip a NULL
  ///       }
  ///       ...
  ///     }
  ///
  /// After:
  ///
  ///     bool all_valid = !array.MayHaveLogicalNulls();
  ///     uint8_t* validity = array.HasValidityBitmap() ? array.buffers[0].data : NULLPTR;
  ///     for (int64_t i = 0; i < array.length; ++i) {
  ///       bool is_valid = all_valid ||
  ///                       (validity && bit_util::GetBit(validity, i)) ||
  ///                       array.IsValid(i);
  ///       if (!is_valid) {
  ///         continue;  // skip a NULL
  ///       }
  ///       ...
  ///     }
  // 更进一步，会检查 Union 或 REE 等类型的内部逻辑是否可能包含空值。
  bool MayHaveLogicalNulls() const {
    if (buffers[0] != NULLPTR) {
      return null_count.load() != 0;
    }
    const auto t = type->id();
    if (t == Type::SPARSE_UNION || t == Type::DENSE_UNION) {
      return internal::UnionMayHaveLogicalNulls(*this);
    }
    if (t == Type::RUN_END_ENCODED) {
      return internal::RunEndEncodedMayHaveLogicalNulls(*this);
    }
    if (t == Type::DICTIONARY) {
      return internal::DictionaryMayHaveLogicalNulls(*this);
    }
    return null_count.load() != 0;
  }

  /// \brief Compute the logical null count for arrays of all types
  ///
  /// If the array has a validity bitmap, this function behaves the same as
  /// GetNullCount. For arrays that have no validity bitmap but whose values
  /// may be logically null (such as union arrays and run-end encoded arrays),
  /// this function recomputes the null count every time it is called.
  ///
  /// \see GetNullCount
  // 针对没有顶级位图的复杂类型，实时重算其逻辑空值总数。
  int64_t ComputeLogicalNullCount() const;

  /// \brief Return the device_type of the underlying buffers and children
  ///
  /// If there are no buffers in this ArrayData object, it just returns
  /// DeviceAllocationType::kCPU as a default. We also assume that all buffers
  /// should be allocated on the same device type and perform DCHECKs to confirm
  /// this in debug mode.
  ///
  /// \return DeviceAllocationType
  // 返回数据所在的设备类型（如 CPU 内存或 GPU 显存）
  DeviceAllocationType device_type() const;
  // 描述该数据的逻辑类型（如 Int32, String, Struct 等）
  std::shared_ptr<DataType> type;
  // 数组包含的元素个数
  int64_t length = 0;
  // 数组中 Null 值的数量。
  // 使用 std::atomic 是为了支持多线程下“懒加载”计算。如果值为 -1（kUnknownNullCount），表示尚未计算。
  mutable std::atomic<int64_t> null_count{0};
  // The logical start point into the physical buffers (in values, not bytes).
  // Note that, for child data, this must be *added* to the child data's own offset.
  // 逻辑起始点在物理缓冲区中的偏移量（以元素为单位，而非字节）。这支持了零拷贝切片。
  int64_t offset = 0;
  // 物理内存块。通常 buffers[0] 是有效性位图，后续 Buffer 根据 DataType 存放值或偏移量。
  std::vector<std::shared_ptr<Buffer>> buffers;
  // 用于嵌套类型（如 Struct 或 List）。每个子元素也是一个 ArrayData 对象。
  std::vector<std::shared_ptr<ArrayData>> child_data;

  // The dictionary for this Array, if any. Only used for dictionary type
  // 仅用于字典编码类型（Dictionary-encoded），存储实际的值字典。
  std::shared_ptr<ArrayData> dictionary;

  // The statistics for this Array.
  // 存储该数组的统计信息（如最小值、最大值）。
  std::shared_ptr<ArrayStatistics> statistics;
};

/// \brief A non-owning Buffer reference
struct ARROW_EXPORT BufferSpan {
  // It is the user of this class's responsibility to ensure that
  // buffers that were const originally are not written to
  // accidentally.
  uint8_t* data = NULLPTR;
  int64_t size = 0;
  // Pointer back to buffer that owns this memory
  const std::shared_ptr<Buffer>* owner = NULLPTR;

  template <typename T>
  const T* data_as() const {
    return reinterpret_cast<const T*>(data);
  }
  template <typename T>
  T* mutable_data_as() {
    return reinterpret_cast<T*>(data);
  }
};

/// \brief EXPERIMENTAL: A non-owning array data container
///
/// Unlike ArrayData, this class doesn't own its referenced data type nor data buffers.
/// It is cheaply copyable and can therefore be suitable for use cases where
/// shared_ptr overhead is not acceptable. However, care should be taken to
/// keep alive the referenced objects and memory while the ArraySpan object is in use.
/// For this reason, this should not be exposed in most public APIs (apart from
/// compute kernel interfaces).
// 在 Apache Arrow 项目中，ArraySpan 是一个极其重要的**高性能、非持有型（Non-owning）**数据容器。
// 它主要用于计算内核（Compute Kernels）内部，旨在消除 shared_ptr 带来的开销。
// ArraySpan 类似于 C++20 中的 std::span 或 std::string_view，但它是针对 Arrow 数组设计的。
// 消除引用计数开销：ArrayData 使用 std::shared_ptr 维护 Buffer 和类型，这在高性能循环或频繁调用中会有明显的原子操作开销。ArraySpan 使用原始指针和轻量级结构，拷贝代价极低。
// 计算内核的标准化输入：它是 Arrow 计算引擎内部处理数据的标准格式。在执行加法、过滤等操作时，数据会被临时转换为 ArraySpan 以获得最高性能。
// 非持有安全性（Experimental）：它不拥有数据的所有权。这意味着使用 ArraySpan 时，必须确保其引用的原始 ArrayData 或 Buffer 在整个生命周期内不会被释放。
// 紧凑的内存布局：它将缓冲区固定为 3 个（大多数 Arrow 类型足够用），对于超过 3 个的情况使用特殊处理。
struct ARROW_EXPORT ArraySpan {
  // 指向逻辑类型的原始指针（而非 shared_ptr）
  const DataType* type = NULLPTR;
  // 数组包含的元素个数。
  int64_t length = 0;
  // 缓存的空值计数。若为 -1 则表示未知。
  mutable int64_t null_count = kUnknownNullCount;
  // 相对于原始缓冲区的逻辑偏移量。
  int64_t offset = 0;
  // 这是一个包含 3 个 BufferSpan 的数组。
  // BufferSpan 结构通常包含原始数据指针 data 和字节大小 size。
  // buffers[0] 总是有效性位图（Validity Bitmap）
  // buffers[1] 和 buffers[2] 根据具体类型存放数据（如偏移量 Buffer 或值 Buffer）
  BufferSpan buffers[3];

  ArraySpan() = default;

  explicit ArraySpan(const DataType* type, int64_t length) : type(type), length(length) {}
  // 将一个有所有权的 ArrayData 转换为非持有的 ArraySpan。这是最常用的转换方式。
  ArraySpan(const ArrayData& data) {  // NOLINT implicit conversion
    SetMembers(data);
  }
  explicit ArraySpan(const Scalar& data) { FillFromScalar(data); }

  /// If dictionary-encoded, put dictionary in the first entry
  // 用于嵌套类型（如 Struct, List）
  std::vector<ArraySpan> child_data;

  /// \brief Populate ArraySpan to look like an array of length 1 pointing at
  /// the data members of a Scalar value
  // 将一个单一的标量值（Scalar）模拟成一个长度为 1 的数组。这在计算内核处理“数组 vs 标量”混合运算时非常有用。
  void FillFromScalar(const Scalar& value);
  // 内部工具函数，用于将 ArrayData 的成员（指针等）提取并填充到 ArraySpan 中。
  void SetMembers(const ArrayData& data);
  // 手动设置特定索引的缓冲区指针。
  void SetBuffer(int index, const std::shared_ptr<Buffer>& buffer) {
    this->buffers[index].data = const_cast<uint8_t*>(buffer->data());
    this->buffers[index].size = buffer->size();
    this->buffers[index].owner = &buffer;
  }

  const ArraySpan& dictionary() const { return child_data[0]; }

  /// \brief Return the number of buffers (out of 3) that are used to
  /// constitute this array
  int num_buffers() const;

  // Access a buffer's data as a typed C pointer
  // 获取第 i 个缓冲区的类型化原始指针。
  template <typename T>
  inline T* GetValues(int i, int64_t absolute_offset) {
    return reinterpret_cast<T*>(buffers[i].data) + absolute_offset;
  }
  // 如果不传 absolute_offset，它会自动应用 ArraySpan 自身的 offset。
  template <typename T>
  inline T* GetValues(int i) {
    return GetValues<T>(i, this->offset);
  }

  // Access a buffer's data as a typed C pointer
  template <typename T>
  inline const T* GetValues(int i, int64_t absolute_offset) const {
    return reinterpret_cast<const T*>(buffers[i].data) + absolute_offset;
  }

  template <typename T>
  inline const T* GetValues(int i) const {
    return GetValues<T>(i, this->offset);
  }

  /// \brief Access a buffer's data as a span
  ///
  /// \param i The buffer index
  /// \param length The required length (in number of typed values) of the requested span
  /// \pre i > 0
  /// \pre length <= the length of the buffer (in number of values) that's expected for
  /// this array type
  /// \return A span<const T> of the requested length
  // 返回一个 util::span<T>，提供一种更安全、带边界检查的连续内存访问方式。
  template <typename T>
  util::span<const T> GetSpan(int i, int64_t length) const {
    const int64_t buffer_length = buffers[i].size / static_cast<int64_t>(sizeof(T));
    assert(i > 0 && length + offset <= buffer_length);
    ARROW_UNUSED(buffer_length);
    return util::span<const T>(buffers[i].data_as<T>() + this->offset, length);
  }

  /// \brief Access a buffer's data as a span
  ///
  /// \param i The buffer index
  /// \param length The required length (in number of typed values) of the requested span
  /// \pre i > 0
  /// \pre length <= the length of the buffer (in number of values) that's expected for
  /// this array type
  /// \return A span<T> of the requested length
  template <typename T>
  util::span<T> GetSpan(int i, int64_t length) {
    const int64_t buffer_length = buffers[i].size / static_cast<int64_t>(sizeof(T));
    assert(i > 0 && length + offset <= buffer_length);
    ARROW_UNUSED(buffer_length);
    return util::span<T>(buffers[i].mutable_data_as<T>() + this->offset, length);
  }
  // 判断第 i 个元素是否为空。
  inline bool IsNull(int64_t i) const { return !IsValid(i); }
  // 判断第 i 个元素是否为空。
  inline bool IsValid(int64_t i) const {
    if (this->buffers[0].data != NULLPTR) {
      return bit_util::GetBit(this->buffers[0].data, i + this->offset);
    } else {
      const auto type = this->type->id();
      if (type == Type::SPARSE_UNION) {
        return !IsNullSparseUnion(i);
      }
      if (type == Type::DENSE_UNION) {
        return !IsNullDenseUnion(i);
      }
      if (type == Type::RUN_END_ENCODED) {
        return !IsNullRunEndEncoded(i);
      }
      return this->null_count != this->length;
    }
  }
  // 将 ArraySpan 转换回具有所有权的 ArrayData 或 Array 对象。这通常涉及创建新的 shared_ptr。
  std::shared_ptr<ArrayData> ToArrayData() const;

  std::shared_ptr<Array> ToArray() const;

  std::shared_ptr<Buffer> GetBuffer(int index) const {
    const BufferSpan& buf = this->buffers[index];
    if (buf.owner) {
      return *buf.owner;
    } else if (buf.data != NULLPTR) {
      // Buffer points to some memory without an owning buffer
      return std::make_shared<Buffer>(buf.data, buf.size);
    } else {
      return NULLPTR;
    }
  }
  // 对 ArraySpan 进行切片操作。
  // 这仅仅是修改了 offset 和 length 指标，不涉及任何内存拷贝。
  void SetSlice(int64_t offset, int64_t length) {
    this->offset = offset;
    this->length = length;
    if (this->type->id() == Type::NA) {
      this->null_count = this->length;
    } else if (buffers[0].data != NULLPTR) {
      this->null_count = kUnknownNullCount;
    } else {
      this->null_count = 0;
    }
  }

  /// \brief Return physical null count, or compute and set it if it's not known
  // 获取物理空值计数。如果缓存为 -1，则即时扫描位图并更新缓存。
  int64_t GetNullCount() const;

  /// \brief Return true if the array has a validity bitmap and the physical null
  /// count is known to be non-zero or not yet known
  ///
  /// Note that this is not the same as MayHaveLogicalNulls, which also checks
  /// for the presence of nulls in child data for types like unions and run-end
  /// encoded types.
  ///
  /// \see HasValidityBitmap
  /// \see MayHaveLogicalNulls
  // 快速判断是否可能存在空值。如果 null_count == 0 则一定没空值。
  bool MayHaveNulls() const {
    // If an ArrayData is slightly malformed it may have kUnknownNullCount set
    // but no buffer
    return null_count != 0 && buffers[0].data != NULLPTR;
  }

  /// \brief Return true if the array has a validity bitmap
  bool HasValidityBitmap() const { return buffers[0].data != NULLPTR; }

  /// \brief Return true if the validity bitmap may have 0's in it, or if the
  /// child arrays (in the case of types without a validity bitmap) may have
  /// nulls, or if the dictionary of dictionay array may have nulls.
  ///
  /// \see ArrayData::MayHaveLogicalNulls
  // 深度检查。对于没有位图但可能有逻辑空值（如 Union 类型的子项有空值）的情况进行判断。
  bool MayHaveLogicalNulls() const {
    if (buffers[0].data != NULLPTR) {
      return null_count != 0;
    }
    const auto t = type->id();
    if (t == Type::SPARSE_UNION || t == Type::DENSE_UNION) {
      return UnionMayHaveLogicalNulls();
    }
    if (t == Type::RUN_END_ENCODED) {
      return RunEndEncodedMayHaveLogicalNulls();
    }
    if (t == Type::DICTIONARY) {
      return DictionaryMayHaveLogicalNulls();
    }
    return null_count != 0;
  }

  /// \brief Compute the logical null count for arrays of all types including
  /// those that do not have a validity bitmap like union and run-end encoded
  /// arrays
  ///
  /// If the array has a validity bitmap, this function behaves the same as
  /// GetNullCount. For types that have no validity bitmap, this function will
  /// recompute the logical null count every time it is called.
  ///
  /// \see GetNullCount
  // 强制重新计算逻辑空值数量，不依赖缓存。
  int64_t ComputeLogicalNullCount() const;

  /// Some DataTypes (StringView, BinaryView) may have an arbitrary number of variadic
  /// buffers. Since ArraySpan only has 3 buffers, we pack the variadic buffers into
  /// buffers[2]; IE buffers[2].data points to the first shared_ptr<Buffer> of the
  /// variadic set and buffers[2].size is the number of variadic buffers times
  /// sizeof(shared_ptr<Buffer>).
  ///
  /// \see HasVariadicBuffers
  // 支持 StringView 或 BinaryView 等可能拥有任意数量缓冲区的类型。
  util::span<const std::shared_ptr<Buffer>> GetVariadicBuffers() const;
  bool HasVariadicBuffers() const;

 private:
  ARROW_FRIEND_EXPORT friend bool internal::IsNullRunEndEncoded(const ArrayData& data,
                                                                int64_t i);

  bool IsNullSparseUnion(int64_t i) const;
  bool IsNullDenseUnion(int64_t i) const;

  /// \brief Return true if the value at logical index i is null
  ///
  /// This function uses binary-search, so it has a O(log N) cost.
  /// Iterating over the whole array and calling IsNull is O(N log N), so
  /// for better performance it is recommended to use a
  /// ree_util::RunEndEncodedArraySpan to iterate run by run instead.
  bool IsNullRunEndEncoded(int64_t i) const;

  bool UnionMayHaveLogicalNulls() const;
  bool RunEndEncodedMayHaveLogicalNulls() const;
  bool DictionaryMayHaveLogicalNulls() const;
};

namespace internal {

void FillZeroLengthArray(const DataType* type, ArraySpan* span);

/// Construct a zero-copy view of this ArrayData with the given type.
///
/// This method checks if the types are layout-compatible.
/// Nested types are traversed in depth-first order. Data buffers must have
/// the same item sizes, even though the logical types may be different.
/// An error is returned if the types are not layout-compatible.
ARROW_EXPORT
Result<std::shared_ptr<ArrayData>> GetArrayView(const std::shared_ptr<ArrayData>& data,
                                                const std::shared_ptr<DataType>& type);

}  // namespace internal
}  // namespace arrow
