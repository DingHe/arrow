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
#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

#include "arrow/array/data.h"
#include "arrow/buffer.h"
#include "arrow/compare.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/type.h"
#include "arrow/util/bit_util.h"
#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"
#include "arrow/visitor.h"

namespace arrow {

// ----------------------------------------------------------------------
// User array accessor types

/// \brief Array base type
/// Immutable data array with some logical type and some length.
///
/// Any memory is owned by the respective Buffer instance (or its parents).
///
/// The base class is only required to have a null bitmap buffer if the null
/// count is greater than 0
///
/// If known, the null count can be provided in the base Array constructor. If
/// the null count is not known, pass -1 to indicate that the null count is to
/// be computed on the first call to null_count()
// 在 Apache Arrow 中，Array 类是整个库最核心的类之一。它代表了内存中实际存放的数据列。
// 如果说 DataType 是逻辑定义（Schema），那么 Array 就是这些定义的具体实现。
// 不可变性 (Immutability)：Array 一旦创建，其内容就是不可变的。这极大地简化了多线程环境下的内存安全。
// 零拷贝切片 (Zero-copy Slicing)：得益于内部的 offset（偏移量）设计，你可以对数组进行切片（Slice）而无需拷贝底层内存。
// 统一的空值处理：通过有效性位图（Validity Bitmap）统一管理数据中的 Null 值。
// 逻辑与物理分离：Array 类作为基类，屏蔽了底层复杂的 ArrayData 结构，为上层计算内核提供一致的访问接口。

class ARROW_EXPORT Array {
 public:
  virtual ~Array() = default;

  /// \brief Return true if value at index is null. Does not boundscheck
  // 判断第 i 个元素是否为有效值或 Null
  // 性能优化：内部直接通过位运算（GetBit）访问缓存的位图数据。针对 Union 和 REE 类型有专门的处理分支。
  bool IsNull(int64_t i) const { return !IsValid(i); }

  /// \brief Return true if value at index is valid (not null). Does not
  /// boundscheck
  bool IsValid(int64_t i) const {
    if (null_bitmap_data_ != NULLPTR) {
      return bit_util::GetBit(null_bitmap_data_, i + data_->offset);
    }
    // Dispatching with a few conditionals like this makes IsNull more
    // efficient for how it is used in practice. Making IsNull virtual
    // would add a vtable lookup to every call and prevent inlining +
    // a potential inner-branch removal.
    if (type_id() == Type::SPARSE_UNION) {
      return !internal::IsNullSparseUnion(*data_, i);
    }
    if (type_id() == Type::DENSE_UNION) {
      return !internal::IsNullDenseUnion(*data_, i);
    }
    if (type_id() == Type::RUN_END_ENCODED) {
      return !internal::IsNullRunEndEncoded(*data_, i);
    }
    return data_->null_count != data_->length;
  }

  /// \brief Return a Scalar containing the value of this array at i
  // 将第 i 个元素封装成一个 Scalar 对象返回（较慢，主要用于交互式操作）。
  Result<std::shared_ptr<Scalar>> GetScalar(int64_t i) const;

  /// Size in the number of elements this array contains.
  // 返回数组中元素的总数。
  int64_t length() const { return data_->length; }

  /// A relative position into another array's data, to enable zero-copy
  /// slicing. This value defaults to zero
  // 返回该数组相对于原始分配内存起始位置的偏移量（常用于切片）
  int64_t offset() const { return data_->offset; }

  /// The number of null entries in the array. If the null count was not known
  /// at time of construction (and set to a negative value), then the null
  /// count will be computed and cached on the first invocation of this
  /// function
  // 返回数组中 Null 值的数量
  // 如果构造时未知（值为 -1），则在第一次调用时会扫描位图进行计算并缓存结果。
  int64_t null_count() const;

  /// \brief Computes the logical null count for arrays of all types including
  /// those that do not have a validity bitmap like union and run-end encoded
  /// arrays
  ///
  /// If the array has a validity bitmap, this function behaves the same as
  /// null_count(). For types that have no validity bitmap, this function will
  /// recompute the null count every time it is called.
  ///
  /// \see GetNullCount
  // 对于没有位图的特殊类型（如 Union 或 Run-End Encoded），重新计算其逻辑上的空值数量。
  int64_t ComputeLogicalNullCount() const;
  // 返回该数组对应的逻辑数据类型（如 Int32）及其枚举 ID。
  const std::shared_ptr<DataType>& type() const { return data_->type; }
  Type::type type_id() const { return data_->type->id(); }

  /// Buffer for the validity (null) bitmap, if any. Note that Union types
  /// never have a null bitmap.
  ///
  /// Note that for `null_count == 0` or for null type, this will be null.
  /// This buffer does not account for any slice offset
  // 分别获取有效性位图的 Buffer 对象或原始指针。
  const std::shared_ptr<Buffer>& null_bitmap() const { return data_->buffers[0]; }

  /// Raw pointer to the null bitmap.
  ///
  /// Note that for `null_count == 0` or for null type, this will be null.
  /// This buffer does not account for any slice offset
  const uint8_t* null_bitmap_data() const { return null_bitmap_data_; }

  /// Equality comparison with another array
  ///
  /// Note that arrow::ArrayStatistics is not included in the comparison.
  // 判断两个数组是否相等。ApproxEquals 专门用于浮点数，支持误差（Epsilon）比较。
  bool Equals(const Array& arr, const EqualOptions& = EqualOptions::Defaults()) const;
  bool Equals(const std::shared_ptr<Array>& arr,
              const EqualOptions& = EqualOptions::Defaults()) const;

  /// \brief Return the formatted unified diff of arrow::Diff between this
  /// Array and another Array
  // 对比两个数组并以文本形式返回差异点（类似于 Unix 的 diff 命令）。
  std::string Diff(const Array& other) const;

  /// Approximate equality comparison with another array
  ///
  /// epsilon is only used if this is FloatArray or DoubleArray
  ///
  /// Note that arrow::ArrayStatistics is not included in the comparison.
  bool ApproxEquals(const std::shared_ptr<Array>& arr,
                    const EqualOptions& = EqualOptions::Defaults()) const;
  bool ApproxEquals(const Array& arr,
                    const EqualOptions& = EqualOptions::Defaults()) const;

  /// Compare if the range of slots specified are equal for the given array and
  /// this array.  end_idx exclusive.  This methods does not bounds check.
  ///
  /// Note that arrow::ArrayStatistics is not included in the comparison.
  // 比较两个数组指定范围内的元素是否相等。
  bool RangeEquals(int64_t start_idx, int64_t end_idx, int64_t other_start_idx,
                   const Array& other,
                   const EqualOptions& = EqualOptions::Defaults()) const;
  bool RangeEquals(int64_t start_idx, int64_t end_idx, int64_t other_start_idx,
                   const std::shared_ptr<Array>& other,
                   const EqualOptions& = EqualOptions::Defaults()) const;
  bool RangeEquals(const Array& other, int64_t start_idx, int64_t end_idx,
                   int64_t other_start_idx,
                   const EqualOptions& = EqualOptions::Defaults()) const;
  bool RangeEquals(const std::shared_ptr<Array>& other, int64_t start_idx,
                   int64_t end_idx, int64_t other_start_idx,
                   const EqualOptions& = EqualOptions::Defaults()) const;

  /// \brief Apply the ArrayVisitor::Visit() method specialized to the array type
  // 实现访问者模式，允许在不知道具体子类的情况下处理数组。
  Status Accept(ArrayVisitor* visitor) const;

  /// Construct a zero-copy view of this array with the given type.
  ///
  /// This method checks if the types are layout-compatible.
  /// Nested types are traversed in depth-first order. Data buffers must have
  /// the same item sizes, even though the logical types may be different.
  /// An error is returned if the types are not layout-compatible.
  // 尝试将当前数组以另一种布局兼容的类型呈现（例如将 Int32 视为 UInt32）。
  Result<std::shared_ptr<Array>> View(const std::shared_ptr<DataType>& type) const;

  /// \brief Construct a copy of the array with all buffers on destination
  /// Memory Manager
  ///
  /// This method recursively copies the array's buffers and those of its children
  /// onto the destination MemoryManager device and returns the new Array.
  // 将数组数据迁移到另一个设备（如从 CPU 内存拷贝到 GPU 显存）。
  Result<std::shared_ptr<Array>> CopyTo(const std::shared_ptr<MemoryManager>& to) const;

  /// \brief Construct a new array attempting to zero-copy view if possible.
  ///
  /// Like CopyTo this method recursively goes through all of the array's buffers
  /// and those of it's children and first attempts to create zero-copy
  /// views on the destination MemoryManager device. If it can't, it falls back
  /// to performing a copy. See Buffer::ViewOrCopy.
  // 将数组数据迁移到另一个设备（如从 CPU 内存拷贝到 GPU 显存）。
  Result<std::shared_ptr<Array>> ViewOrCopyTo(
      const std::shared_ptr<MemoryManager>& to) const;

  /// Construct a zero-copy slice of the array with the indicated offset and
  /// length
  ///
  /// \param[in] offset the position of the first element in the constructed
  /// slice
  /// \param[in] length the length of the slice. If there are not enough
  /// elements in the array, the length will be adjusted accordingly
  ///
  /// \return a new object wrapped in std::shared_ptr<Array>
  // 创建数组的一个切片。
  // 这是零拷贝操作，只是创建了一个新的 Array 对象，修改了其 offset 和 length。
  std::shared_ptr<Array> Slice(int64_t offset, int64_t length) const;

  /// Slice from offset until end of the array
  std::shared_ptr<Array> Slice(int64_t offset) const;

  /// Input-checking variant of Array::Slice
  Result<std::shared_ptr<Array>> SliceSafe(int64_t offset, int64_t length) const;
  /// Input-checking variant of Array::Slice
  Result<std::shared_ptr<Array>> SliceSafe(int64_t offset) const;

  const std::shared_ptr<ArrayData>& data() const { return data_; }

  int num_fields() const { return static_cast<int>(data_->child_data.size()); }

  /// \return PrettyPrint representation of array suitable for debugging
  std::string ToString() const;

  /// \brief Perform cheap validation checks to determine obvious inconsistencies
  /// within the array's internal data.
  ///
  /// This is O(k) where k is the number of descendents.
  ///
  /// \return Status
  // 检查数组内部逻辑是否一致（如 Buffer 大小是否匹配长度）。Full 版本会执行耗时的深度检查。
  Status Validate() const;

  /// \brief Perform extensive validation checks to determine inconsistencies
  /// within the array's internal data.
  ///
  /// This is potentially O(k*n) where k is the number of descendents and n
  /// is the array length.
  ///
  /// \return Status
  Status ValidateFull() const;

  /// \brief Return the device_type that this array's data is allocated on
  ///
  /// This just delegates to calling device_type on the underlying ArrayData
  /// object which backs this Array.
  ///
  /// \return DeviceAllocationType
  // 返回数据所在的设备类型（如 CPU 或 GPU/CUDA）
  DeviceAllocationType device_type() const { return data_->device_type(); }

  /// \brief Return the statistics of this Array
  ///
  /// This just delegates to calling statistics on the underlying ArrayData
  /// object which backs this Array.
  ///
  /// \return const std::shared_ptr<ArrayStatistics>&
  // 返回与该数组相关的统计信息（如最大值、最小值，如果存在的话）
  const std::shared_ptr<ArrayStatistics>& statistics() const { return data_->statistics; }

 protected:
  Array() = default;
  ARROW_DEFAULT_MOVE_AND_ASSIGN(Array);

  // 存储数组的核心数据结构
  // 它包含了一组缓冲区（Buffers）、长度（Length）、偏移量（Offset）和子数组数据（用于嵌套类型）。
  std::shared_ptr<ArrayData> data_;
  // 指向有效性位图原始内存的指针
  // 缓存此指针是为了在调用 IsValid(i) 时避免频繁的 shared_ptr 解引用，提升性能。
  const uint8_t* null_bitmap_data_ = NULLPTR;

  /// Protected method for constructors
  void SetData(const std::shared_ptr<ArrayData>& data) {
    if (data->buffers.size() > 0) {
      null_bitmap_data_ = data->GetValuesSafe<uint8_t>(0, /*offset=*/0);
    } else {
      null_bitmap_data_ = NULLPTR;
    }
    data_ = data;
  }

 private:
  ARROW_DISALLOW_COPY_AND_ASSIGN(Array);
};

ARROW_EXPORT void PrintTo(const Array& x, std::ostream* os);

static inline std::ostream& operator<<(std::ostream& os, const Array& x) {
  os << x.ToString();
  return os;
}

/// Base class for non-nested arrays
// 非嵌套类型（如整数、浮点数）的基类
class ARROW_EXPORT FlatArray : public Array {
 protected:
  using Array::Array;
};

/// Base class for arrays of fixed-size logical types
// 专门提供一个方法来访问存储实际数值的 Buffer（索引为 1 的 Buffer）
class ARROW_EXPORT PrimitiveArray : public FlatArray {
 public:
  /// Does not account for any slice offset
  const std::shared_ptr<Buffer>& values() const { return data_->buffers[1]; }

 protected:
  PrimitiveArray(const std::shared_ptr<DataType>& type, int64_t length,
                 const std::shared_ptr<Buffer>& data,
                 const std::shared_ptr<Buffer>& null_bitmap = NULLPTR,
                 int64_t null_count = kUnknownNullCount, int64_t offset = 0);

  PrimitiveArray() : raw_values_(NULLPTR) {}

  void SetData(const std::shared_ptr<ArrayData>& data) {
    this->Array::SetData(data);
    raw_values_ = data->GetValuesSafe<uint8_t>(1, /*offset=*/0);
  }

  explicit PrimitiveArray(const std::shared_ptr<ArrayData>& data) { SetData(data); }
  // 缓存的数值原始指针，用于极速访问。
  const uint8_t* raw_values_;
};

/// Degenerate null type Array
// 特殊的 NullType 数组。
// 特性：其 null_count 始终等于 length，且不分配任何有效性位图。
class ARROW_EXPORT NullArray : public FlatArray {
 public:
  using TypeClass = NullType;

  explicit NullArray(const std::shared_ptr<ArrayData>& data) { SetData(data); }
  explicit NullArray(int64_t length);

 private:
  void SetData(const std::shared_ptr<ArrayData>& data) {
    null_bitmap_data_ = NULLPTR;
    data->null_count = data->length;
    data_ = data;
  }
};

}  // namespace arrow
