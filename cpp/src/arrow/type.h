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

#include <atomic>
#include <climits>
#include <cstdint>
#include <iosfwd>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "arrow/result.h"
#include "arrow/type_fwd.h"  // IWYU pragma: export
#include "arrow/util/checked_cast.h"
#include "arrow/util/endian.h"
#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"
#include "arrow/visitor.h"  // IWYU pragma: keep

namespace arrow {
namespace detail {

/// \defgroup numeric-datatypes Datatypes for numeric data
/// @{
/// @}

/// \defgroup binary-datatypes Datatypes for binary/string data
/// @{
/// @}

/// \defgroup temporal-datatypes Datatypes for temporal data
/// @{
/// @}

/// \defgroup nested-datatypes Datatypes for nested data
/// @{
/// @}
// 核心作用是为复杂的对象（主要是 DataType 和 Field）提供一种高效的身份识别机制。
// 在分布式系统或内存计算中，经常需要判断两个类型是否完全相同。如果每次都通过递归遍历复杂的嵌套结构（如深层的 Struct 或 List）来比较，性能开销极大。Fingerprintable 通过生成“指纹”（Fingerprint）解决了这个问题：
// 唯一性：相同的逻辑结构产生相同的指纹字符串。
// 高性能：使用懒加载（Lazy Initialization）和原子操作，只在第一次需要时计算指纹，之后直接返回缓存的字符串指针。
// 区分元数据：它区分了“不含元数据”的指纹和“包含元数据”的指纹，方便在不同场景下（如仅逻辑匹配 vs 完全一致性匹配）使用。
// ARROW_EXPORT 作用是告诉编译器：“这个类、函数或变量不仅在库内部使用，还要公开给外部用户使用。”
// 当你编译一个项目为动态库（Windows 上的 .dll 或 Linux 上的 .so）时，编译器默认可能不会让库里的所有内容都能被外部程序访问。
// 如果不使用 ARROW_EXPORT：其他程序在链接 Arrow 库时，会找不到这个类的定义，导致“未定义的引用”（Undefined Reference）错误。
// 如果使用 ARROW_EXPORT：编译器会将该符号标记为 public，外部程序通过包含头文件并链接库文件，就能正常调用这个类。
class ARROW_EXPORT Fingerprintable {
 public:
  virtual ~Fingerprintable();
  // 获取该对象的逻辑指纹
  // 采用了**双重检查锁定（DCL）**风格的优化。
  // 首先尝试通过原子加载（load）读取指针，如果指针不为空（ARROW_PREDICT_TRUE 提示编译器这是大概率事件），则直接返回缓存结果。如果为空，则进入慢速加载路径。
  const std::string& fingerprint() const {
    auto p = fingerprint_.load();
    if (ARROW_PREDICT_TRUE(p != NULLPTR)) {
      return *p;
    }
    return LoadFingerprintSlow();
  }
  // 获取包含元数据的逻辑指纹
  // 逻辑与 fingerprint() 完全一致，区别在于它返回的是带元数据的版本。
  const std::string& metadata_fingerprint() const {
    auto p = metadata_fingerprint_.load();
    if (ARROW_PREDICT_TRUE(p != NULLPTR)) {
      return *p;
    }
    return LoadMetadataFingerprintSlow();
  }

 protected:
  // 指纹加载的“慢路径”实现。
  // 此方法在内部会调用 ComputeFingerprint()，并将生成的字符串存入内存，最后原子地更新 fingerprint_ 属性
  const std::string& LoadFingerprintSlow() const;
  // 包含元数据的指纹加载“慢路径”。
  // 逻辑同上，负责调用 ComputeMetadataFingerprint() 并缓存结果。
  const std::string& LoadMetadataFingerprintSlow() const;

  virtual std::string ComputeFingerprint() const = 0;
  virtual std::string ComputeMetadataFingerprint() const = 0;
  // 存储不含元数据的指纹字符串指针。
  mutable std::atomic<std::string*> fingerprint_{NULLPTR};
  // 存储包含元数据（Metadata）的指纹字符串指针。
  mutable std::atomic<std::string*> metadata_fingerprint_{NULLPTR};
};

}  // namespace detail

/// EXPERIMENTAL: Layout specification for a data type
// DataTypeLayout 用于详细规定一个数据类型在内存中是如何组织的。Arrow 遵循列式存储，一种逻辑类型通常由一个或多个连续的缓冲区 (Buffers) 组成。
// 例如：
// Int32 类型：需要一个有效性位图（Validity Bitmap）和一个固定宽度的值缓冲区。
// String 类型：需要一个有效性位图、一个存储偏移量的缓冲区（Offset Buffer）和一个存储原始字符的变长数据缓冲区。
// 通过这个结构体，Arrow 可以在不知道具体类型逻辑的情况下，通用的处理底层的内存分配、数据对齐和跨进程传输。
struct ARROW_EXPORT DataTypeLayout {
  // 定义了缓冲区内数据的基本特性：
  // FIXED_WIDTH: 固定宽度，如 int32 每个元素占 4 字节。
  // VARIABLE_WIDTH: 可变宽度，如 Binary 类型中存储原始字节的数据块。
  // BITMAP: 位图，通常用于有效性掩码（Validity Mask），每位代表一个值是否为 null。
  // ALWAYS_NULL: 特殊标记，表示该类型不占用物理空间（如 NullType）。
  enum BufferKind { FIXED_WIDTH, VARIABLE_WIDTH, BITMAP, ALWAYS_NULL };

  /// Layout specification for a single data type buffer
  // 描述单个缓冲区的具体规格。
  struct BufferSpec {
    // 属于上述 BufferKind 中的哪一种。
    BufferKind kind;
    // 仅在 FIXED_WIDTH 时有效，表示单个元素的字节数。
    int64_t byte_width;  // For FIXED_WIDTH

    // 用于比较两个缓冲区规格是否一致。
    bool operator==(const BufferSpec& other) const {
      return kind == other.kind &&
             (kind != FIXED_WIDTH || byte_width == other.byte_width);
    }
    bool operator!=(const BufferSpec& other) const { return !(*this == other); }
  };
  // 创建一个宽度为 w 字节的固定宽度规格。
  static BufferSpec FixedWidth(int64_t w) { return BufferSpec{FIXED_WIDTH, w}; }
  static BufferSpec VariableWidth() { return BufferSpec{VARIABLE_WIDTH, -1}; }
  static BufferSpec Bitmap() { return BufferSpec{BITMAP, -1}; }
  static BufferSpec AlwaysNull() { return BufferSpec{ALWAYS_NULL, -1}; }

  /// A vector of buffer layout specifications, one for each expected buffer
  // 核心属性。
  // 它是一个列表，按索引顺序定义了该类型所需的所有缓冲区。
  // 例如，List<Int32> 的 buffers 向量会包含一个位图规格和一个偏移量规格。
  std::vector<BufferSpec> buffers;
  /// Whether this type expects an associated dictionary array.
  // 标记该类型是否包含字典编码（Dictionary Encoding）。
  // 如果为 true，则该类型的数据并不直接存储值，而是存储字典的索引。
  bool has_dictionary = false;
  /// If this is provided, the number of buffers expected is only lower-bounded by
  /// buffers.size(). Buffers beyond this lower bound are expected to conform to
  /// variadic_spec.
  // 可变数量规格。
  // 这是为了支持像 BinaryView 这样拥有不确定数量缓冲区的类型。
  std::optional<BufferSpec> variadic_spec;

  explicit DataTypeLayout(std::vector<BufferSpec> buffers,
                          std::optional<BufferSpec> variadic_spec = {})
      : buffers(std::move(buffers)), variadic_spec(variadic_spec) {}
};

/// \brief Base class for all data types
///
/// Data types in this library are all *logical*. They can be expressed as
/// either a primitive physical type (bytes or bits of some fixed size), a
/// nested type consisting of other data types, or another data type (e.g. a
/// timestamp encoded as an int64).
///
/// Simple datatypes may be entirely described by their Type::type id, but
/// complex datatypes are usually parametric.
// 在 Apache Arrow 中，数据被分为“逻辑类型”和“物理存储”。DataType 类的主要作用如下：
// 定义逻辑语义：区分数据是整数、字符串、时间戳还是复杂的嵌套结构（如 Struct 或 List）。
// 解耦逻辑与物理：例如，Timestamp（时间戳）在逻辑上是时间，但在物理存储上可能只是一个 int64。DataType 负责维护这些逻辑层面的定义。
// Schema 树的基础：Arrow 的 Schema 是由字段（Field）组成的，而每个字段都包含一个 DataType。对于嵌套类型，DataType 内部又会包含子字段，形成递归的树状结构。
// 类型安全与分发：通过 id() 和访问者模式（Visitor Pattern），程序可以在运行时安全地判断数据类型并执行相应的逻辑（如不同的计算函数）。

class ARROW_EXPORT DataType : public std::enable_shared_from_this<DataType>, // 允许在类的成员函数内部安全地生成指向自身的 shared_ptr。
                              public detail::Fingerprintable, // 提供“指纹”功能，用于快速比较复杂类型的唯一性。
                              public util::EqualityComparable<DataType> { // 提供运算符重载（如 ==），支持对象间的相等性比较。
 public:
  // 接收一个 Type::type 枚举值（如 INT32, STRING 等）作为该类型的唯一标识符。
  explicit DataType(Type::type id) : detail::Fingerprintable(), id_(id) {}
  ~DataType() override;

  /// \brief Return whether the types are equal
  ///
  /// Types that are logically convertible from one to another (e.g. List<UInt8>
  /// and Binary) are NOT equal.
  // 判断两个逻辑类型是否相等
  // 如果两个类型逻辑上可以转换（如 List<UInt8> 和 Binary 物理内存相似），但逻辑定义不同，则返回 false。check_metadata 参数决定是否同时比较子字段中的元数据。
  bool Equals(const DataType& other, bool check_metadata = false) const;

  /// \brief Return whether the types are equal
  bool Equals(const std::shared_ptr<DataType>& other, bool check_metadata = false) const;

  /// \brief Return the child field at index i.
  // 获取索引为 i 的子字段。
  const std::shared_ptr<Field>& field(int i) const { return children_[i]; }

  /// \brief Return the children fields associated with this type.
  // 返回所有子字段的集合（即 children_ 向量）
  const FieldVector& fields() const { return children_; }

  /// \brief Return the number of children fields associated with this type.
  // 返回子字段的数量。
  // 对于基本类型（如 Int32），返回 0；对于嵌套类型（如 Struct），返回成员数量。
  int num_fields() const { return static_cast<int>(children_.size()); }

  /// \brief Apply the TypeVisitor::Visit() method specialized to the data type
  // 实现访问者模式。
  Status Accept(TypeVisitor* visitor) const;

  /// \brief A string representation of the type, including any children
  // 纯虚函数，
  // 返回类型的完整字符串表示（包含子字段）。
  virtual std::string ToString(bool show_metadata = false) const = 0;

  /// \brief Return hash value (excluding metadata in child fields)
  size_t Hash() const;

  /// \brief A string name of the type, omitting any child fields
  ///
  /// \since 0.7.0
  // 纯虚函数，
  // 返回类型的名称（不含子字段），如 "list"。
  virtual std::string name() const = 0;

  /// \brief Return the data type layout.  Children are not included.
  ///
  /// \note Experimental API
  // 纯虚函数
  // 描述该类型在内存中需要的缓冲区（Buffer）布局。
  virtual DataTypeLayout layout() const = 0;

  /// \brief Return the type category
  // 返回该类型的枚举 ID。
  constexpr Type::type id() const { return id_; }

  /// \brief Return the type category of the storage type
  // 返回该类型底层存储的 ID。通常与 id() 相同，但对于自定义逻辑类型可能不同。
  virtual Type::type storage_id() const { return id_; }

  /// \brief Returns the type's fixed byte width, if any. Returns -1
  /// for non-fixed-width types, and should only be used for
  /// subclasses of FixedWidthType
  // 返回固定宽度类型的字节数。如果是变长类型（如 String），返回 -1。
  virtual int32_t byte_width() const {
    int32_t num_bits = this->bit_width();
    return num_bits > 0 ? num_bits / 8 : -1;
  }

  /// \brief Returns the type's fixed bit width, if any. Returns -1
  /// for non-fixed-width types, and should only be used for
  /// subclasses of FixedWidthType
  // 返回固定宽度类型的位宽（Bit width）
  virtual int bit_width() const { return -1; }

  // \brief EXPERIMENTAL: Enable retrieving shared_ptr<DataType> from a const
  // context.
  // 从 const 上下文中获取指向自身的 shared_ptr
  std::shared_ptr<DataType> GetSharedPtr() const {
    return const_cast<DataType*>(this)->shared_from_this();
  }

 protected:
  // Dummy version that returns a null string (indicating not implemented).
  // Subclasses should override for fast equality checks.
  // 计算不含元数据的结构化指纹，用于快速类型比对。
  std::string ComputeFingerprint() const override;

  // Generic versions that works for all regular types, nested or not.
  // 计算包含元数据的指纹。
  std::string ComputeMetadataFingerprint() const override;
  // 存储类型 ID 的成员变量。
  Type::type id_;
  // 存储子字段的容器。例如 StructType 会在这里存储它的所有成员列。
  FieldVector children_;

 private:
  // 禁用拷贝构造函数和赋值运算符，强制通过智能指针（shared_ptr）共享类型对象。
  ARROW_DISALLOW_COPY_AND_ASSIGN(DataType);
};

/// \brief EXPERIMENTAL: Container for a type pointer which can hold a
/// dynamically created shared_ptr<DataType> if it needs to.
// TypeHolder 是 Apache Arrow 中一个非常精巧的工具类。
// 它被设计为一个轻量级的类型容器，主要用于在高性能计算路径中优化内存管理和性能。
// 在 C++ 中，std::shared_ptr 虽然安全，但频繁的拷贝（递增/递减引用计数）在高性能内循环中会产生明显的开销。
// TypeHolder 的核心作用是：统一处理“长期存在”和“临时生成”的类型对象。
// 场景 A： 你有一个现成的 DataType 指针（生命周期由外部控制），你想直接用它而不产生引用计数开销。
// 场景 B： 你在函数内部动态生成了一个类型，需要一个 shared_ptr 来保证它不被销毁。
// TypeHolder 可以同时兼容这两种场景，让代码在处理类型时无需关心它是谁拥有的，从而在保持灵活性的同时最大化性能
struct ARROW_EXPORT TypeHolder {
  // 指向数据类型的原始指针。
  // 这是访问类型信息的主要入口。
  // 无论是外部传入的原始指针，还是内部 shared_ptr 管理的对象，都会映射到这个变量上。
  const DataType* type = NULLPTR;
  // 可选的类型所有权持有者。
  // 如果该类型是动态生成的，这个 shared_ptr 会负责它的生命周期。如果是引用外部现有的类型，这个变量则为空。
  std::shared_ptr<DataType> owned_type;

  TypeHolder() = default;
  TypeHolder(const TypeHolder& other) = default;
  TypeHolder& operator=(const TypeHolder& other) = default;
  TypeHolder(TypeHolder&& other) = default;
  TypeHolder& operator=(TypeHolder&& other) = default;
  // 从共享指针构造。
  // 将 type 指向该对象，并将 shared_ptr 移动到 owned_type 中。这样 TypeHolder 就拥有了该类型的所有权。
  TypeHolder(std::shared_ptr<DataType> owned_type)  // NOLINT implicit construction
      : type(owned_type.get()), owned_type(std::move(owned_type)) {}
  // 从原始指针构造。
  // 仅记录地址，不参与生命周期管理。
  // 这在处理已知生命周期的全局类型（如内置的 int32() 类型）时极快。
  TypeHolder(const DataType* type)  // NOLINT implicit construction
      : type(type) {}

  Type::type id() const { return this->type->id(); }

  std::shared_ptr<DataType> GetSharedPtr() const {
    return this->type != NULLPTR ? this->type->GetSharedPtr() : NULLPTR;
  }

  const DataType& operator*() const { return *this->type; }

  operator bool() const { return this->type != NULLPTR; }

  bool operator==(const TypeHolder& other) const {
    if (type == other.type) return true;
    if (type == NULLPTR || other.type == NULLPTR) return false;
    return type->Equals(*other.type);
  }

  bool operator==(decltype(NULLPTR)) const { return this->type == NULLPTR; }

  bool operator==(const DataType& other) const {
    if (this->type == NULLPTR) return false;
    return other.Equals(*this->type);
  }

  bool operator!=(const DataType& other) const { return !(*this == other); }

  bool operator==(const std::shared_ptr<DataType>& other) const {
    return *this == *other;
  }

  bool operator!=(const TypeHolder& other) const { return !(*this == other); }

  std::string ToString(bool show_metadata = false) const {
    return this->type ? this->type->ToString(show_metadata) : "<NULLPTR>";
  }

  static std::string ToString(const std::vector<TypeHolder>&, bool show_metadata = false);

  static std::vector<TypeHolder> FromTypes(
      const std::vector<std::shared_ptr<DataType>>& types);
};

ARROW_EXPORT
std::ostream& operator<<(std::ostream& os, const DataType& type);

ARROW_EXPORT
std::ostream& operator<<(std::ostream& os, const TypeHolder& type);

/// \brief Return the compatible physical data type
///
/// Some types may have distinct logical meanings but the exact same physical
/// representation.  For example, TimestampType has Int64Type as a physical
/// type (defined as TimestampType::PhysicalType).
///
/// The return value is as follows:
/// - if a `PhysicalType` alias exists in the concrete type class, return
///   an instance of `PhysicalType`.
/// - otherwise, return the input type itself.
ARROW_EXPORT
std::shared_ptr<DataType> GetPhysicalType(const std::shared_ptr<DataType>& type);

/// \brief Base class for all fixed-width data types
// 固定宽度类型
// 所有在内存中占用固定字节数的数据类型的基类。除了数值，还包括布尔值（Boolean）、固定长度二进制（FixedSizeBinary）等。
class ARROW_EXPORT FixedWidthType : public DataType {
 public:
  // 继承父类的构造函数，允许使用 Type::type ID 进行初始化。
  using DataType::DataType;
  // This is only for preventing defining this class in each
  // translation unit to avoid one-definition-rule violation.
  ~FixedWidthType() override;
};

/// \brief Base class for all data types representing primitive values
// 原始 C 类型
// 代表那些可以直接映射到底层 C/C++ 原始数据类型（如 char, int, double）的 Arrow 类型。
class ARROW_EXPORT PrimitiveCType : public FixedWidthType {
 public:
  // 透传构造函数。
  using FixedWidthType::FixedWidthType;
  // This is only for preventing defining this class in each
  // translation unit to avoid one-definition-rule violation.
  ~PrimitiveCType() override;
};

/// \brief Base class for all numeric data types
// 数值类型
// 所有数字类型的父类，包括整数（Integer）和浮点数（Floating Point）。
class ARROW_EXPORT NumberType : public PrimitiveCType {
 public:
  using PrimitiveCType::PrimitiveCType;
  // This is only for preventing defining this class in each
  // translation unit to avoid one-definition-rule violation.
  ~NumberType() override;
};

/// \brief Base class for all integral data types
// 整数类型
// 所有整数类型（有符号和无符号）的抽象基类。
class ARROW_EXPORT IntegerType : public NumberType {
 public:
  // 透传构造函数。
  using NumberType::NumberType;
  // This is only for preventing defining this class in each
  // translation unit to avoid one-definition-rule violation.
  ~IntegerType() override;
  // 要求所有整数子类（如 Int32Type, UInt32Type）明确报告自己是否有符号。
  virtual bool is_signed() const = 0;
};

/// \brief Base class for all floating-point data types
class ARROW_EXPORT FloatingPointType : public NumberType {
 public:
  using NumberType::NumberType;
  // This is only for preventing defining this class in each
  // translation unit to avoid one-definition-rule violation.
  ~FloatingPointType() override;
  enum Precision { HALF, SINGLE, DOUBLE };
  virtual Precision precision() const = 0;
};

/// \brief Base class for all parametric data types
class ParametricType {};

class ARROW_EXPORT NestedType : public DataType, public ParametricType {
 public:
  using DataType::DataType;
  // This is only for preventing defining this class in each
  // translation unit to avoid one-definition-rule violation.
  ~NestedType() override;
};

/// \brief The combination of a field name and data type, with optional metadata
///
/// Fields are used to describe the individual constituents of a
/// nested DataType or a Schema.
///
/// A field's metadata is represented by a KeyValueMetadata instance,
/// which holds arbitrary key-value pairs.
// 在 Apache Arrow 中，Field 类是构建数据结构的基石之一。如果把 DataType 比作“数据长什么样”，那么 Field 就是“这一列叫什么、属于什么类型、是否有额外限制”。
// Field 类代表了** Schema 中的一个字段（列）**。它是名称（Name）、数据类型（DataType）和元数据（Metadata）的集合体。
// 定义列属性：它是 Schema（表格结构）和 NestedType（如 Struct 或 List 的成员）的基本组成单元。
// 承载元数据：允许用户为特定的列绑定键值对信息（如列描述、来源、原始数据库类型等）。
// 控制空值限制：明确规定该列是否允许包含 Null 值。
// 提供 Schema 演进基础：通过内置的 MergeWith 功能，支持在读取不同版本数据时进行类型的自动提升和合并。
class ARROW_EXPORT Field : public detail::Fingerprintable,
                           public util::EqualityComparable<Field> {
 public:
  // 构造函数。通常需要提供名称和类型，nullable 默认为 true。
  Field(std::string name, std::shared_ptr<DataType> type, bool nullable = true,
        std::shared_ptr<const KeyValueMetadata> metadata = NULLPTR)
      : detail::Fingerprintable(),
        name_(std::move(name)),
        type_(std::move(type)),
        nullable_(nullable),
        metadata_(std::move(metadata)) {}

  ~Field() override;

  /// \brief Return the field's attached metadata
  std::shared_ptr<const KeyValueMetadata> metadata() const { return metadata_; }

  /// \brief Return whether the field has non-empty metadata
  // 判断当前字段是否有关联的元数据。
  bool HasMetadata() const;

  /// \brief Return a copy of this field with the given metadata attached to it
  // 修改方法会返回一个新的 Field 对象：
  // 返回一个替换了元数据的新字段。
  std::shared_ptr<Field> WithMetadata(
      const std::shared_ptr<const KeyValueMetadata>& metadata) const;

  /// \brief EXPERIMENTAL: Return a copy of this field with the given metadata
  /// merged with existing metadata (any colliding keys will be overridden by
  /// the passed metadata)
  // 将传入的元数据与现有元数据合并（冲突时以传入的为准）。
  std::shared_ptr<Field> WithMergedMetadata(
      const std::shared_ptr<const KeyValueMetadata>& metadata) const;

  /// \brief Return a copy of this field without any metadata attached to it
  // 剥离元数据。
  std::shared_ptr<Field> RemoveMetadata() const;

  /// \brief Return a copy of this field with the replaced type.
  // 分别返回修改了名称、类型或空值属性的新字段。
  std::shared_ptr<Field> WithType(const std::shared_ptr<DataType>& type) const;

  /// \brief Return a copy of this field with the replaced name.
  std::shared_ptr<Field> WithName(const std::string& name) const;

  /// \brief Return a copy of this field with the replaced nullability.
  std::shared_ptr<Field> WithNullable(bool nullable) const;

  /// \brief Options that control the behavior of `MergeWith`.
  /// Options are to be added to allow type conversions, including integer
  /// widening, promotion from integer to float, or conversion to or from boolean.
  // 这是 Field 中非常复杂且强大的部分，用于定义在合并两个 Schema 时如何“提升”类型：
  struct ARROW_EXPORT MergeOptions : public util::ToStringOstreamable<MergeOptions> {
    /// If true, a Field of NullType can be unified with a Field of another type.
    /// The unified field will be of the other type and become nullable.
    /// Nullability will be promoted to the looser option (nullable if one is not
    /// nullable).
    // 是否允许将非空字段提升为可空字段。
    // not null + nullable = nullable
    bool promote_nullability = true;

    /// Allow a decimal to be unified with another decimal of the same
    /// width, adjusting scale and precision as appropriate. May fail
    /// if the adjustment is not possible.
    bool promote_decimal = false;

    /// Allow a decimal to be promoted to a float. The float type will
    /// not itself be promoted (e.g. Decimal128 + Float32 = Float32).
    bool promote_decimal_to_float = false;

    /// Allow an integer to be promoted to a decimal.
    ///
    /// May fail if the decimal has insufficient precision to
    /// accommodate the integer (see promote_numeric_width).
    bool promote_integer_to_decimal = false;

    /// Allow an integer of a given bit width to be promoted to a
    /// float; the result will be a float of an equal or greater bit
    /// width to both of the inputs. Examples:
    ///  - int8 + float32 = float32
    ///  - int32 + float32 = float64
    ///  - int32 + float64 = float64
    /// Because an int32 cannot always be represented exactly in the
    /// 24 bits of a float32 mantissa.
    bool promote_integer_to_float = false;

    /// Allow an unsigned integer of a given bit width to be promoted
    /// to a signed integer that fits into the signed type:
    /// uint + int16 = int16
    /// When widening is needed, set promote_numeric_width to true:
    /// uint16 + int16 = int32
    // 是否允许将无符号提升为有符号以容纳数据。
    // uint16 + int16 = int32
    bool promote_integer_sign = false;

    /// Allow an integer, float, or decimal of a given bit width to be
    /// promoted to an equivalent type of a greater bit width.
    // 是否允许数值类型加宽。
    // int16 + int32 = int32
    bool promote_numeric_width = false;

    /// Allow strings to be promoted to binary types. Promotion of fixed size
    /// binary types to variable sized formats, and binary to large binary,
    /// and string to large string.
    bool promote_binary = false;

    /// Second to millisecond, Time32 to Time64, Time32(SECOND) to Time32(MILLI), etc
    bool promote_temporal_unit = false;

    /// Allow promotion from a list to a large-list and from a fixed-size list to a
    /// variable sized list
    bool promote_list = false;

    /// Unify dictionary index types and dictionary value types.
    bool promote_dictionary = false;

    /// Allow merging ordered and non-ordered dictionaries.
    /// The result will be ordered if and only if both inputs
    /// are ordered.
    bool promote_dictionary_ordered = false;

    /// Get default options. Only NullType will be merged with other types.
    static MergeOptions Defaults() { return MergeOptions(); }
    /// Get permissive options. All options are enabled, except
    /// promote_dictionary_ordered.
    static MergeOptions Permissive();
    /// Get a human-readable representation of the options.
    std::string ToString() const;
  };

  /// \brief Merge the current field with a field of the same name.
  ///
  /// The two fields must be compatible, i.e:
  ///   - have the same name
  ///   - have the same type, or of compatible types according to `options`.
  ///
  /// The metadata of the current field is preserved; the metadata of the other
  /// field is discarded.
  // 尝试将当前字段与另一个同名字段合并。
  // 如果类型不同，会根据 MergeOptions 尝试寻找“最小公共类型”（例如 int32 合并 int64 得到 int64）。
  Result<std::shared_ptr<Field>> MergeWith(
      const Field& other, MergeOptions options = MergeOptions::Defaults()) const;
  Result<std::shared_ptr<Field>> MergeWith(
      const std::shared_ptr<Field>& other,
      MergeOptions options = MergeOptions::Defaults()) const;
  // 如果是嵌套类型（如 Struct），该方法会展开内部的子字段。
  FieldVector Flatten() const;

  /// \brief Indicate if fields are equals.
  ///
  /// \param[in] other field to check equality with.
  /// \param[in] check_metadata controls if it should check for metadata
  ///            equality.
  ///
  /// \return true if fields are equal, false otherwise.
  // 判断两个字段是否逻辑一致。可选是否严格匹配元数据。
  bool Equals(const Field& other, bool check_metadata = false) const;
  bool Equals(const std::shared_ptr<Field>& other, bool check_metadata = false) const;

  /// \brief Indicate if fields are compatibles.
  ///
  /// See the criteria of MergeWith.
  ///
  /// \return true if fields are compatible, false otherwise.
  // 判断两个字段是否可以合并或互相替代。
  bool IsCompatibleWith(const Field& other) const;
  bool IsCompatibleWith(const std::shared_ptr<Field>& other) const;

  /// \brief Return a string representation ot the field
  /// \param[in] show_metadata when true, if KeyValueMetadata is non-empty,
  /// print keys and values in the output
  // 返回易读的字符串表示。例如：user_id: int64 not null。
  std::string ToString(bool show_metadata = false) const;

  /// \brief Return the field name
  const std::string& name() const { return name_; }
  /// \brief Return the field data type
  const std::shared_ptr<DataType>& type() const { return type_; }
  /// \brief Return whether the field is nullable
  bool nullable() const { return nullable_; }

  std::shared_ptr<Field> Copy() const;

 private:
  std::string ComputeFingerprint() const override;
  std::string ComputeMetadataFingerprint() const override;

  // Field name
  // 字段的名称（例如 "User_ID"）。
  std::string name_;

  // The field's data type
  // 指向该字段数据类型的指针。
  std::shared_ptr<DataType> type_;

  // Fields can be nullable
  // 布尔值。
  // true 表示允许 Null，false 表示该列必须全为有效数据。
  bool nullable_;

  // The field's metadata, if any
  // 一个可选的容器，存储用户自定义的键值对。
  std::shared_ptr<const KeyValueMetadata> metadata_;

  ARROW_DISALLOW_COPY_AND_ASSIGN(Field);
};

ARROW_EXPORT void PrintTo(const Field& field, std::ostream* os);

namespace detail {
// 在 Apache Arrow 的源代码中，CTypeImpl 和 IntegerTypeImpl 是非常核心的模板基类。
// 它们利用 C++ 模板元编程技术，为所有“简单”或“原始”数据类型（如 Int8, Int32, Float64 等）提供统一的实现逻辑。
// CTypeImpl 的全称是 "C-Type Implementation"。其主要作用是减少代码重复（DRY原则）。
// 由于所有的数值类型（整数、浮点数、布尔值）在 Arrow 内存中的表现形式非常相似——通常都是一个有效性位图（Bitmap）加上一个固定宽度的连续内存块（Buffer）——因此没有必要为每种类型都手写一遍布局逻辑。
// CTypeImpl 通过模板参数将特定的 C++ 类型（如 int32_t）与 Arrow 的逻辑类型（如 Int32Type）绑定在一起，自动生成该类型所需的属性和方法。
// DERIVED: 最终的具体子类（例如 Int32Type）。这被用于实现 CRTP（奇异递归模板模式），以便基类可以访问子类的静态成员（如 type_name()）。
// BASE: 父类（通常是 NumberType 或 PrimitiveCType）。
// TYPE_ID: Arrow 的枚举类型 ID（如 Type::INT32）。
// C_TYPE: 对应的底层 C++ 原始类型（如 int32_t）。
template <typename DERIVED, typename BASE, Type::type TYPE_ID, typename C_TYPE>
class CTypeImpl : public BASE {
 public:
  // 在编译期保存该类型的唯一 ID。
  static constexpr Type::type type_id = TYPE_ID;
  // 定义别名，让外部可以通过 Type::c_type 获取其对应的 C++ 类型。
  using c_type = C_TYPE;
  // 定义物理存储类型，通常指向自身或其具体子类。
  using PhysicalType = DERIVED;
  // 构造函数。
  // 调用父类的构造函数并传入 TYPE_ID，确保逻辑类型被正确初始化。
  CTypeImpl() : BASE(TYPE_ID) {}
  // 返回该类型的位宽。
  int bit_width() const override { return static_cast<int>(sizeof(C_TYPE) * CHAR_BIT); }
  // 定义该类型在内存中的物理布局。
  // 返回一个标准布局：一个 Bitmap（用于标记 Null 值）加上一个 FixedWidth（宽度由 sizeof(C_TYPE) 决定）。
  DataTypeLayout layout() const override {
    return DataTypeLayout(
        {DataTypeLayout::Bitmap(), DataTypeLayout::FixedWidth(sizeof(C_TYPE))});
  }
  // 返回类型的名称。
  std::string name() const override { return DERIVED::type_name(); }
  // 返回易读的字符串表示。通常直接返回 name() 的结果。
  std::string ToString(bool show_metadata = false) const override { return this->name(); }
};

template <typename DERIVED, typename BASE, Type::type TYPE_ID, typename C_TYPE>
constexpr Type::type CTypeImpl<DERIVED, BASE, TYPE_ID, C_TYPE>::type_id;

// 这是 CTypeImpl 的一个特化版本，专门用于整数类型。
template <typename DERIVED, Type::type TYPE_ID, typename C_TYPE>
class IntegerTypeImpl : public detail::CTypeImpl<DERIVED, IntegerType, TYPE_ID, C_TYPE> {
  // 判断整数类型是否有符号。
  // 使用 C++ 标准库的 std::is_signed<C_TYPE>::value。这在编译期就能确定结果。例如，如果是 UInt32Type（对应 uint32_t），该方法自动返回 false。
  bool is_signed() const override { return std::is_signed<C_TYPE>::value; }
};

}  // namespace detail

/// Concrete type class for always-null data
// 在 Apache Arrow 中，NullType 是一个非常特殊的具体类型类，用于表示完全没有值（或所有值均为 Null）的数据列。
// NullType 代表逻辑上的“空类型”（NA）。与其它类型（如 Int32）中某个位置可能是 Null 不同，NullType 的所有元素在逻辑上都是 Null。
// 占位符：在 Schema 推断中，如果某一列完全没有数据，通常会先标记为 NullType。
// 零内存消耗：物理上，它不占用任何存储数据的值缓冲区（Value Buffer），仅在逻辑数组中记录长度。
// 兼容性：它常作为类型提升的起点，例如一个全是 Null 的列可以被“合并”或“提升”为任何其他类型（如 Int32）。
class ARROW_EXPORT NullType : public DataType {
 public:
  static constexpr Type::type type_id = Type::NA;

  static constexpr const char* type_name() { return "null"; }

  NullType() : DataType(Type::NA) {}

  std::string ToString(bool show_metadata = false) const override;

  DataTypeLayout layout() const override {
    return DataTypeLayout({DataTypeLayout::AlwaysNull()});
  }

  std::string name() const override { return "null"; }

 protected:
  std::string ComputeFingerprint() const override;
};

/// Concrete type class for boolean data
class ARROW_EXPORT BooleanType
    : public detail::CTypeImpl<BooleanType, PrimitiveCType, Type::BOOL, bool> {
 public:
  static constexpr const char* type_name() { return "bool"; }

  // BooleanType within arrow use a single bit instead of the C 8-bits layout.
  int bit_width() const final { return 1; }

  DataTypeLayout layout() const override {
    return DataTypeLayout({DataTypeLayout::Bitmap(), DataTypeLayout::Bitmap()});
  }

 protected:
  std::string ComputeFingerprint() const override;
};

/// \addtogroup numeric-datatypes
///
/// @{

/// Concrete type class for unsigned 8-bit integer data
class ARROW_EXPORT UInt8Type
    : public detail::IntegerTypeImpl<UInt8Type, Type::UINT8, uint8_t> {
 public:
  static constexpr const char* type_name() { return "uint8"; }

 protected:
  std::string ComputeFingerprint() const override;
};

/// Concrete type class for signed 8-bit integer data
class ARROW_EXPORT Int8Type
    : public detail::IntegerTypeImpl<Int8Type, Type::INT8, int8_t> {
 public:
  static constexpr const char* type_name() { return "int8"; }

 protected:
  std::string ComputeFingerprint() const override;
};

/// Concrete type class for unsigned 16-bit integer data
class ARROW_EXPORT UInt16Type
    : public detail::IntegerTypeImpl<UInt16Type, Type::UINT16, uint16_t> {
 public:
  static constexpr const char* type_name() { return "uint16"; }

 protected:
  std::string ComputeFingerprint() const override;
};

/// Concrete type class for signed 16-bit integer data
class ARROW_EXPORT Int16Type
    : public detail::IntegerTypeImpl<Int16Type, Type::INT16, int16_t> {
 public:
  static constexpr const char* type_name() { return "int16"; }

 protected:
  std::string ComputeFingerprint() const override;
};

/// Concrete type class for unsigned 32-bit integer data
class ARROW_EXPORT UInt32Type
    : public detail::IntegerTypeImpl<UInt32Type, Type::UINT32, uint32_t> {
 public:
  static constexpr const char* type_name() { return "uint32"; }

 protected:
  std::string ComputeFingerprint() const override;
};

/// Concrete type class for signed 32-bit integer data
class ARROW_EXPORT Int32Type
    : public detail::IntegerTypeImpl<Int32Type, Type::INT32, int32_t> {
 public:
  static constexpr const char* type_name() { return "int32"; }

 protected:
  std::string ComputeFingerprint() const override;
};

/// Concrete type class for unsigned 64-bit integer data
class ARROW_EXPORT UInt64Type
    : public detail::IntegerTypeImpl<UInt64Type, Type::UINT64, uint64_t> {
 public:
  static constexpr const char* type_name() { return "uint64"; }

 protected:
  std::string ComputeFingerprint() const override;
};

/// Concrete type class for signed 64-bit integer data
class ARROW_EXPORT Int64Type
    : public detail::IntegerTypeImpl<Int64Type, Type::INT64, int64_t> {
 public:
  static constexpr const char* type_name() { return "int64"; }

 protected:
  std::string ComputeFingerprint() const override;
};

/// Concrete type class for 16-bit floating-point data
class ARROW_EXPORT HalfFloatType
    : public detail::CTypeImpl<HalfFloatType, FloatingPointType, Type::HALF_FLOAT,
                               uint16_t> {
 public:
  Precision precision() const override;
  static constexpr const char* type_name() { return "halffloat"; }

 protected:
  std::string ComputeFingerprint() const override;
};

/// Concrete type class for 32-bit floating-point data (C "float")
class ARROW_EXPORT FloatType
    : public detail::CTypeImpl<FloatType, FloatingPointType, Type::FLOAT, float> {
 public:
  Precision precision() const override;
  static constexpr const char* type_name() { return "float"; }

 protected:
  std::string ComputeFingerprint() const override;
};

/// Concrete type class for 64-bit floating-point data (C "double")
class ARROW_EXPORT DoubleType
    : public detail::CTypeImpl<DoubleType, FloatingPointType, Type::DOUBLE, double> {
 public:
  Precision precision() const override;
  static constexpr const char* type_name() { return "double"; }

 protected:
  std::string ComputeFingerprint() const override;
};

/// @}

/// \brief Base class for all variable-size binary data types
class ARROW_EXPORT BaseBinaryType : public DataType {
 public:
  using DataType::DataType;
  // This is only for preventing defining this class in each
  // translation unit to avoid one-definition-rule violation.
  ~BaseBinaryType() override;
};

constexpr int64_t kBinaryMemoryLimit = std::numeric_limits<int32_t>::max() - 1;

/// \addtogroup binary-datatypes
///
/// @{

/// \brief Concrete type class for variable-size binary data
class ARROW_EXPORT BinaryType : public BaseBinaryType {
 public:
  static constexpr Type::type type_id = Type::BINARY;
  static constexpr bool is_utf8 = false;
  using offset_type = int32_t;
  using PhysicalType = BinaryType;

  static constexpr const char* type_name() { return "binary"; }

  BinaryType() : BinaryType(Type::BINARY) {}

  DataTypeLayout layout() const override {
    return DataTypeLayout({DataTypeLayout::Bitmap(),
                           DataTypeLayout::FixedWidth(sizeof(offset_type)),
                           DataTypeLayout::VariableWidth()});
  }

  std::string ToString(bool show_metadata = false) const override;
  std::string name() const override { return "binary"; }

 protected:
  std::string ComputeFingerprint() const override;

  // Allow subclasses like StringType to change the logical type.
  explicit BinaryType(Type::type logical_type) : BaseBinaryType(logical_type) {}
};

/// \brief Concrete type class for variable-size binary view data
class ARROW_EXPORT BinaryViewType : public DataType {
 public:
  static constexpr Type::type type_id = Type::BINARY_VIEW;
  static constexpr bool is_utf8 = false;
  using PhysicalType = BinaryViewType;

  static constexpr int kSize = 16;
  static constexpr int kInlineSize = 12;
  static constexpr int kPrefixSize = 4;

  /// Variable length string or binary with inline optimization for small values (12 bytes
  /// or fewer). This is similar to std::string_view except limited in size to INT32_MAX
  /// and at least the first four bytes of the string are copied inline (accessible
  /// without pointer dereference). This inline prefix allows failing comparisons early.
  /// Furthermore when dealing with short strings the CPU cache working set is reduced
  /// since many can be inline.
  ///
  /// This union supports two states:
  ///
  /// - Entirely inlined string data
  /// \code{.unparsed}
  ///                |----|--------------|
  ///                 ^    ^
  ///                 |    |
  ///              size    in-line string data, zero padded
  /// \endcode
  ///
  /// - Reference into a buffer
  /// \code{.unparsed}
  ///                |----|----|----|----|
  ///                 ^    ^    ^    ^
  ///                 |    |    |    |
  ///              size    |    |    `------.
  ///                  prefix   |           |
  ///                        buffer index   |
  ///                                  offset in buffer
  /// \endcode
  ///
  /// Adapted from TU Munich's UmbraDB [1], Velox, DuckDB.
  ///
  /// [1]: https://db.in.tum.de/~freitag/papers/p29-neumann-cidr20.pdf
  ///
  /// Alignment to 64 bits enables an aligned load of the size and prefix into
  /// a single 64 bit integer, which is useful to the comparison fast path.
  union alignas(int64_t) c_type {
    struct {
      int32_t size;
      std::array<uint8_t, kInlineSize> data;
    } inlined;

    struct {
      int32_t size;
      std::array<uint8_t, kPrefixSize> prefix;
      int32_t buffer_index;
      int32_t offset;
    } ref;

    /// The number of bytes viewed.
    int32_t size() const {
      // Size is in the common initial subsequence of each member of the union,
      // so accessing `inlined.size` is legal even if another member is active.
      return inlined.size;
    }

    /// True if the view's data is entirely stored inline.
    bool is_inline() const { return size() <= kInlineSize; }

    /// Return a pointer to the inline data of a view.
    ///
    /// For inline views, this points to the entire data of the view.
    /// For other views, this points to the 4 byte prefix.
    const uint8_t* inline_data() const& {
      // Since `ref.prefix` has the same address as `inlined.data`,
      // the branch will be trivially optimized out.
      return is_inline() ? inlined.data.data() : ref.prefix.data();
    }
    const uint8_t* inline_data() && = delete;
  };
  static_assert(sizeof(c_type) == kSize);
  static_assert(std::is_trivial_v<c_type>);

  static constexpr const char* type_name() { return "binary_view"; }

  BinaryViewType() : BinaryViewType(Type::BINARY_VIEW) {}

  DataTypeLayout layout() const override {
    return DataTypeLayout({DataTypeLayout::Bitmap(), DataTypeLayout::FixedWidth(kSize)},
                          DataTypeLayout::VariableWidth());
  }

  std::string ToString(bool show_metadata = false) const override;
  std::string name() const override { return "binary_view"; }

 protected:
  std::string ComputeFingerprint() const override;

  // Allow subclasses like StringType to change the logical type.
  explicit BinaryViewType(Type::type logical_type) : DataType(logical_type) {}
};

/// \brief Concrete type class for large variable-size binary data
class ARROW_EXPORT LargeBinaryType : public BaseBinaryType {
 public:
  static constexpr Type::type type_id = Type::LARGE_BINARY;
  static constexpr bool is_utf8 = false;
  using offset_type = int64_t;
  using PhysicalType = LargeBinaryType;

  static constexpr const char* type_name() { return "large_binary"; }

  LargeBinaryType() : LargeBinaryType(Type::LARGE_BINARY) {}

  DataTypeLayout layout() const override {
    return DataTypeLayout({DataTypeLayout::Bitmap(),
                           DataTypeLayout::FixedWidth(sizeof(offset_type)),
                           DataTypeLayout::VariableWidth()});
  }

  std::string ToString(bool show_metadata = false) const override;
  std::string name() const override { return "large_binary"; }

 protected:
  std::string ComputeFingerprint() const override;

  // Allow subclasses like LargeStringType to change the logical type.
  explicit LargeBinaryType(Type::type logical_type) : BaseBinaryType(logical_type) {}
};

/// \brief Concrete type class for variable-size string data, utf8-encoded
class ARROW_EXPORT StringType : public BinaryType {
 public:
  static constexpr Type::type type_id = Type::STRING;
  static constexpr bool is_utf8 = true;
  using PhysicalType = BinaryType;

  static constexpr const char* type_name() { return "utf8"; }

  StringType() : BinaryType(Type::STRING) {}

  std::string ToString(bool show_metadata = false) const override;
  std::string name() const override { return "utf8"; }

 protected:
  std::string ComputeFingerprint() const override;
};

/// \brief Concrete type class for variable-size string data, utf8-encoded
class ARROW_EXPORT StringViewType : public BinaryViewType {
 public:
  static constexpr Type::type type_id = Type::STRING_VIEW;
  static constexpr bool is_utf8 = true;
  using PhysicalType = BinaryViewType;

  static constexpr const char* type_name() { return "utf8_view"; }

  StringViewType() : BinaryViewType(Type::STRING_VIEW) {}

  std::string ToString(bool show_metadata = false) const override;
  std::string name() const override { return "utf8_view"; }

 protected:
  std::string ComputeFingerprint() const override;
};

/// \brief Concrete type class for large variable-size string data, utf8-encoded
class ARROW_EXPORT LargeStringType : public LargeBinaryType {
 public:
  static constexpr Type::type type_id = Type::LARGE_STRING;
  static constexpr bool is_utf8 = true;
  using PhysicalType = LargeBinaryType;

  static constexpr const char* type_name() { return "large_utf8"; }

  LargeStringType() : LargeBinaryType(Type::LARGE_STRING) {}

  std::string ToString(bool show_metadata = false) const override;
  std::string name() const override { return "large_utf8"; }

 protected:
  std::string ComputeFingerprint() const override;
};

/// \brief Concrete type class for fixed-size binary data
class ARROW_EXPORT FixedSizeBinaryType : public FixedWidthType, public ParametricType {
 public:
  static constexpr Type::type type_id = Type::FIXED_SIZE_BINARY;
  static constexpr bool is_utf8 = false;

  static constexpr const char* type_name() { return "fixed_size_binary"; }

  explicit FixedSizeBinaryType(int32_t byte_width)
      : FixedWidthType(Type::FIXED_SIZE_BINARY), byte_width_(byte_width) {}
  explicit FixedSizeBinaryType(int32_t byte_width, Type::type override_type_id)
      : FixedWidthType(override_type_id), byte_width_(byte_width) {}

  std::string ToString(bool show_metadata = false) const override;
  std::string name() const override { return "fixed_size_binary"; }

  DataTypeLayout layout() const override {
    return DataTypeLayout(
        {DataTypeLayout::Bitmap(), DataTypeLayout::FixedWidth(byte_width())});
  }

  int byte_width() const override { return byte_width_; }

  int bit_width() const override;

  // Validating constructor
  static Result<std::shared_ptr<DataType>> Make(int32_t byte_width);

 protected:
  std::string ComputeFingerprint() const override;

  int32_t byte_width_;
};

/// @}

/// \addtogroup numeric-datatypes
///
/// @{

/// \brief Base type class for (fixed-size) decimal data
class ARROW_EXPORT DecimalType : public FixedSizeBinaryType {
 public:
  explicit DecimalType(Type::type type_id, int32_t byte_width, int32_t precision,
                       int32_t scale)
      : FixedSizeBinaryType(byte_width, type_id), precision_(precision), scale_(scale) {}

  /// Constructs concrete decimal types
  static Result<std::shared_ptr<DataType>> Make(Type::type type_id, int32_t precision,
                                                int32_t scale);

  int32_t precision() const { return precision_; }
  int32_t scale() const { return scale_; }

  /// \brief Returns the number of bytes needed for precision.
  ///
  /// precision must be >= 1
  static int32_t DecimalSize(int32_t precision);

 protected:
  std::string ComputeFingerprint() const override;

  int32_t precision_;
  int32_t scale_;
};

/// \brief Concrete type class for 32-bit decimal data
///
/// Arrow decimals are fixed-point decimal numbers encoded as a scaled
/// integer.  The precision is the number of significant digits that the
/// decimal type can represent; the scale is the number of digits after
/// the decimal point (note the scale can be negative).
///
/// As an example, `Decimal32Type(7, 3)` can exactly represent the numbers
/// 1234.567 and -1234.567 (encoded internally as the 32-bit integers
/// 1234567 and -1234567, respectively), but neither 12345.67 nor 123.4567.
///
/// Decimal32Type has a maximum precision of 9 significant digits
/// (also available as Decimal32Type::kMaxPrecision).
/// If higher precision is needed, consider using Decimal64Type,
/// Decimal128Type or Decimal256Type.
class ARROW_EXPORT Decimal32Type : public DecimalType {
 public:
  static constexpr Type::type type_id = Type::DECIMAL32;

  static constexpr const char* type_name() { return "decimal32"; }

  /// Decimal32Type constructor that aborts on invalid input.
  explicit Decimal32Type(int32_t precision, int32_t scale);

  /// Decimal32Type constructor that returns an error on invalid input
  static Result<std::shared_ptr<DataType>> Make(int32_t precision, int32_t scale);

  std::string ToString(bool show_metadata = false) const override;
  std::string name() const override { return "decimal32"; }

  static constexpr int32_t kMinPrecision = 1;
  static constexpr int32_t kMaxPrecision = 9;
  static constexpr int32_t kByteWidth = 4;
};

/// \brief Concrete type class for 64-bit decimal data
///
/// Arrow decimals are fixed-point decimal numbers encoded as a scaled
/// integer.  The precision is the number of significant digits that the
/// decimal type can represent; the scale is the number of digits after
/// the decimal point (note the scale can be negative).
///
/// As an example, `Decimal64Type(7, 3)` can exactly represent the numbers
/// 1234.567 and -1234.567 (encoded internally as the 64-bit integers
/// 1234567 and -1234567, respectively), but neither 12345.67 nor 123.4567.
///
/// Decimal64Type has a maximum precision of 18 significant digits
/// (also available as Decimal64Type::kMaxPrecision).
/// If higher precision is needed, consider using Decimal128Type or
/// Decimal256Type.
class ARROW_EXPORT Decimal64Type : public DecimalType {
 public:
  static constexpr Type::type type_id = Type::DECIMAL64;

  static constexpr const char* type_name() { return "decimal64"; }

  /// Decimal32Type constructor that aborts on invalid input.
  explicit Decimal64Type(int32_t precision, int32_t scale);

  /// Decimal32Type constructor that returns an error on invalid input
  static Result<std::shared_ptr<DataType>> Make(int32_t precision, int32_t scale);

  std::string ToString(bool show_metadata = false) const override;
  std::string name() const override { return "decimal64"; }

  static constexpr int32_t kMinPrecision = 1;
  static constexpr int32_t kMaxPrecision = 18;
  static constexpr int32_t kByteWidth = 8;
};

/// \brief Concrete type class for 128-bit decimal data
///
/// Arrow decimals are fixed-point decimal numbers encoded as a scaled
/// integer.  The precision is the number of significant digits that the
/// decimal type can represent; the scale is the number of digits after
/// the decimal point (note the scale can be negative).
///
/// As an example, `Decimal128Type(7, 3)` can exactly represent the numbers
/// 1234.567 and -1234.567 (encoded internally as the 128-bit integers
/// 1234567 and -1234567, respectively), but neither 12345.67 nor 123.4567.
///
/// Decimal128Type has a maximum precision of 38 significant digits
/// (also available as Decimal128Type::kMaxPrecision).
/// If higher precision is needed, consider using Decimal256Type.
class ARROW_EXPORT Decimal128Type : public DecimalType {
 public:
  static constexpr Type::type type_id = Type::DECIMAL128;

  static constexpr const char* type_name() { return "decimal128"; }

  /// Decimal128Type constructor that aborts on invalid input.
  explicit Decimal128Type(int32_t precision, int32_t scale);

  /// Decimal128Type constructor that returns an error on invalid input.
  static Result<std::shared_ptr<DataType>> Make(int32_t precision, int32_t scale);

  std::string ToString(bool show_metadata = false) const override;
  std::string name() const override { return "decimal128"; }

  static constexpr int32_t kMinPrecision = 1;
  static constexpr int32_t kMaxPrecision = 38;
  static constexpr int32_t kByteWidth = 16;
};

/// \brief Concrete type class for 256-bit decimal data
///
/// Arrow decimals are fixed-point decimal numbers encoded as a scaled
/// integer.  The precision is the number of significant digits that the
/// decimal type can represent; the scale is the number of digits after
/// the decimal point (note the scale can be negative).
///
/// Decimal256Type has a maximum precision of 76 significant digits.
/// (also available as Decimal256Type::kMaxPrecision).
///
/// For most use cases, the maximum precision offered by Decimal128Type
/// is sufficient, and it will result in a more compact and more efficient
/// encoding.
class ARROW_EXPORT Decimal256Type : public DecimalType {
 public:
  static constexpr Type::type type_id = Type::DECIMAL256;

  static constexpr const char* type_name() { return "decimal256"; }

  /// Decimal256Type constructor that aborts on invalid input.
  explicit Decimal256Type(int32_t precision, int32_t scale);

  /// Decimal256Type constructor that returns an error on invalid input.
  static Result<std::shared_ptr<DataType>> Make(int32_t precision, int32_t scale);

  std::string ToString(bool show_metadata = false) const override;
  std::string name() const override { return "decimal256"; }

  static constexpr int32_t kMinPrecision = 1;
  static constexpr int32_t kMaxPrecision = 76;
  static constexpr int32_t kByteWidth = 32;
};

/// @}

/// \addtogroup nested-datatypes
///
/// @{

/// \brief Base class for all variable-size list data types
class ARROW_EXPORT BaseListType : public NestedType {
 public:
  using NestedType::NestedType;
  // This is only for preventing defining this class in each
  // translation unit to avoid one-definition-rule violation.
  ~BaseListType() override;
  const std::shared_ptr<Field>& value_field() const { return children_[0]; }

  const std::shared_ptr<DataType>& value_type() const { return children_[0]->type(); }
};

/// \brief Concrete type class for list data
///
/// List data is nested data where each value is a variable number of
/// child items.  Lists can be recursively nested, for example
/// list(list(int32)).
class ARROW_EXPORT ListType : public BaseListType {
 public:
  static constexpr Type::type type_id = Type::LIST;
  using offset_type = int32_t;

  static constexpr const char* type_name() { return "list"; }

  // List can contain any other logical value type
  explicit ListType(std::shared_ptr<DataType> value_type)
      : ListType(std::make_shared<Field>("item", std::move(value_type))) {}

  explicit ListType(std::shared_ptr<Field> value_field) : BaseListType(type_id) {
    children_ = {std::move(value_field)};
  }

  DataTypeLayout layout() const override {
    return DataTypeLayout(
        {DataTypeLayout::Bitmap(), DataTypeLayout::FixedWidth(sizeof(offset_type))});
  }

  std::string ToString(bool show_metadata = false) const override;

  std::string name() const override { return "list"; }

 protected:
  std::string ComputeFingerprint() const override;
};

/// \brief Concrete type class for large list data
///
/// LargeListType is like ListType but with 64-bit rather than 32-bit offsets.
class ARROW_EXPORT LargeListType : public BaseListType {
 public:
  static constexpr Type::type type_id = Type::LARGE_LIST;
  using offset_type = int64_t;

  static constexpr const char* type_name() { return "large_list"; }

  // List can contain any other logical value type
  explicit LargeListType(std::shared_ptr<DataType> value_type)
      : LargeListType(std::make_shared<Field>("item", std::move(value_type))) {}

  explicit LargeListType(std::shared_ptr<Field> value_field) : BaseListType(type_id) {
    children_ = {std::move(value_field)};
  }

  DataTypeLayout layout() const override {
    return DataTypeLayout(
        {DataTypeLayout::Bitmap(), DataTypeLayout::FixedWidth(sizeof(offset_type))});
  }

  std::string ToString(bool show_metadata = false) const override;

  std::string name() const override { return "large_list"; }

 protected:
  std::string ComputeFingerprint() const override;
};

/// \brief Type class for array of list views
class ARROW_EXPORT ListViewType : public BaseListType {
 public:
  static constexpr Type::type type_id = Type::LIST_VIEW;
  using offset_type = int32_t;

  static constexpr const char* type_name() { return "list_view"; }

  // ListView can contain any other logical value type
  explicit ListViewType(const std::shared_ptr<DataType>& value_type)
      : ListViewType(std::make_shared<Field>("item", value_type)) {}

  explicit ListViewType(const std::shared_ptr<Field>& value_field)
      : BaseListType(type_id) {
    children_ = {value_field};
  }

  DataTypeLayout layout() const override {
    return DataTypeLayout({DataTypeLayout::Bitmap(),
                           DataTypeLayout::FixedWidth(sizeof(offset_type)),
                           DataTypeLayout::FixedWidth(sizeof(offset_type))});
  }

  std::string ToString(bool show_metadata = false) const override;

  std::string name() const override { return "list_view"; }

 protected:
  std::string ComputeFingerprint() const override;
};

/// \brief Concrete type class for large list-view data
///
/// LargeListViewType is like ListViewType but with 64-bit rather than 32-bit offsets and
/// sizes.
class ARROW_EXPORT LargeListViewType : public BaseListType {
 public:
  static constexpr Type::type type_id = Type::LARGE_LIST_VIEW;
  using offset_type = int64_t;

  static constexpr const char* type_name() { return "large_list_view"; }

  // LargeListView can contain any other logical value type
  explicit LargeListViewType(const std::shared_ptr<DataType>& value_type)
      : LargeListViewType(std::make_shared<Field>("item", value_type)) {}

  explicit LargeListViewType(const std::shared_ptr<Field>& value_field)
      : BaseListType(type_id) {
    children_ = {value_field};
  }

  DataTypeLayout layout() const override {
    return DataTypeLayout({DataTypeLayout::Bitmap(),
                           DataTypeLayout::FixedWidth(sizeof(offset_type)),
                           DataTypeLayout::FixedWidth(sizeof(offset_type))});
  }

  std::string ToString(bool show_metadata = false) const override;

  std::string name() const override { return "large_list_view"; }

 protected:
  std::string ComputeFingerprint() const override;
};

/// \brief Concrete type class for map data
///
/// Map data is nested data where each value is a variable number of
/// key-item pairs.  Its physical representation is the same as
/// a list of `{key, item}` structs.
///
/// Maps can be recursively nested, for example map(utf8, map(utf8, int32)).
class ARROW_EXPORT MapType : public ListType {
 public:
  static constexpr Type::type type_id = Type::MAP;

  static constexpr const char* type_name() { return "map"; }

  MapType(std::shared_ptr<DataType> key_type, std::shared_ptr<DataType> item_type,
          bool keys_sorted = false);

  MapType(std::shared_ptr<DataType> key_type, std::shared_ptr<Field> item_field,
          bool keys_sorted = false);

  MapType(std::shared_ptr<Field> key_field, std::shared_ptr<Field> item_field,
          bool keys_sorted = false);

  explicit MapType(std::shared_ptr<Field> value_field, bool keys_sorted = false);

  // Validating constructor
  static Result<std::shared_ptr<DataType>> Make(std::shared_ptr<Field> value_field,
                                                bool keys_sorted = false);

  std::shared_ptr<Field> key_field() const { return value_type()->field(0); }
  std::shared_ptr<DataType> key_type() const { return key_field()->type(); }

  std::shared_ptr<Field> item_field() const { return value_type()->field(1); }
  std::shared_ptr<DataType> item_type() const { return item_field()->type(); }

  std::string ToString(bool show_metadata = false) const override;

  std::string name() const override { return "map"; }

  bool keys_sorted() const { return keys_sorted_; }

 private:
  std::string ComputeFingerprint() const override;

  bool keys_sorted_;
};

/// \brief Concrete type class for fixed size list data
class ARROW_EXPORT FixedSizeListType : public BaseListType {
 public:
  static constexpr Type::type type_id = Type::FIXED_SIZE_LIST;
  // While the individual item size is 32-bit, the overall data size
  // (item size * list length) may not fit in a 32-bit int.
  using offset_type = int64_t;

  static constexpr const char* type_name() { return "fixed_size_list"; }

  // List can contain any other logical value type
  FixedSizeListType(std::shared_ptr<DataType> value_type, int32_t list_size)
      : FixedSizeListType(std::make_shared<Field>("item", std::move(value_type)),
                          list_size) {}

  FixedSizeListType(std::shared_ptr<Field> value_field, int32_t list_size)
      : BaseListType(type_id), list_size_(list_size) {
    children_ = {std::move(value_field)};
  }

  DataTypeLayout layout() const override {
    return DataTypeLayout({DataTypeLayout::Bitmap()});
  }

  std::string ToString(bool show_metadata = false) const override;

  std::string name() const override { return "fixed_size_list"; }

  int32_t list_size() const { return list_size_; }

 protected:
  std::string ComputeFingerprint() const override;

  int32_t list_size_;
};

/// \brief Concrete type class for struct data
class ARROW_EXPORT StructType : public NestedType {
 public:
  static constexpr Type::type type_id = Type::STRUCT;

  static constexpr const char* type_name() { return "struct"; }

  explicit StructType(const FieldVector& fields);

  ~StructType() override;

  DataTypeLayout layout() const override {
    return DataTypeLayout({DataTypeLayout::Bitmap()});
  }

  std::string ToString(bool show_metadata = false) const override;
  std::string name() const override { return "struct"; }

  /// Returns null if name not found
  std::shared_ptr<Field> GetFieldByName(const std::string& name) const;

  /// Return all fields having this name
  FieldVector GetAllFieldsByName(const std::string& name) const;

  /// Returns -1 if name not found or if there are multiple fields having the
  /// same name
  int GetFieldIndex(const std::string& name) const;

  /// \brief Return the indices of all fields having this name in sorted order
  std::vector<int> GetAllFieldIndices(const std::string& name) const;

  /// \brief Create a new StructType with field added at given index
  Result<std::shared_ptr<StructType>> AddField(int i,
                                               const std::shared_ptr<Field>& field) const;
  /// \brief Create a new StructType by removing the field at given index
  Result<std::shared_ptr<StructType>> RemoveField(int i) const;
  /// \brief Create a new StructType by changing the field at given index
  Result<std::shared_ptr<StructType>> SetField(int i,
                                               const std::shared_ptr<Field>& field) const;

 private:
  std::string ComputeFingerprint() const override;

  class Impl;
  std::unique_ptr<Impl> impl_;
};

/// \brief Base type class for union data
class ARROW_EXPORT UnionType : public NestedType {
 public:
  static constexpr int8_t kMaxTypeCode = 127;
  static constexpr int kInvalidChildId = -1;

  static Result<std::shared_ptr<DataType>> Make(
      const FieldVector& fields, const std::vector<int8_t>& type_codes,
      UnionMode::type mode = UnionMode::SPARSE) {
    if (mode == UnionMode::SPARSE) {
      return sparse_union(fields, type_codes);
    } else {
      return dense_union(fields, type_codes);
    }
  }

  DataTypeLayout layout() const override;

  std::string ToString(bool show_metadata = false) const override;

  /// The array of logical type ids.
  ///
  /// For example, the first type in the union might be denoted by the id 5
  /// (instead of 0).
  const std::vector<int8_t>& type_codes() const { return type_codes_; }

  /// An array mapping logical type ids to physical child ids.
  const std::vector<int>& child_ids() const { return child_ids_; }

  uint8_t max_type_code() const;

  UnionMode::type mode() const;

 protected:
  UnionType(FieldVector fields, std::vector<int8_t> type_codes, Type::type id);

  static Status ValidateParameters(const FieldVector& fields,
                                   const std::vector<int8_t>& type_codes,
                                   UnionMode::type mode);

 private:
  std::string ComputeFingerprint() const override;

  std::vector<int8_t> type_codes_;
  std::vector<int> child_ids_;
};

/// \brief Concrete type class for sparse union data
///
/// A sparse union is a nested type where each logical value is taken from
/// a single child.  A buffer of 8-bit type ids indicates which child
/// a given logical value is to be taken from.
///
/// In a sparse union, each child array should have the same length as the
/// union array, regardless of the actual number of union values that
/// refer to it.
///
/// Note that, unlike most other types, unions don't have a top-level validity bitmap.
class ARROW_EXPORT SparseUnionType : public UnionType {
 public:
  static constexpr Type::type type_id = Type::SPARSE_UNION;

  static constexpr const char* type_name() { return "sparse_union"; }

  SparseUnionType(FieldVector fields, std::vector<int8_t> type_codes);

  // A constructor variant that validates input parameters
  static Result<std::shared_ptr<DataType>> Make(FieldVector fields,
                                                std::vector<int8_t> type_codes);

  std::string name() const override { return "sparse_union"; }
};

/// \brief Concrete type class for dense union data
///
/// A dense union is a nested type where each logical value is taken from
/// a single child, at a specific offset.  A buffer of 8-bit type ids
/// indicates which child a given logical value is to be taken from,
/// and a buffer of 32-bit offsets indicates at which physical position
/// in the given child array the logical value is to be taken from.
///
/// Unlike a sparse union, a dense union allows encoding only the child array
/// values which are actually referred to by the union array.  This is
/// counterbalanced by the additional footprint of the offsets buffer, and
/// the additional indirection cost when looking up values.
///
/// Note that, unlike most other types, unions don't have a top-level validity bitmap.
class ARROW_EXPORT DenseUnionType : public UnionType {
 public:
  static constexpr Type::type type_id = Type::DENSE_UNION;

  static constexpr const char* type_name() { return "dense_union"; }

  DenseUnionType(FieldVector fields, std::vector<int8_t> type_codes);

  // A constructor variant that validates input parameters
  static Result<std::shared_ptr<DataType>> Make(FieldVector fields,
                                                std::vector<int8_t> type_codes);

  std::string name() const override { return "dense_union"; }
};

/// \brief Type class for run-end encoded data
class ARROW_EXPORT RunEndEncodedType : public NestedType {
 public:
  static constexpr Type::type type_id = Type::RUN_END_ENCODED;

  static constexpr const char* type_name() { return "run_end_encoded"; }

  explicit RunEndEncodedType(std::shared_ptr<DataType> run_end_type,
                             std::shared_ptr<DataType> value_type);
  ~RunEndEncodedType() override;

  DataTypeLayout layout() const override {
    // A lot of existing code expects at least one buffer
    return DataTypeLayout({DataTypeLayout::AlwaysNull()});
  }

  const std::shared_ptr<DataType>& run_end_type() const { return fields()[0]->type(); }
  const std::shared_ptr<DataType>& value_type() const { return fields()[1]->type(); }

  std::string ToString(bool show_metadata = false) const override;

  std::string name() const override { return "run_end_encoded"; }

  static bool RunEndTypeValid(const DataType& run_end_type);

 private:
  std::string ComputeFingerprint() const override;
};

/// @}

// ----------------------------------------------------------------------
// Date and time types

/// \addtogroup temporal-datatypes
///
/// @{

/// \brief Base type for all date and time types
class ARROW_EXPORT TemporalType : public FixedWidthType {
 public:
  using FixedWidthType::FixedWidthType;
  // This is only for preventing defining this class in each
  // translation unit to avoid one-definition-rule violation.
  ~TemporalType() override;

  DataTypeLayout layout() const override {
    return DataTypeLayout(
        {DataTypeLayout::Bitmap(), DataTypeLayout::FixedWidth(bit_width() / 8)});
  }
};

/// \brief Base type class for date data
class ARROW_EXPORT DateType : public TemporalType {
 public:
  virtual DateUnit unit() const = 0;

 protected:
  explicit DateType(Type::type type_id);
};

/// Concrete type class for 32-bit date data (as number of days since UNIX epoch)
class ARROW_EXPORT Date32Type : public DateType {
 public:
  static constexpr Type::type type_id = Type::DATE32;
  static constexpr DateUnit UNIT = DateUnit::DAY;
  using c_type = int32_t;
  using PhysicalType = Int32Type;

  static constexpr const char* type_name() { return "date32"; }

  Date32Type();

  int bit_width() const override { return static_cast<int>(sizeof(c_type) * CHAR_BIT); }

  std::string ToString(bool show_metadata = false) const override;

  std::string name() const override { return "date32"; }
  DateUnit unit() const override { return UNIT; }

 protected:
  std::string ComputeFingerprint() const override;
};

/// Concrete type class for 64-bit date data (as number of milliseconds since UNIX epoch)
class ARROW_EXPORT Date64Type : public DateType {
 public:
  static constexpr Type::type type_id = Type::DATE64;
  static constexpr DateUnit UNIT = DateUnit::MILLI;
  using c_type = int64_t;
  using PhysicalType = Int64Type;

  static constexpr const char* type_name() { return "date64"; }

  Date64Type();

  int bit_width() const override { return static_cast<int>(sizeof(c_type) * CHAR_BIT); }

  std::string ToString(bool show_metadata = false) const override;

  std::string name() const override { return "date64"; }
  DateUnit unit() const override { return UNIT; }

 protected:
  std::string ComputeFingerprint() const override;
};

ARROW_EXPORT
std::ostream& operator<<(std::ostream& os, TimeUnit::type unit);

/// Base type class for time data
class ARROW_EXPORT TimeType : public TemporalType, public ParametricType {
 public:
  TimeUnit::type unit() const { return unit_; }

 protected:
  TimeType(Type::type type_id, TimeUnit::type unit);
  std::string ComputeFingerprint() const override;

  TimeUnit::type unit_;
};

/// Concrete type class for 32-bit time data (as number of seconds or milliseconds
/// since midnight)
class ARROW_EXPORT Time32Type : public TimeType {
 public:
  static constexpr Type::type type_id = Type::TIME32;
  using c_type = int32_t;
  using PhysicalType = Int32Type;

  static constexpr const char* type_name() { return "time32"; }

  int bit_width() const override { return static_cast<int>(sizeof(c_type) * CHAR_BIT); }

  explicit Time32Type(TimeUnit::type unit = TimeUnit::MILLI);

  std::string ToString(bool show_metadata = false) const override;

  std::string name() const override { return "time32"; }
};

/// Concrete type class for 64-bit time data (as number of microseconds or nanoseconds
/// since midnight)
class ARROW_EXPORT Time64Type : public TimeType {
 public:
  static constexpr Type::type type_id = Type::TIME64;
  using c_type = int64_t;
  using PhysicalType = Int64Type;

  static constexpr const char* type_name() { return "time64"; }

  int bit_width() const override { return static_cast<int>(sizeof(c_type) * CHAR_BIT); }

  explicit Time64Type(TimeUnit::type unit = TimeUnit::NANO);

  std::string ToString(bool show_metadata = false) const override;

  std::string name() const override { return "time64"; }
};

/// \brief Concrete type class for datetime data (as number of seconds, milliseconds,
/// microseconds or nanoseconds since UNIX epoch)
///
/// If supplied, the timezone string should take either the form (i) "Area/Location",
/// with values drawn from the names in the IANA Time Zone Database (such as
/// "Europe/Zurich"); or (ii) "(+|-)HH:MM" indicating an absolute offset from GMT
/// (such as "-08:00").  To indicate a native UTC timestamp, one of the strings "UTC",
/// "Etc/UTC" or "+00:00" should be used.
///
/// If any non-empty string is supplied as the timezone for a TimestampType, then the
/// Arrow field containing that timestamp type (and by extension the column associated
/// with such a field) is considered "timezone-aware".  The integer arrays that comprise
/// a timezone-aware column must contain UTC normalized datetime values, regardless of
/// the contents of their timezone string.  More precisely, (i) the producer of a
/// timezone-aware column must populate its constituent arrays with valid UTC values
/// (performing offset conversions from non-UTC values if necessary); and (ii) the
/// consumer of a timezone-aware column may assume that the column's values are directly
/// comparable (that is, with no offset adjustment required) to the values of any other
/// timezone-aware column or to any other valid UTC datetime value (provided all values
/// are expressed in the same units).
///
/// If a TimestampType is constructed without a timezone (or, equivalently, if the
/// timezone supplied is an empty string) then the resulting Arrow field (column) is
/// considered "timezone-naive".  The producer of a timezone-naive column may populate
/// its constituent integer arrays with datetime values from any timezone; the consumer
/// of a timezone-naive column should make no assumptions about the interoperability or
/// comparability of the values of such a column with those of any other timestamp
/// column or datetime value.
///
/// If a timezone-aware field contains a recognized timezone, its values may be
/// localized to that locale upon display; the values of timezone-naive fields must
/// always be displayed "as is", with no localization performed on them.
class ARROW_EXPORT TimestampType : public TemporalType, public ParametricType {
 public:
  using Unit = TimeUnit;

  static constexpr Type::type type_id = Type::TIMESTAMP;
  using c_type = int64_t;
  using PhysicalType = Int64Type;

  static constexpr const char* type_name() { return "timestamp"; }

  int bit_width() const override { return static_cast<int>(sizeof(int64_t) * CHAR_BIT); }

  explicit TimestampType(TimeUnit::type unit = TimeUnit::MILLI)
      : TemporalType(Type::TIMESTAMP), unit_(unit) {}

  explicit TimestampType(TimeUnit::type unit, const std::string& timezone)
      : TemporalType(Type::TIMESTAMP), unit_(unit), timezone_(timezone) {}

  std::string ToString(bool show_metadata = false) const override;
  std::string name() const override { return "timestamp"; }

  TimeUnit::type unit() const { return unit_; }
  const std::string& timezone() const { return timezone_; }

 protected:
  std::string ComputeFingerprint() const override;

 private:
  TimeUnit::type unit_;
  std::string timezone_;
};

// Base class for the different kinds of calendar intervals.
class ARROW_EXPORT IntervalType : public TemporalType, public ParametricType {
 public:
  enum type { MONTHS, DAY_TIME, MONTH_DAY_NANO };

  virtual type interval_type() const = 0;

 protected:
  explicit IntervalType(Type::type subtype) : TemporalType(subtype) {}
  std::string ComputeFingerprint() const override;
};

/// \brief Represents a number of months.
///
/// Type representing a number of months.  Corresponds to YearMonth type
/// in Schema.fbs (years are defined as 12 months).
class ARROW_EXPORT MonthIntervalType : public IntervalType {
 public:
  static constexpr Type::type type_id = Type::INTERVAL_MONTHS;
  using c_type = int32_t;
  using PhysicalType = Int32Type;

  static constexpr const char* type_name() { return "month_interval"; }

  IntervalType::type interval_type() const override { return IntervalType::MONTHS; }

  int bit_width() const override { return static_cast<int>(sizeof(c_type) * CHAR_BIT); }

  MonthIntervalType() : IntervalType(type_id) {}

  std::string ToString(bool ARROW_ARG_UNUSED(show_metadata) = false) const override {
    return name();
  }
  std::string name() const override { return "month_interval"; }
};

/// \brief Represents a number of days and milliseconds (fraction of day).
class ARROW_EXPORT DayTimeIntervalType : public IntervalType {
 public:
  struct DayMilliseconds {
    int32_t days = 0;
    int32_t milliseconds = 0;
    constexpr DayMilliseconds() = default;
    constexpr DayMilliseconds(int32_t days, int32_t milliseconds)
        : days(days), milliseconds(milliseconds) {}
    bool operator==(DayMilliseconds other) const {
      return this->days == other.days && this->milliseconds == other.milliseconds;
    }
    bool operator!=(DayMilliseconds other) const { return !(*this == other); }
    bool operator<(DayMilliseconds other) const {
      return this->days < other.days || this->milliseconds < other.milliseconds;
    }
  };
  using c_type = DayMilliseconds;
  using PhysicalType = DayTimeIntervalType;

  static_assert(sizeof(DayMilliseconds) == 8,
                "DayMilliseconds struct assumed to be of size 8 bytes");
  static constexpr Type::type type_id = Type::INTERVAL_DAY_TIME;

  static constexpr const char* type_name() { return "day_time_interval"; }

  IntervalType::type interval_type() const override { return IntervalType::DAY_TIME; }

  DayTimeIntervalType() : IntervalType(type_id) {}

  int bit_width() const override { return static_cast<int>(sizeof(c_type) * CHAR_BIT); }

  std::string ToString(bool ARROW_ARG_UNUSED(show_metadata) = false) const override {
    return name();
  }
  std::string name() const override { return "day_time_interval"; }
};

ARROW_EXPORT
std::ostream& operator<<(std::ostream& os, DayTimeIntervalType::DayMilliseconds interval);

/// \brief Represents a number of months, days and nanoseconds between
/// two dates.
///
/// All fields are independent from one another.
class ARROW_EXPORT MonthDayNanoIntervalType : public IntervalType {
 public:
  struct MonthDayNanos {
    int32_t months;
    int32_t days;
    int64_t nanoseconds;
    bool operator==(MonthDayNanos other) const {
      return this->months == other.months && this->days == other.days &&
             this->nanoseconds == other.nanoseconds;
    }
    bool operator!=(MonthDayNanos other) const { return !(*this == other); }
  };
  using c_type = MonthDayNanos;
  using PhysicalType = MonthDayNanoIntervalType;

  static_assert(sizeof(MonthDayNanos) == 16,
                "MonthDayNanos struct assumed to be of size 16 bytes");
  static constexpr Type::type type_id = Type::INTERVAL_MONTH_DAY_NANO;

  static constexpr const char* type_name() { return "month_day_nano_interval"; }

  IntervalType::type interval_type() const override {
    return IntervalType::MONTH_DAY_NANO;
  }

  MonthDayNanoIntervalType() : IntervalType(type_id) {}

  int bit_width() const override { return static_cast<int>(sizeof(c_type) * CHAR_BIT); }

  std::string ToString(bool ARROW_ARG_UNUSED(show_metadata) = false) const override {
    return name();
  }
  std::string name() const override { return "month_day_nano_interval"; }
};

ARROW_EXPORT
std::ostream& operator<<(std::ostream& os,
                         MonthDayNanoIntervalType::MonthDayNanos interval);

/// \brief Represents an elapsed time without any relation to a calendar artifact.
class ARROW_EXPORT DurationType : public TemporalType, public ParametricType {
 public:
  using Unit = TimeUnit;

  static constexpr Type::type type_id = Type::DURATION;
  using c_type = int64_t;
  using PhysicalType = Int64Type;

  static constexpr const char* type_name() { return "duration"; }

  int bit_width() const override { return static_cast<int>(sizeof(int64_t) * CHAR_BIT); }

  explicit DurationType(TimeUnit::type unit = TimeUnit::MILLI)
      : TemporalType(Type::DURATION), unit_(unit) {}

  std::string ToString(bool show_metadata = false) const override;
  std::string name() const override { return "duration"; }

  TimeUnit::type unit() const { return unit_; }

 protected:
  std::string ComputeFingerprint() const override;

 private:
  TimeUnit::type unit_;
};

/// @}

// ----------------------------------------------------------------------
// Dictionary type (for representing categorical or dictionary-encoded
// in memory)

/// \brief Dictionary-encoded value type with data-dependent
/// dictionary. Indices are represented by any integer types.
class ARROW_EXPORT DictionaryType : public FixedWidthType {
 public:
  static constexpr Type::type type_id = Type::DICTIONARY;

  static constexpr const char* type_name() { return "dictionary"; }

  DictionaryType(const std::shared_ptr<DataType>& index_type,
                 const std::shared_ptr<DataType>& value_type, bool ordered = false);

  // A constructor variant that validates its input parameters
  static Result<std::shared_ptr<DataType>> Make(
      const std::shared_ptr<DataType>& index_type,
      const std::shared_ptr<DataType>& value_type, bool ordered = false);

  std::string ToString(bool show_metadata = false) const override;
  std::string name() const override { return "dictionary"; }

  int bit_width() const override;

  DataTypeLayout layout() const override;

  const std::shared_ptr<DataType>& index_type() const { return index_type_; }
  const std::shared_ptr<DataType>& value_type() const { return value_type_; }

  bool ordered() const { return ordered_; }

 protected:
  static Status ValidateParameters(const DataType& index_type,
                                   const DataType& value_type);

  std::string ComputeFingerprint() const override;

  // Must be an integer type (not currently checked)
  std::shared_ptr<DataType> index_type_;
  std::shared_ptr<DataType> value_type_;
  bool ordered_;
};

// ----------------------------------------------------------------------
// FieldRef

/// \class FieldPath
///
/// Represents a path to a nested field using indices of child fields.
/// For example, given indices {5, 9, 3} the field would be retrieved with
/// schema->field(5)->type()->field(9)->type()->field(3)
///
/// Attempting to retrieve a child field using a FieldPath which is not valid for
/// a given schema will raise an error. Invalid FieldPaths include:
/// - an index is out of range
/// - the path is empty (note: a default constructed FieldPath will be empty)
///
/// FieldPaths provide a number of accessors for drilling down to potentially nested
/// children. They are overloaded for convenience to support Schema (returns a field),
/// DataType (returns a child field), Field (returns a child field of this field's type)
/// Array (returns a child array), RecordBatch (returns a column).
class ARROW_EXPORT FieldPath {
 public:
  FieldPath() = default;

  FieldPath(std::vector<int> indices)  // NOLINT runtime/explicit
      : indices_(std::move(indices)) {}

  FieldPath(std::initializer_list<int> indices)  // NOLINT runtime/explicit
      : indices_(std::move(indices)) {}

  std::string ToString() const;

  size_t hash() const;
  struct Hash {
    size_t operator()(const FieldPath& path) const { return path.hash(); }
  };

  bool empty() const { return indices_.empty(); }
  bool operator==(const FieldPath& other) const { return indices() == other.indices(); }
  bool operator!=(const FieldPath& other) const { return indices() != other.indices(); }

  const std::vector<int>& indices() const { return indices_; }
  int operator[](size_t i) const { return indices_[i]; }
  std::vector<int>::const_iterator begin() const { return indices_.begin(); }
  std::vector<int>::const_iterator end() const { return indices_.end(); }

  /// \brief Retrieve the referenced child Field from a Schema, Field, or DataType
  Result<std::shared_ptr<Field>> Get(const Schema& schema) const;
  Result<std::shared_ptr<Field>> Get(const Field& field) const;
  Result<std::shared_ptr<Field>> Get(const DataType& type) const;
  Result<std::shared_ptr<Field>> Get(const FieldVector& fields) const;

  static Result<std::shared_ptr<Schema>> GetAll(const Schema& schema,
                                                const std::vector<FieldPath>& paths);

  /// \brief Retrieve the referenced column from a RecordBatch or Table
  Result<std::shared_ptr<Array>> Get(const RecordBatch& batch) const;
  Result<std::shared_ptr<ChunkedArray>> Get(const Table& table) const;

  /// \brief Retrieve the referenced child from an Array or ArrayData
  Result<std::shared_ptr<Array>> Get(const Array& array) const;
  Result<std::shared_ptr<ArrayData>> Get(const ArrayData& data) const;

  /// \brief Retrieve the referenced child from a ChunkedArray
  Result<std::shared_ptr<ChunkedArray>> Get(const ChunkedArray& chunked_array) const;

  /// \brief Retrieve the referenced child/column from an Array, ArrayData, ChunkedArray,
  /// RecordBatch, or Table
  ///
  /// Unlike `FieldPath::Get`, these variants are not zero-copy and the retrieved child's
  /// null bitmap is ANDed with its ancestors'
  Result<std::shared_ptr<Array>> GetFlattened(const Array& array,
                                              MemoryPool* pool = NULLPTR) const;
  Result<std::shared_ptr<ArrayData>> GetFlattened(const ArrayData& data,
                                                  MemoryPool* pool = NULLPTR) const;
  Result<std::shared_ptr<ChunkedArray>> GetFlattened(const ChunkedArray& chunked_array,
                                                     MemoryPool* pool = NULLPTR) const;
  Result<std::shared_ptr<Array>> GetFlattened(const RecordBatch& batch,
                                              MemoryPool* pool = NULLPTR) const;
  Result<std::shared_ptr<ChunkedArray>> GetFlattened(const Table& table,
                                                     MemoryPool* pool = NULLPTR) const;

 private:
  std::vector<int> indices_;
};

/// \class FieldRef
/// \brief Descriptor of a (potentially nested) field within a schema.
///
/// Unlike FieldPath (which exclusively uses indices of child fields), FieldRef may
/// reference a field by name. It is intended to replace parameters like `int field_index`
/// and `const std::string& field_name`; it can be implicitly constructed from either a
/// field index or a name.
///
/// Nested fields can be referenced as well. Given
///     schema({field("a", struct_({field("n", null())})), field("b", int32())})
///
/// the following all indicate the nested field named "n":
///     FieldRef ref1(0, 0);
///     FieldRef ref2("a", 0);
///     FieldRef ref3("a", "n");
///     FieldRef ref4(0, "n");
///     ARROW_ASSIGN_OR_RAISE(FieldRef ref5,
///                           FieldRef::FromDotPath(".a[0]"));
///
/// FieldPaths matching a FieldRef are retrieved using the member function FindAll.
/// Multiple matches are possible because field names may be duplicated within a schema.
/// For example:
///     Schema a_is_ambiguous({field("a", int32()), field("a", float32())});
///     auto matches = FieldRef("a").FindAll(a_is_ambiguous);
///     assert(matches.size() == 2);
///     assert(matches[0].Get(a_is_ambiguous)->Equals(a_is_ambiguous.field(0)));
///     assert(matches[1].Get(a_is_ambiguous)->Equals(a_is_ambiguous.field(1)));
///
/// Convenience accessors are available which raise a helpful error if the field is not
/// found or ambiguous, and for immediately calling FieldPath::Get to retrieve any
/// matching children:
///     auto maybe_match = FieldRef("struct", "field_i32").FindOneOrNone(schema);
///     auto maybe_column = FieldRef("struct", "field_i32").GetOne(some_table);
class ARROW_EXPORT FieldRef : public util::EqualityComparable<FieldRef> {
 public:
  FieldRef() = default;

  /// Construct a FieldRef using a string of indices. The reference will be retrieved as:
  /// schema.fields[self.indices[0]].type.fields[self.indices[1]] ...
  ///
  /// Empty indices are not valid.
  FieldRef(FieldPath indices);  // NOLINT runtime/explicit

  /// Construct a by-name FieldRef. Multiple fields may match a by-name FieldRef:
  /// [f for f in schema.fields where f.name == self.name]
  FieldRef(std::string name) : impl_(std::move(name)) {}    // NOLINT runtime/explicit
  FieldRef(const char* name) : impl_(std::string(name)) {}  // NOLINT runtime/explicit

  /// Equivalent to a single index string of indices.
  FieldRef(int index) : impl_(FieldPath({index})) {}  // NOLINT runtime/explicit

  /// Construct a nested FieldRef.
  explicit FieldRef(std::vector<FieldRef> refs) { Flatten(std::move(refs)); }

  /// Convenience constructor for nested FieldRefs: each argument will be used to
  /// construct a FieldRef
  template <typename A0, typename A1, typename... A>
  FieldRef(A0&& a0, A1&& a1, A&&... a) {
    Flatten({// cpplint thinks the following are constructor decls
             FieldRef(std::forward<A0>(a0)),     // NOLINT runtime/explicit
             FieldRef(std::forward<A1>(a1)),     // NOLINT runtime/explicit
             FieldRef(std::forward<A>(a))...});  // NOLINT runtime/explicit
  }

  /// Parse a dot path into a FieldRef.
  ///
  /// dot_path = '.' name
  ///          | '[' digit+ ']'
  ///          | dot_path+
  ///
  /// Examples:
  ///   ".alpha" => FieldRef("alpha")
  ///   "[2]" => FieldRef(2)
  ///   ".beta[3]" => FieldRef("beta", 3)
  ///   "[5].gamma.delta[7]" => FieldRef(5, "gamma", "delta", 7)
  ///   ".hello world" => FieldRef("hello world")
  ///   R"(.\[y\]\\tho\.\)" => FieldRef(R"([y]\tho.\)")
  ///
  /// Note: When parsing a name, a '\' preceding any other character will be dropped from
  /// the resulting name. Therefore if a name must contain the characters '.', '\', or '['
  /// those must be escaped with a preceding '\'.
  static Result<FieldRef> FromDotPath(const std::string& dot_path);
  std::string ToDotPath() const;

  bool Equals(const FieldRef& other) const { return impl_ == other.impl_; }

  std::string ToString() const;

  size_t hash() const;
  struct Hash {
    size_t operator()(const FieldRef& ref) const { return ref.hash(); }
  };

  explicit operator bool() const { return Equals(FieldPath{}); }
  bool operator!() const { return !Equals(FieldPath{}); }

  bool IsFieldPath() const { return std::holds_alternative<FieldPath>(impl_); }
  bool IsName() const { return std::holds_alternative<std::string>(impl_); }
  bool IsNested() const {
    if (IsName()) return false;
    if (IsFieldPath()) return std::get<FieldPath>(impl_).indices().size() > 1;
    return true;
  }

  /// \brief Return true if this ref is a name or a nested sequence of only names
  ///
  /// Useful for determining if iteration is possible without recursion or inner loops
  bool IsNameSequence() const {
    if (IsName()) return true;
    if (const auto* nested = nested_refs()) {
      for (const auto& ref : *nested) {
        if (!ref.IsName()) return false;
      }
      return !nested->empty();
    }
    return false;
  }

  const FieldPath* field_path() const {
    return IsFieldPath() ? &std::get<FieldPath>(impl_) : NULLPTR;
  }
  const std::string* name() const {
    return IsName() ? &std::get<std::string>(impl_) : NULLPTR;
  }
  const std::vector<FieldRef>* nested_refs() const {
    return std::holds_alternative<std::vector<FieldRef>>(impl_)
               ? &std::get<std::vector<FieldRef>>(impl_)
               : NULLPTR;
  }

  /// \brief Retrieve FieldPath of every child field which matches this FieldRef.
  std::vector<FieldPath> FindAll(const Schema& schema) const;
  std::vector<FieldPath> FindAll(const Field& field) const;
  std::vector<FieldPath> FindAll(const DataType& type) const;
  std::vector<FieldPath> FindAll(const FieldVector& fields) const;

  /// \brief Convenience function which applies FindAll to arg's type or schema.
  std::vector<FieldPath> FindAll(const ArrayData& array) const;
  std::vector<FieldPath> FindAll(const Array& array) const;
  std::vector<FieldPath> FindAll(const ChunkedArray& chunked_array) const;
  std::vector<FieldPath> FindAll(const RecordBatch& batch) const;
  std::vector<FieldPath> FindAll(const Table& table) const;

  /// \brief Convenience function: raise an error if matches is empty.
  template <typename T>
  Status CheckNonEmpty(const std::vector<FieldPath>& matches, const T& root) const {
    if (matches.empty()) {
      return Status::Invalid("No match for ", ToString(), " in ", root.ToString());
    }
    return Status::OK();
  }

  /// \brief Convenience function: raise an error if matches contains multiple FieldPaths.
  template <typename T>
  Status CheckNonMultiple(const std::vector<FieldPath>& matches, const T& root) const {
    if (matches.size() > 1) {
      return Status::Invalid("Multiple matches for ", ToString(), " in ",
                             root.ToString());
    }
    return Status::OK();
  }

  /// \brief Retrieve FieldPath of a single child field which matches this
  /// FieldRef. Emit an error if none or multiple match.
  template <typename T>
  Result<FieldPath> FindOne(const T& root) const {
    auto matches = FindAll(root);
    ARROW_RETURN_NOT_OK(CheckNonEmpty(matches, root));
    ARROW_RETURN_NOT_OK(CheckNonMultiple(matches, root));
    return std::move(matches[0]);
  }

  /// \brief Retrieve FieldPath of a single child field which matches this
  /// FieldRef. Emit an error if multiple match. An empty (invalid) FieldPath
  /// will be returned if none match.
  template <typename T>
  Result<FieldPath> FindOneOrNone(const T& root) const {
    auto matches = FindAll(root);
    ARROW_RETURN_NOT_OK(CheckNonMultiple(matches, root));
    if (matches.empty()) {
      return FieldPath();
    }
    return std::move(matches[0]);
  }

  template <typename T>
  using GetType = decltype(std::declval<FieldPath>().Get(std::declval<T>()).ValueOrDie());

  /// \brief Get all children matching this FieldRef.
  template <typename T>
  std::vector<GetType<T>> GetAll(const T& root) const {
    std::vector<GetType<T>> out;
    for (const auto& match : FindAll(root)) {
      out.push_back(match.Get(root).ValueOrDie());
    }
    return out;
  }
  /// \brief Get all children matching this FieldRef.
  ///
  /// Unlike `FieldRef::GetAll`, this variant is not zero-copy and the retrieved
  /// children's null bitmaps are ANDed with their ancestors'
  template <typename T>
  Result<std::vector<GetType<T>>> GetAllFlattened(const T& root,
                                                  MemoryPool* pool = NULLPTR) const {
    std::vector<GetType<T>> out;
    for (const auto& match : FindAll(root)) {
      ARROW_ASSIGN_OR_RAISE(auto child, match.GetFlattened(root, pool));
      out.push_back(std::move(child));
    }
    return out;
  }

  /// \brief Get the single child matching this FieldRef.
  /// Emit an error if none or multiple match.
  template <typename T>
  Result<GetType<T>> GetOne(const T& root) const {
    ARROW_ASSIGN_OR_RAISE(auto match, FindOne(root));
    return match.Get(root).ValueOrDie();
  }
  /// \brief Get the single child matching this FieldRef.
  ///
  /// Unlike `FieldRef::GetOne`, this variant is not zero-copy and the retrieved
  /// child's null bitmap is ANDed with its ancestors'
  template <typename T>
  Result<GetType<T>> GetOneFlattened(const T& root, MemoryPool* pool = NULLPTR) const {
    ARROW_ASSIGN_OR_RAISE(auto match, FindOne(root));
    return match.GetFlattened(root, pool);
  }

  /// \brief Get the single child matching this FieldRef.
  /// Return nullptr if none match, emit an error if multiple match.
  template <typename T>
  Result<GetType<T>> GetOneOrNone(const T& root) const {
    ARROW_ASSIGN_OR_RAISE(auto match, FindOneOrNone(root));
    if (match.empty()) {
      return static_cast<GetType<T>>(NULLPTR);
    }
    return match.Get(root).ValueOrDie();
  }
  /// \brief Get the single child matching this FieldRef.
  ///
  /// Return nullptr if none match, emit an error if multiple match.
  /// Unlike `FieldRef::GetOneOrNone`, this variant is not zero-copy and the
  /// retrieved child's null bitmap is ANDed with its ancestors'
  template <typename T>
  Result<GetType<T>> GetOneOrNoneFlattened(const T& root,
                                           MemoryPool* pool = NULLPTR) const {
    ARROW_ASSIGN_OR_RAISE(auto match, FindOneOrNone(root));
    if (match.empty()) {
      return static_cast<GetType<T>>(NULLPTR);
    }
    return match.GetFlattened(root, pool);
  }

 private:
  void Flatten(std::vector<FieldRef> children);

  std::variant<FieldPath, std::string, std::vector<FieldRef>> impl_;
};

ARROW_EXPORT void PrintTo(const FieldRef& ref, std::ostream* os);

ARROW_EXPORT
std::ostream& operator<<(std::ostream& os, const FieldRef&);

// ----------------------------------------------------------------------
// Schema

enum class Endianness {
  Little = 0,
  Big = 1,
#if ARROW_LITTLE_ENDIAN
  Native = Little
#else
  Native = Big
#endif
};

/// \class Schema
/// \brief Sequence of arrow::Field objects describing the columns of a record
/// batch or table data structure
// 在 Apache Arrow 中，Schema 类是表格数据结构的“蓝图”。如果把 Table 或 RecordBatch 比作一张数据库表，那么 Schema 就是这张表的表结构定义。
// Schema 的核心作用是定义数据列的逻辑视图。它并不包含任何实际的数据（如整数或字符串），而是描述了：
// 列的顺序与组成：由一系列 Field 对象组成的有序列表。
// 类型信息：每一列的名称、逻辑数据类型（DataType）以及是否允许为空。
// 全局元数据：整个数据集层面的键值对信息（如创建者、数据描述）。
// 字节序 (Endianness)：规定了数据在底层存储时的字节顺序，确保跨平台时的正确解析。
class ARROW_EXPORT Schema : public detail::Fingerprintable,
                            public util::EqualityComparable<Schema>,
                            public util::ToStringOstreamable<Schema> {
 public:
  // 全功能构造函数。允许指定字段列表、字节序以及全局元数据。
  explicit Schema(FieldVector fields, Endianness endianness,
                  std::shared_ptr<const KeyValueMetadata> metadata = NULLPTR);
  // 常用构造函数。默认使用平台原生的字节序（Native Endian）。
  explicit Schema(FieldVector fields,
                  std::shared_ptr<const KeyValueMetadata> metadata = NULLPTR);

  Schema(const Schema&);

  ~Schema() override;

  /// Returns true if all of the schema fields are equal
  bool Equals(const Schema& other, bool check_metadata = false) const;
  bool Equals(const std::shared_ptr<Schema>& other, bool check_metadata = false) const;

  /// \brief Set endianness in the schema
  ///
  /// \return new Schema
  std::shared_ptr<Schema> WithEndianness(Endianness endianness) const;

  /// \brief Return endianness in the schema
  // 获取该 Schema 定义的字节序，并判断是否与当前运行环境的系统字节序一致。
  Endianness endianness() const;

  /// \brief Indicate if endianness is equal to platform-native endianness
  bool is_native_endian() const;

  /// \brief Return the number of fields (columns) in the schema
  // 返回 Schema 中包含的列（字段）总数。
  int num_fields() const;

  /// Return the ith schema element. Does not boundscheck
  // 通过索引 i 获取对应的 Field 对象。
  const std::shared_ptr<Field>& field(int i) const;
  // 返回包含所有 Field 对象的向量（FieldVector）。
  const FieldVector& fields() const;
  // 提取所有列名，返回一个 std::vector<std::string>。
  std::vector<std::string> field_names() const;

  /// Returns null if name not found
  std::shared_ptr<Field> GetFieldByName(std::string_view name) const;

  /// \brief Return the indices of all fields having this name in sorted order
  FieldVector GetAllFieldsByName(std::string_view name) const;

  /// Returns -1 if name not found
  int GetFieldIndex(std::string_view name) const;

  /// Return the indices of all fields having this name
  std::vector<int> GetAllFieldIndices(std::string_view name) const;

  /// Indicate if field named `name` can be found unambiguously in the schema.
  Status CanReferenceFieldByName(std::string_view name) const;

  /// Indicate if fields named `names` can be found unambiguously in the schema.
  Status CanReferenceFieldsByNames(const std::vector<std::string>& names) const;

  /// \brief The custom key-value metadata, if any
  ///
  /// \return metadata may be null
  const std::shared_ptr<const KeyValueMetadata>& metadata() const;

  /// \brief Render a string representation of the schema suitable for debugging
  /// \param[in] show_metadata when true, if KeyValueMetadata is non-empty,
  /// print keys and values in the output
  std::string ToString(bool show_metadata = false) const;

  Result<std::shared_ptr<Schema>> AddField(int i,
                                           const std::shared_ptr<Field>& field) const;
  Result<std::shared_ptr<Schema>> RemoveField(int i) const;
  Result<std::shared_ptr<Schema>> SetField(int i,
                                           const std::shared_ptr<Field>& field) const;

  /// \brief Replace field names with new names
  ///
  /// \param[in] names new names
  /// \return new Schema
  Result<std::shared_ptr<Schema>> WithNames(const std::vector<std::string>& names) const;

  /// \brief Replace key-value metadata with new metadata
  ///
  /// \param[in] metadata new KeyValueMetadata
  /// \return new Schema
  std::shared_ptr<Schema> WithMetadata(
      const std::shared_ptr<const KeyValueMetadata>& metadata) const;

  /// \brief Return copy of Schema without the KeyValueMetadata
  std::shared_ptr<Schema> RemoveMetadata() const;

  /// \brief Indicate that the Schema has non-empty KevValueMetadata
  bool HasMetadata() const;

  /// \brief Indicate that the Schema has distinct field names.
  bool HasDistinctFieldNames() const;

 protected:
  std::string ComputeFingerprint() const override;
  std::string ComputeMetadataFingerprint() const override;

 private:
  class Impl;
  // 指向实现类（Pimpl 模式）。
  // Arrow 使用 Pimpl 模式来隐藏具体实现细节，减少头文件依赖，提高编译速度，并保证 ABI 的稳定性。它实际存储了 fields 列表、metadata 指针等数据。
  std::unique_ptr<Impl> impl_;
};

ARROW_EXPORT void PrintTo(const Schema& s, std::ostream* os);

ARROW_EXPORT
std::string EndiannessToString(Endianness endianness);

// ----------------------------------------------------------------------

/// \brief Convenience class to incrementally construct/merge schemas.
///
/// This class amortizes the cost of validating field name conflicts by
/// maintaining the mapping. The caller also controls the conflict resolution
/// scheme.
class ARROW_EXPORT SchemaBuilder {
 public:
  // Indicate how field conflict(s) should be resolved when building a schema. A
  // conflict arise when a field is added to the builder and one or more field(s)
  // with the same name already exists.
  enum ConflictPolicy {
    // Ignore the conflict and append the field. This is the default behavior of the
    // Schema constructor and the `arrow::schema` factory function.
    CONFLICT_APPEND = 0,
    // Keep the existing field and ignore the newer one.
    CONFLICT_IGNORE,
    // Replace the existing field with the newer one.
    CONFLICT_REPLACE,
    // Merge the fields. The merging behavior can be controlled by `Field::MergeOptions`
    // specified at construction time. Also see documentation of `Field::MergeWith`.
    CONFLICT_MERGE,
    // Refuse the new field and error out.
    CONFLICT_ERROR
  };

  /// \brief Construct an empty SchemaBuilder
  /// `field_merge_options` is only effective when `conflict_policy` == `CONFLICT_MERGE`.
  SchemaBuilder(
      ConflictPolicy conflict_policy = CONFLICT_APPEND,
      Field::MergeOptions field_merge_options = Field::MergeOptions::Defaults());
  /// \brief Construct a SchemaBuilder from a list of fields
  /// `field_merge_options` is only effective when `conflict_policy` == `CONFLICT_MERGE`.
  SchemaBuilder(
      FieldVector fields, ConflictPolicy conflict_policy = CONFLICT_APPEND,
      Field::MergeOptions field_merge_options = Field::MergeOptions::Defaults());
  /// \brief Construct a SchemaBuilder from a schema, preserving the metadata
  /// `field_merge_options` is only effective when `conflict_policy` == `CONFLICT_MERGE`.
  SchemaBuilder(
      const std::shared_ptr<Schema>& schema,
      ConflictPolicy conflict_policy = CONFLICT_APPEND,
      Field::MergeOptions field_merge_options = Field::MergeOptions::Defaults());

  /// \brief Return the conflict resolution method.
  ConflictPolicy policy() const;

  /// \brief Set the conflict resolution method.
  void SetPolicy(ConflictPolicy resolution);

  /// \brief Add a field to the constructed schema.
  ///
  /// \param[in] field to add to the constructed Schema.
  /// \return A failure if encountered.
  Status AddField(const std::shared_ptr<Field>& field);

  /// \brief Add multiple fields to the constructed schema.
  ///
  /// \param[in] fields to add to the constructed Schema.
  /// \return The first failure encountered, if any.
  Status AddFields(const FieldVector& fields);

  /// \brief Add fields of a Schema to the constructed Schema.
  ///
  /// \param[in] schema to take fields to add to the constructed Schema.
  /// \return The first failure encountered, if any.
  Status AddSchema(const std::shared_ptr<Schema>& schema);

  /// \brief Add fields of multiple Schemas to the constructed Schema.
  ///
  /// \param[in] schemas to take fields to add to the constructed Schema.
  /// \return The first failure encountered, if any.
  Status AddSchemas(const std::vector<std::shared_ptr<Schema>>& schemas);

  Status AddMetadata(const KeyValueMetadata& metadata);

  /// \brief Return the constructed Schema.
  ///
  /// The builder internal state is not affected by invoking this method, i.e.
  /// a single builder can yield multiple incrementally constructed schemas.
  ///
  /// \return the constructed schema.
  Result<std::shared_ptr<Schema>> Finish() const;

  /// \brief Merge schemas in a unified schema according to policy.
  static Result<std::shared_ptr<Schema>> Merge(
      const std::vector<std::shared_ptr<Schema>>& schemas,
      ConflictPolicy policy = CONFLICT_MERGE);

  /// \brief Indicate if schemas are compatible to merge according to policy.
  static Status AreCompatible(const std::vector<std::shared_ptr<Schema>>& schemas,
                              ConflictPolicy policy = CONFLICT_MERGE);

  /// \brief Reset internal state with an empty schema (and metadata).
  void Reset();

  ~SchemaBuilder();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;

  Status AppendField(const std::shared_ptr<Field>& field);
};

/// \brief Unifies schemas by merging fields by name.
///
/// The behavior of field merging can be controlled via `Field::MergeOptions`.
///
/// The resulting schema will contain the union of fields from all schemas.
/// Fields with the same name will be merged. See `Field::MergeOptions`.
/// - They are expected to be mergeable under provided `field_merge_options`.
/// - The unified field will inherit the metadata from the schema where
///   that field is first defined.
/// - The first N fields in the schema will be ordered the same as the
///   N fields in the first schema.
/// The resulting schema will inherit its metadata from the first input schema.
/// Returns an error if:
/// - Any input schema contains fields with duplicate names.
/// - Fields of the same name are not mergeable.
ARROW_EXPORT
Result<std::shared_ptr<Schema>> UnifySchemas(
    const std::vector<std::shared_ptr<Schema>>& schemas,
    Field::MergeOptions field_merge_options = Field::MergeOptions::Defaults());

namespace internal {

constexpr bool may_have_validity_bitmap(Type::type id) {
  switch (id) {
    case Type::NA:
    case Type::DENSE_UNION:
    case Type::SPARSE_UNION:
    case Type::RUN_END_ENCODED:
      return false;
    default:
      return true;
  }
}

constexpr bool has_variadic_buffers(Type::type id) {
  switch (id) {
    case Type::BINARY_VIEW:
    case Type::STRING_VIEW:
      return true;
    default:
      return false;
  }
}

ARROW_DEPRECATED("Deprecated in 17.0.0. Use may_have_validity_bitmap() instead.")
constexpr bool HasValidityBitmap(Type::type id) { return may_have_validity_bitmap(id); }

ARROW_EXPORT
std::string ToString(Type::type id);

ARROW_EXPORT
std::string ToTypeName(Type::type id);

ARROW_EXPORT
std::string ToString(TimeUnit::type unit);

}  // namespace internal

// Helpers to get instances of data types based on general categories

/// \brief Signed integer types
ARROW_EXPORT
const std::vector<std::shared_ptr<DataType>>& SignedIntTypes();
/// \brief Unsigned integer types
ARROW_EXPORT
const std::vector<std::shared_ptr<DataType>>& UnsignedIntTypes();
/// \brief Signed and unsigned integer types
ARROW_EXPORT
const std::vector<std::shared_ptr<DataType>>& IntTypes();
/// \brief Floating point types
ARROW_EXPORT
const std::vector<std::shared_ptr<DataType>>& FloatingPointTypes();
/// \brief Number types without boolean - integer and floating point types
ARROW_EXPORT
const std::vector<std::shared_ptr<DataType>>& NumericTypes();
/// \brief Binary and string-like types (except fixed-size binary)
ARROW_EXPORT
const std::vector<std::shared_ptr<DataType>>& BaseBinaryTypes();
/// \brief Binary and large-binary types
ARROW_EXPORT
const std::vector<std::shared_ptr<DataType>>& BinaryTypes();
/// \brief String and large-string types
ARROW_EXPORT
const std::vector<std::shared_ptr<DataType>>& StringTypes();
/// \brief String-view and Binary-view
ARROW_EXPORT
const std::vector<std::shared_ptr<DataType>>& BinaryViewTypes();
/// \brief Temporal types including date, time and timestamps for each unit
ARROW_EXPORT
const std::vector<std::shared_ptr<DataType>>& TemporalTypes();
/// \brief Interval types
ARROW_EXPORT
const std::vector<std::shared_ptr<DataType>>& IntervalTypes();
/// \brief Duration types for each unit
ARROW_EXPORT
const std::vector<std::shared_ptr<DataType>>& DurationTypes();
/// \brief Numeric, base binary, date, boolean and null types
ARROW_EXPORT
const std::vector<std::shared_ptr<DataType>>& PrimitiveTypes();

/// \brief Decimal type ids
ARROW_EXPORT
const std::vector<Type::type>& DecimalTypeIds();

/// \brief Create a data type instance from a type ID for parameter-free types
///
/// This function creates a data type instance for types that don't require
/// additional parameters (where TypeTraits<T>::is_parameter_free is true).
/// For types that require parameters (like TimestampType or ListType),
/// this function will return an error.
///
/// \param[in] id The type ID to create a type instance for
/// \return The type instance for the given type ID,
///         or a TypeError if the type requires parameters
ARROW_EXPORT
Result<std::shared_ptr<DataType>> type_singleton(Type::type id);

}  // namespace arrow
