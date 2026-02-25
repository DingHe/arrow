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

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "arrow/buffer.h"
#include "arrow/compute/exec.h"
#include "arrow/datum.h"
#include "arrow/device_allocation_type_set.h"
#include "arrow/memory_pool.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/type.h"
#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"

// macOS defines PREALLOCATE as a preprocessor macro in the header sys/vnode.h.
// No other BSD seems to do so. The name is used as an identifier in MemAllocation enum.
#if defined(__APPLE__) && defined(PREALLOCATE)
#  undef PREALLOCATE
#endif

namespace arrow {
namespace compute {

class FunctionOptions;

/// \brief Base class for opaque kernel-specific state. For example, if there
/// is some kind of initialization required.
struct ARROW_EXPORT KernelState {
  virtual ~KernelState() = default;
};

/// \brief Context/state for the execution of a particular kernel.
// 在 Apache Arrow 的计算向量化引擎（Compute Engine）中，KernelContext 是一个非常核心的“管家”类。它负责在执行具体的计算函数（即 Kernel）时，提供必要的运行时环境
// KernelContext 的主要作用是隔离与封装计算过程中的状态和资源。
// 当 Arrow 执行一个计算任务（如：两个数组相加、字符串过滤等）时，计算逻辑（Kernel）本身应该是尽量纯粹的。而计算过程中需要的内存分配、配置信息以及中间状态，都由 KernelContext 统一管理。
// 资源接入点：为 Kernel 提供分配内存（Buffer）的统一接口。
// 状态承载：承载 Kernel 运行时的私有状态（如：正则表达式的编译结果、聚合函数的中间累加值）。
// 上下文共享：连接更高级别的执行上下文（ExecContext），从而获取全局配置（如线程池、内存池）。
class ARROW_EXPORT KernelContext {
 public:
  // Can pass optional backreference; not used consistently for the
  // moment but will be made so in the future
  explicit KernelContext(ExecContext* exec_ctx, const Kernel* kernel = NULLPTR)
      : exec_ctx_(exec_ctx), kernel_(kernel) {}

  /// \brief Allocate buffer from the context's memory pool. The contents are
  /// not initialized.
  // 从内存池中分配指定字节大小的缓冲区。
  Result<std::shared_ptr<ResizableBuffer>> Allocate(int64_t nbytes);

  /// \brief Allocate buffer for bitmap from the context's memory pool. Like
  /// Allocate, the contents of the buffer are not initialized but the last
  /// byte is preemptively zeroed to help avoid ASAN or valgrind issues.
  // 专门为位图（Validity Bitmap，有效性位图）分配内存
  // Arrow 中的 Null 值是通过位图表示的。这个方法会根据比特数计算所需的字节数，并且会预先将最后一个字节清零。
  // 这是为了防止 ASAN（内存检查工具）或 Valgrind 在读取最后几个比特（可能跨越字节边界）时报出“读取未初始化内存”的错误
  Result<std::shared_ptr<ResizableBuffer>> AllocateBitmap(int64_t num_bits);

  /// \brief Assign the active KernelState to be utilized for each stage of
  /// kernel execution. Ownership and memory lifetime of the KernelState must
  /// be minded separately.
 // 绑定运行时的状态对象。
  void SetState(KernelState* state) { state_ = state; }

  // Set kernel that is being invoked since some kernel
  // implementations will examine the kernel state.
  // 在运行时动态设置或更改关联的 Kernel 定义。
  void SetKernel(const Kernel* kernel) { kernel_ = kernel; }
  // 获取当前的私有状态指针。通常需要转换（Cast）为具体的子类状态使用。
  KernelState* state() { return state_; }

  /// \brief Configuration related to function execution that is to be shared
  /// across multiple kernels.
  // 获取关联的 ExecContext 指针。
  ExecContext* exec_context() { return exec_ctx_; }

  /// \brief The memory pool to use for allocations. For now, it uses the
  /// MemoryPool contained in the ExecContext used to create the KernelContext.
  // 快捷方法，直接获取 ExecContext 中配置的 MemoryPool
  MemoryPool* memory_pool() { return exec_ctx_->memory_pool(); }

  const Kernel* kernel() const { return kernel_; }

 private:
  // 指向全局执行上下文的指针。
  // ExecContext 包含了比单个 Kernel 更高层级的配置，例如当前使用的 MemoryPool（内存池）和 CPU 调度器。KernelContext 通过它与外部环境通信。
  ExecContext* exec_ctx_;
  // 指向当前 Kernel 运行时的私有状态。
  // 某些 Kernel 是有状态的。例如，一个“去重计算”可能需要维护一个哈希表。这个状态就存储在 KernelState 中。
  KernelState* state_ = NULLPTR;
  // 指向当前正在执行的 Kernel 定义的指针。
  const Kernel* kernel_ = NULLPTR;
};

/// \brief An type-checking interface to permit customizable validation rules
/// for use with InputType and KernelSignature. This is for scenarios where the
/// acceptance is not an exact type instance, such as a TIMESTAMP type for a
/// specific TimeUnit, but permitting any time zone.
// TypeMatcher 是一个非常关键的策略接口，它允许在函数分发（Function Dispatch）和内核签名（Kernel Signature）匹配过程中使用非精确的类型匹配规则。
// TypeMatcher 的核心作用是实现模糊或参数化的类型验证。
// 在常规情况下，Arrow 的函数内核（Kernel）通常要求精确的类型匹配（例如：Int32 必须对应 Int32）。但在许多实际场景中，这种限制过于死板
// 时区无关性：一个处理时间戳（Timestamp）的内核，可能只要求精度是“秒”，而不关心具体的时区（Timezone）。
// 宽度泛化：某些内核可能可以处理任何位宽的整数。
// 自定义逻辑：用户可能需要实现特定的逻辑，比如“匹配所有属于嵌套类型的列表”。
// TypeMatcher 提供了一个抽象层，使得 InputType 和 KernelSignature 不再仅仅依赖具体的 DataType 实例，而是可以包含一段“匹配逻辑”。
struct ARROW_EXPORT TypeMatcher {
  virtual ~TypeMatcher() = default;

  /// \brief Return true if this matcher accepts the data type.
  // 判断给定的 DataType 是否符合匹配规则。
  // 如果接受该类型，返回 true；否则返回 false。
  virtual bool Matches(const DataType& type) const = 0;

  /// \brief A human-interpretable string representation of what the type
  /// matcher checks for, usable when printing KernelSignature or formatting
  /// error messages.
  // 生成该匹配规则的可读描述。
  virtual std::string ToString() const = 0;

  /// \brief Return true if this TypeMatcher contains the same matching rule as
  /// the other. Currently depends on RTTI.
  // 判断两个 TypeMatcher 对象是否代表相同的匹配规则。
  virtual bool Equals(const TypeMatcher& other) const = 0;
};

namespace match {

/// \brief Match any DataType instance having the same DataType::id.
ARROW_EXPORT std::shared_ptr<TypeMatcher> SameTypeId(Type::type type_id);

/// \brief Match any TimestampType instance having the same unit, but the time
/// zones can be different.
ARROW_EXPORT std::shared_ptr<TypeMatcher> TimestampTypeUnit(TimeUnit::type unit);
ARROW_EXPORT std::shared_ptr<TypeMatcher> Time32TypeUnit(TimeUnit::type unit);
ARROW_EXPORT std::shared_ptr<TypeMatcher> Time64TypeUnit(TimeUnit::type unit);
ARROW_EXPORT std::shared_ptr<TypeMatcher> DurationTypeUnit(TimeUnit::type unit);

// \brief Match any integer type
ARROW_EXPORT std::shared_ptr<TypeMatcher> Integer();

// Match types using 32-bit varbinary representation
ARROW_EXPORT std::shared_ptr<TypeMatcher> BinaryLike();

// Match types using 64-bit varbinary representation
ARROW_EXPORT std::shared_ptr<TypeMatcher> LargeBinaryLike();

// Match any fixed binary type
ARROW_EXPORT std::shared_ptr<TypeMatcher> FixedSizeBinaryLike();

// \brief Match any primitive type (boolean or any type representable as a C
// Type)
ARROW_EXPORT std::shared_ptr<TypeMatcher> Primitive();

// \brief Match any integer type that can be used as run-end in run-end encoded
// arrays
ARROW_EXPORT std::shared_ptr<TypeMatcher> RunEndInteger();

/// \brief Match run-end encoded types that use any valid run-end type and
/// encode specific value types
///
/// @param[in] value_type_matcher a matcher that is applied to the values field
ARROW_EXPORT std::shared_ptr<TypeMatcher> RunEndEncoded(
    std::shared_ptr<TypeMatcher> value_type_matcher);

/// \brief Match run-end encoded types that use any valid run-end type and
/// encode specific value types
///
/// @param[in] value_type_id a type id that the type of the values field should match
ARROW_EXPORT std::shared_ptr<TypeMatcher> RunEndEncoded(Type::type value_type_id);

/// \brief Match run-end encoded types that encode specific run-end and value types
///
/// @param[in] run_end_type_matcher a matcher that is applied to the run_ends field
/// @param[in] value_type_matcher a matcher that is applied to the values field
ARROW_EXPORT std::shared_ptr<TypeMatcher> RunEndEncoded(
    std::shared_ptr<TypeMatcher> run_end_type_matcher,
    std::shared_ptr<TypeMatcher> value_type_matcher);

}  // namespace match

/// \brief An object used for type-checking arguments to be passed to a kernel
/// and stored in a KernelSignature. The type-checking rule can be supplied
/// either with an exact DataType instance or a custom TypeMatcher.
// InputType 是函数分发系统（Function Dispatch System）的基石类。它定义了计算函数（Kernel）对输入参数的类型约束规则
// InputType 的核心作用是参数类型校验。
// 当你在 Arrow 中调用一个函数（如 add(array1, array2)）时，引擎需要找到一个合适的“内核”（Kernel）来执行计算。InputType 就负责描述一个内核所能接受的参数特征：
// 定义约束：它既可以要求参数必须是完全一致的类型（如 Int32），也可以定义灵活的匹配规则（如“任何单位的时间戳”）。
// 内核签名（KernelSignature）的组成部分：一个内核通常有多个输入，每个输入都对应一个 InputType，共同构成了该内核的身份标识。
// 路由决策：在运行时，它负责判断传入的实际数据（Datum 或 DataType）是否满足该内核的要求。
class ARROW_EXPORT InputType {
 public:
  /// \brief The kind of type-checking rule that the InputType contains.
  // 匹配规则类型
  // 它决定了 InputType 当前处于哪种工作模式：
  enum Kind {
    /// \brief Accept any value type.
    ANY_TYPE, // 万能匹配，接受任何 Arrow 类型。

    /// \brief A fixed arrow::DataType and will only exact match having this
    /// exact type (e.g. same TimestampType unit, same decimal scale and
    /// precision, or same nested child types).
    EXACT_TYPE, // 精确匹配，要求物理和逻辑元数据完全一致（例如：同为 Decimal128 且精度和刻度必须相同）。

    /// \brief Uses a TypeMatcher implementation to check the type.
    USE_TYPE_MATCHER // 委托匹配，使用用户自定义的 TypeMatcher 逻辑进行复杂校验。
  };

  /// \brief Accept any value type
  // 默认构造函数，创建一个 ANY_TYPE 模式的匹配器。
  InputType() : kind_(ANY_TYPE) {}

  /// \brief Accept an exact value type.
  // 精确类型构造。例如传入 int32()，则只匹配 Int32。
  InputType(std::shared_ptr<DataType> type)  // NOLINT implicit construction
      : kind_(EXACT_TYPE), type_(std::move(type)) {}

  /// \brief Use the passed TypeMatcher to type check.
  // 使用自定义匹配逻辑构造。
  InputType(std::shared_ptr<TypeMatcher> type_matcher)  // NOLINT implicit construction
      : kind_(USE_TYPE_MATCHER), type_matcher_(std::move(type_matcher)) {}

  /// \brief Match any type with the given Type::type. Uses a TypeMatcher for
  /// its implementation.
  // 通过 TypeID 快速匹配，底层会自动创建一个 SameTypeId 的匹配器（属于 USE_TYPE_MATCHER 模式）
  InputType(Type::type type_id)  // NOLINT implicit construction
      : InputType(match::SameTypeId(type_id)) {}

  InputType(const InputType& other) { CopyInto(other); }

  void operator=(const InputType& other) { CopyInto(other); }

  InputType(InputType&& other) { MoveInto(std::forward<InputType>(other)); }

  void operator=(InputType&& other) { MoveInto(std::forward<InputType>(other)); }

  // \brief Match any input (array, scalar of any type)
  // 显式返回一个匹配任何类型的实例
  static InputType Any() { return InputType(); }

  /// \brief Return true if this input type matches the same type cases as the
  /// other.
  bool Equals(const InputType& other) const;

  bool operator==(const InputType& other) const { return this->Equals(other); }

  bool operator!=(const InputType& other) const { return !(*this == other); }

  /// \brief Return hash code.
  size_t Hash() const;

  /// \brief Render a human-readable string representation.
  std::string ToString() const;

  /// \brief Return true if the Datum matches this argument kind in
  /// type (and only allows scalar or array-like Datums).
  // 判断传入的数据（数组或标量）是否符合要求。它会先提取 Datum 的类型，然后调用下面的 Matches(DataType)。
  bool Matches(const Datum& value) const;

  /// \brief Return true if the type matches this InputType
  // 判断给定的元数据类型是否符合规则。
  bool Matches(const DataType& type) const;

  /// \brief The type matching rule that this InputType uses.
  Kind kind() const { return kind_; }

  /// \brief For InputType::EXACT_TYPE kind, the exact type that this InputType
  /// must match. Otherwise this function should not be used and will assert in
  /// debug builds.
  const std::shared_ptr<DataType>& type() const;

  /// \brief For InputType::USE_TYPE_MATCHER, the TypeMatcher to be used for
  /// checking the type of a value. Otherwise this function should not be used
  /// and will assert in debug builds.
  const TypeMatcher& type_matcher() const;

 private:
  void CopyInto(const InputType& other) {
    this->kind_ = other.kind_;
    this->type_ = other.type_;
    this->type_matcher_ = other.type_matcher_;
  }

  void MoveInto(InputType&& other) {
    this->kind_ = other.kind_;
    this->type_ = std::move(other.type_);
    this->type_matcher_ = std::move(other.type_matcher_);
  }

  Kind kind_;

  // For EXACT_TYPE Kind
  std::shared_ptr<DataType> type_;

  // For USE_TYPE_MATCHER Kind
  std::shared_ptr<TypeMatcher> type_matcher_;
};

/// \brief Container to capture both exact and input-dependent output types.
// 在 Apache Arrow 的计算向量化引擎中，OutputType 是用于推导计算结果类型的关键类。它与 InputType 呼应，共同定义了内核（Kernel）的完整签名。
// OutputType 的核心作用是确定计算函数执行后的返回数据类型。
// 在静态类型语言中，函数的返回类型通常是固定的。但在 Arrow 这种处理多种动态列类型的系统中，情况要复杂得多：
// 固定类型：例如 equal(int32, int32) 永远返回 bool 类型。
// 依赖输入类型：例如 add(int32, int32) 返回 int32，但 add(float64, float64) 则返回 float64。
// 复杂推导：例如 cast 函数，其输出类型取决于用户提供的 CastOptions；或者 if_else 函数，其输出类型必须是两个分支输入类型的“公共父类型”（Common Supertype）。
class ARROW_EXPORT OutputType {
 public:
  /// \brief An enum indicating whether the value type is an invariant fixed
  /// value or one that's computed by a kernel-defined resolver function.
  // 定义了该输出类型是如何确定的：
  // FIXED：固定模式。输出类型在内核注册时就已经确定，不随输入变化。
  // COMPUTED：计算模式。输出类型由一个特殊的函数（Resolver）在运行时根据输入参数动态计算。
  enum ResolveKind { FIXED, COMPUTED };

  /// Type resolution function. Given input types, return output type.  This
  /// function MAY may use the kernel state to decide the output type based on
  /// the FunctionOptions.
  ///
  /// This function SHOULD _not_ be used to check for arity, that is to be
  /// performed one or more layers above.
  // 解析器函数签名
  using Resolver =
      std::function<Result<TypeHolder>(KernelContext*, const std::vector<TypeHolder>&)>;

  /// \brief Output an exact type
  OutputType(std::shared_ptr<DataType> type)  // NOLINT implicit construction
      : kind_(FIXED), type_(std::move(type)) {}

  /// \brief Output a computed type depending on actual input types
  template <typename Fn>
  OutputType(Fn resolver)  // NOLINT implicit construction
      : kind_(COMPUTED), resolver_(std::move(resolver)) {}

  OutputType(const OutputType& other) {
    this->kind_ = other.kind_;
    this->type_ = other.type_;
    this->resolver_ = other.resolver_;
  }

  OutputType(OutputType&& other) {
    this->kind_ = other.kind_;
    this->type_ = std::move(other.type_);
    this->resolver_ = other.resolver_;
  }

  OutputType& operator=(const OutputType&) = default;
  OutputType& operator=(OutputType&&) = default;

  /// \brief Return the type of the expected output value of the kernel given
  /// the input argument types. The resolver may make use of state information
  /// kept in the KernelContext.
  Result<TypeHolder> Resolve(KernelContext* ctx,
                             const std::vector<TypeHolder>& args) const;

  /// \brief The exact output value type for the FIXED kind.
  const std::shared_ptr<DataType>& type() const;

  /// \brief For use with COMPUTED resolution strategy. It may be more
  /// convenient to invoke this with OutputType::Resolve returned from this
  /// method.
  const Resolver& resolver() const;

  /// \brief Render a human-readable string representation.
  std::string ToString() const;

  /// \brief Return the kind of type resolution of this output type, whether
  /// fixed/invariant or computed by a resolver.
  ResolveKind kind() const { return kind_; }

 private:
  ResolveKind kind_;

  // For FIXED resolution
  std::shared_ptr<DataType> type_;

  // For COMPUTED resolution
  Resolver resolver_ = NULLPTR;
};

/// \brief Additional constraints to apply to the input types of a kernel when matching a
/// specific kernel signature.
// MatchConstraint 提供了一种跨参数（Cross-argument）验证的机制。
// 在通常的内核签名匹配中，InputType 只能单独验证每一个参数。但在某些复杂的函数中，仅仅检查单个参数是不够的，我们需要检查参数之间的关系。例如：
// 类型一致性约束：要求两个输入参数的类型必须完全相同（例如：Add 算子要求两个输入必须同为 Int32 或同为 Float64，不能一左一右混合）。
// 属性依赖约束：要求所有输入的时间戳具有相同的单位（TimeUnit）。
// 参数数量逻辑：虽然元数（Arity）通常由其他层处理，但约束器可以根据参数列表的整体特征进行二次检查。
class ARROW_EXPORT MatchConstraint {
 public:
  virtual ~MatchConstraint() = default;

  /// \brief Return true if the input types satisfy the constraint.
  // 执行具体的约束逻辑断言。
  virtual bool Matches(const std::vector<TypeHolder>& types) const = 0;

  /// \brief Convenience function to create a MatchConstraint from a match function.
  // 用于快速创建约束对象。
  static std::shared_ptr<MatchConstraint> Make(
      std::function<bool(const std::vector<TypeHolder>&)> matches);
};

/// \brief Constraint that all input types are decimal types and have the same scale.
ARROW_EXPORT std::shared_ptr<MatchConstraint> DecimalsHaveSameScale();

/// \brief Holds the input types, optional match constraint and output type of the kernel.
///
/// VarArgs functions with minimum N arguments should pass up to N input types to be
/// used to validate the input types of a function invocation. The first N-1 types
/// will be matched against the first N-1 arguments, and the last type will be
/// matched against the remaining arguments.
// 在 Apache Arrow 的计算引擎中，KernelSignature 类是内核（Kernel）的“身份证”。它将我们之前讨论过的 InputType、OutputType 和 MatchConstraint 整合在一起，构成了完整的函数匹配契约。
// 该类的核心作用是描述和验证内核的类型规则，具体体现在：
// 定义契约：它明确了一个计算内核需要什么样的输入（in_types），并承诺会产生什么样的输出（out_type）。
// 支持多态分发：Arrow 的同一个函数（如 add）可能有多个内核（如 int32_add、float64_add）。引擎通过比对 KernelSignature 来决定具体调用哪一个内核。
// 处理变长参数（VarArgs）：它定义了变长参数函数的匹配规则（例如：前 $N-1$ 个参数按类型匹配，剩余所有参数按最后一个类型匹配）。
// 性能优化（哈希与缓存）：由于函数调用极其频繁，该类通过计算哈希值来支持在高速查找表（Hash Map）中快速定位内核。
class ARROW_EXPORT KernelSignature {
 public:
  KernelSignature(std::vector<InputType> in_types, OutputType out_type,
                  bool is_varargs = false,
                  std::shared_ptr<MatchConstraint> constraint = NULLPTR);

  /// \brief Convenience ctor since make_shared can be awkward
  static std::shared_ptr<KernelSignature> Make(
      std::vector<InputType> in_types, OutputType out_type, bool is_varargs = false,
      std::shared_ptr<MatchConstraint> constraint = NULLPTR);

  /// \brief Return true if the signature is compatible with the list of input
  /// value descriptors and satisfies the match constraint, if any.
  bool MatchesInputs(const std::vector<TypeHolder>& types) const;

  /// \brief Returns true if the input types of each signature are
  /// equal. Well-formed functions should have a deterministic output type
  /// given input types, but currently it is the responsibility of the
  /// developer to ensure this.
  bool Equals(const KernelSignature& other) const;

  bool operator==(const KernelSignature& other) const { return this->Equals(other); }

  bool operator!=(const KernelSignature& other) const { return !(*this == other); }

  /// \brief Compute a hash code for the signature
  size_t Hash() const;

  /// \brief The input types for the kernel. For VarArgs functions, this should
  /// generally contain a single validator to use for validating all of the
  /// function arguments.
  const std::vector<InputType>& in_types() const { return in_types_; }

  /// \brief The output type for the kernel. Use Resolve to return the
  /// exact output given input argument types, since many kernels'
  /// output types depend on their input types (or their type
  /// metadata).
  const OutputType& out_type() const { return out_type_; }

  /// \brief Render a human-readable string representation
  std::string ToString() const;

  bool is_varargs() const { return is_varargs_; }

 private:
  // 输入参数的类型约束列表。
  // 存储了每一个输入参数必须满足的 InputType。如果是普通函数，列表长度通常等于参数个数；如果是变长参数函数，则定义了匹配的基准。
  std::vector<InputType> in_types_;
  // 输出类型的推导规则。
  // 可以是固定类型，也可以是根据输入动态计算的解析器。
  OutputType out_type_;
  // 标识该内核是否接受变长参数。
  bool is_varargs_;
  // 可选的额外约束。
  // 用于处理跨参数的复杂校验（如：要求两个参数的单位必须一致）
  std::shared_ptr<MatchConstraint> constraint_;

  // For caching the hash code after it's computed the first time
  // 缓存哈希值。
  mutable uint64_t hash_code_;
};

/// \brief A function may contain multiple variants of a kernel for a given
/// type combination for different SIMD levels. Based on the active system's
/// CPU info or the user's preferences, we can elect to use one over the other.
// SimdLevel 是一个用于硬件加速优化的枚举结构。
// 它定义了 CPU 不同的单指令多数据（SIMD）扩展指令集级别。
// 该类的核心作用是实现运行时多态加速（Runtime Dispatching）。
// 现代 CPU 通常支持各种加速指令集（如 Intel 的 AVX 或 ARM 的 NEON），这些指令集可以并行处理多个数据，极大地提升计算吞吐量。
// 多版本共存：同一个计算函数（如 Add）在 Arrow 中可能有多个版本的内核实现：一个普通的 C++ 版本，一个 SSE4.2 优化版本，以及一个 AVX2 优化版本。
// 自适应选择：在程序运行时，Arrow 会检测当前执行环境的 CPU 硬件信息。如果 CPU 支持 AVX512，系统会自动选择效率最高的 AVX512 内核；如果硬件不支持，则降级（Fallback）到基础版本。
// 性能调优：它允许开发者根据目标平台的硬件特性，精细化地提供针对性优化代码。
struct SimdLevel {
  enum type { NONE = 0, SSE4_2, AVX, AVX2, AVX512, NEON, MAX };
};

/// \brief The strategy to use for propagating or otherwise populating the
/// validity bitmap of a kernel output.
// 在 Apache Arrow 的计算引擎中，NullHandling 枚举类定义了内核（Kernel）在执行过程中如何处理有效性位图（Validity Bitmap），即如何处理数据中的空值（Null）。
struct NullHandling {
  enum type {
    /// Compute the output validity bitmap by intersecting the validity bitmaps
    /// of the arguments using bitwise-and operations. This means that values
    /// in the output are valid/non-null only if the corresponding values in
    /// all input arguments were valid/non-null. Kernel generally need not
    /// touch the bitmap thereafter, but a kernel's exec function is permitted
    /// to alter the bitmap after the null intersection is computed if it needs
    /// to.
    // 位图交集
    // 遵循最常见的 SQL 逻辑：输入有 Null，结果即为 Null。
    INTERSECTION,

    /// Kernel expects a pre-allocated buffer to write the result bitmap
    /// into. The preallocated memory is not zeroed (except for the last byte),
    /// so the kernel should ensure to completely populate the bitmap.
    // 由内核逻辑决定 Null 值，但由框架负责分配内存。
    COMPUTED_PREALLOCATE,

    /// Kernel allocates and sets the validity bitmap of the output.
    // 完全由内核控制位图的生命周期。
    COMPUTED_NO_PREALLOCATE,

    /// Kernel output is never null and a validity bitmap does not need to be
    /// allocated.
    // 告知框架，该函数的输出永远不包含 Null 值。
    OUTPUT_NOT_NULL
  };
};

/// \brief The preference for memory preallocation of fixed-width type outputs
/// in kernel execution.
struct MemAllocation {
  enum type {
    // For data types that support pre-allocation (i.e. fixed-width), the
    // kernel expects to be provided a pre-allocated data buffer to write
    // into. Non-fixed-width types must always allocate their own data
    // buffers. The allocation made for the same length as the execution batch,
    // so vector kernels yielding differently sized output should not use this.
    //
    // It is valid for the data to not be preallocated but the validity bitmap
    // is (or is computed using the intersection/bitwise-and method).
    //
    // For variable-size output types like BinaryType or StringType, or for
    // nested types, this option has no effect.
    PREALLOCATE,

    // The kernel is responsible for allocating its own data buffer for
    // fixed-width type outputs.
    NO_PREALLOCATE
  };
};

struct Kernel;

/// \brief Arguments to pass to an KernelInit function. A struct is used to help
/// avoid API breakage should the arguments passed need to be expanded.
struct KernelInitArgs {
  /// \brief A pointer to the kernel being initialized. The init function may
  /// depend on the kernel's KernelSignature or other data contained there.
  const Kernel* kernel;

  /// \brief The types of the input arguments that the kernel is
  /// about to be executed against.
  const std::vector<TypeHolder>& inputs;

  /// \brief Opaque options specific to this kernel. May be nullptr for functions
  /// that do not require options.
  const FunctionOptions* options;
};

/// \brief Common initializer function for all kernel types.
// 函数包装器，定义了内核初始化的标准接口
// 在执行内核之前，框架会调用这个初始化函数，根据用户传入的配置（FunctionOptions）生成一个 KernelState 对象（如包含配置参数或预分配的查找表）。
using KernelInit = std::function<Result<std::unique_ptr<KernelState>>(
    KernelContext*, const KernelInitArgs&)>;

/// \brief Base type for kernels. Contains the function signature and
/// optionally the state initialization function, along with some common
/// attributes
// 在 Apache Arrow 的计算引擎中，Kernel（内核）类是执行计算逻辑的最小功能单元。它将特定的计算算法与其支持的数据类型、初始化逻辑以及硬件加速级别绑定在一起。
// Kernel 类是一个基础结构体，作为所有特定类型内核（如 ScalarKernel 标量内核、VectorKernel 向量内核、ScalarAggregateKernel 聚合内核）的基类。
// 绑定签名与逻辑：将具体的 C++ 计算函数与它能处理的输入/输出类型（KernelSignature）关联起来。
// 管理运行状态：定义了如何初始化内核运行所需的私有状态（KernelState），例如在字符串匹配时预编译正则表达式。
// 硬件适配：支持针对不同 CPU 指令集（如 AVX2, NEON）提供同一功能的多个实现，以便运行时选择最优解。
// 执行策略配置：告知执行引擎该内核是否支持并行化。
struct ARROW_EXPORT Kernel {
  Kernel() = default;

  Kernel(std::shared_ptr<KernelSignature> sig, KernelInit init)
      : signature(std::move(sig)), init(std::move(init)) {}

  Kernel(std::vector<InputType> in_types, OutputType out_type, KernelInit init)
      : Kernel(KernelSignature::Make(std::move(in_types), std::move(out_type)),
               std::move(init)) {}

  /// \brief The "signature" of the kernel containing the InputType input
  /// argument validators and OutputType output type resolver.
  // 内核的“指纹”。
  // 包含了该内核支持的输入参数类型约束（InputType）和结果类型的推导规则（OutputType）。执行引擎通过比对它来确定是否可以使用这个内核处理当前数据。
  std::shared_ptr<KernelSignature> signature;

  /// \brief Create a new KernelState for invocations of this kernel, e.g. to
  /// set up any options or state relevant for execution.
  // 状态初始化器。
  // 一个回调函数，用于创建 KernelState。如果不涉及任何状态或配置，可以为 NULLPTR。
  KernelInit init;

  /// \brief Create a vector of new KernelState for invocations of this kernel.
  // 批量初始化内核状态。
  // 通常在需要并发准备多个内核实例（或分块执行）时被调用，确保所有状态正确初始化。
  static Status InitAll(KernelContext*, const KernelInitArgs&,
                        std::vector<std::unique_ptr<KernelState>>*);

  /// \brief Indicates whether execution can benefit from parallelization
  /// (splitting large chunks into smaller chunks and using multiple
  /// threads). Some kernels may not support parallel execution at
  /// all. Synchronization and concurrency-related issues are currently the
  /// responsibility of the Kernel's implementation.
  // 并行化标识。
  // 默认为 true。它告诉执行引擎（如 ExecPlan），当输入数据量很大时，是否可以将数据切分成小块（Chunks）并使用多线程同时调用此内核。某些依赖顺序的算法需要将其设为 false。
  bool parallelizable = true;

  /// \brief Indicates the level of SIMD instruction support in the host CPU is
  /// required to use the function. The intention is for functions to be able to
  /// contain multiple kernels with the same signature but different levels of SIMD,
  /// so that the most optimized kernel supported on a host's processor can be chosen.
  // SIMD 指令集级别。
  // 标记该内核实现所针对的硬件优化。例如，一个算子可以注册两个内核：一个 simd_level 为 NONE 的通用版本，另一个为 AVX2 的优化版本。Arrow 会在运行时自动检测并优先使用最高支持级别的内核。
  SimdLevel::type simd_level = SimdLevel::NONE;

  // Additional kernel-specific data
  // 内核特定的持久数据。
  // 用于存储那些在内核整个生命周期内不变的共享数据，区别于由 init 动态生成的、针对单次执行请求的 KernelState。
  std::shared_ptr<KernelState> data;
};

/// \brief The scalar kernel execution API that must be implemented for SCALAR
/// kernel types. This includes both stateless and stateful kernels. Kernels
/// depending on some execution state access that state via subclasses of
/// KernelState set on the KernelContext object. Implementations should
/// endeavor to write into pre-allocated memory if they are able, though for
/// some kernels (e.g. in cases when a builder like StringBuilder) must be
/// employed this may not be possible.
  // 指向实际计算逻辑的函数指针。
  // KernelContext*：提供运行时环境（如分配器、内核状态）。
  // const ExecSpan&：包含输入数据（以轻量级 Span 形式组织，支持 Chunk 访问）。
  // ExecResult*：存放计算结果的容器。
using ArrayKernelExec = Status (*)(KernelContext*, const ExecSpan&, ExecResult*);

/// \brief Kernel data structure for implementations of ScalarFunction. In
/// addition to the members found in Kernel, contains the null handling
/// and memory pre-allocation preferences.
// 在 Apache Arrow 的计算引擎中，ScalarKernel 是处理**标量函数（Scalar Function）**的核心数据结构。它是 Kernel 基类的具体子类，专门用于实现那些“输入一行，输出一行”且行与行之间互不依赖的计算逻辑。
// ScalarKernel 的主要作用是定义逐元素（Element-wise）计算的执行契约。
// 标量函数（如 $A + B$、$\sin(x)$、字符串长度等）具有以下特点：
// 行独立性：第 $i$ 行的结果仅取决于第 $i$ 行的输入，不依赖于第 $j$ 行。
// 结构一致性：输出数组的长度与输入数组的长度完全一致。
// ScalarKernel 通过扩展基类 Kernel，增加了关于内存预分配和空值自动处理的配置，使得开发者可以极简地实现高效的计算逻辑，而无需手动管理繁琐的内存布局。
struct ARROW_EXPORT ScalarKernel : public Kernel {
  ScalarKernel() = default;

  ScalarKernel(std::shared_ptr<KernelSignature> sig, ArrayKernelExec exec,
               KernelInit init = NULLPTR)
      : Kernel(std::move(sig), init), exec(exec) {}

  ScalarKernel(std::vector<InputType> in_types, OutputType out_type, ArrayKernelExec exec,
               KernelInit init = NULLPTR)
      : Kernel(std::move(in_types), std::move(out_type), std::move(init)), exec(exec) {}

  /// \brief Perform a single invocation of this kernel. Depending on the
  /// implementation, it may only write into preallocated memory, while in some
  /// cases it will allocate its own memory. Any required state is managed
  /// through the KernelContext.

  // 存储该内核的执行入口。
  ArrayKernelExec exec;

  /// \brief Writing execution results into larger contiguous allocations
  /// requires that the kernel be able to write into sliced output ArrayData*,
  /// including sliced output validity bitmaps. Some kernel implementations may
  /// not be able to do this, so setting this to false disables this
  /// functionality.
  // 标识内核是否支持写入“切片”数据。
  // 在 Arrow 中，为了提高效率，引擎可能会给内核一个预先分配好的长数组的一部分（切片）进行写入。
  // 如果某些内核实现（例如使用了特定的第三方库，或底层是 StringBuilder）只能从地址 0 开始写，或者无法处理非对齐的位图，则需将此值设为 false。
  bool can_write_into_slices = true;

  // For scalar functions preallocated data and intersecting arg validity
  // bitmaps is a reasonable default
  // 定义空值处理策略。
  // 默认为 NullHandling::INTERSECTION。
  // 这意味着框架会自动处理空值：如果输入参数中有任何一个是 Null，对应的输出行也会被自动设为 Null。
  NullHandling::type null_handling = NullHandling::INTERSECTION;
  // 定义结果缓冲区的分配方式。
  // 默认为 MemAllocation::PREALLOCATE。
  // 由于标量函数的输出长度是已知的，框架可以在调用 exec 之前，预先根据数据类型和长度分配好内存
  // 内核只需要通过指针直接写入结果，这极大减少了动态内存分配带来的开销。
  MemAllocation::type mem_allocation = MemAllocation::PREALLOCATE;
};

// ----------------------------------------------------------------------
// VectorKernel (for VectorFunction)

/// \brief Kernel data structure for implementations of VectorFunction. In
/// contains an optional finalizer function, the null handling and memory
/// pre-allocation preferences (which have different defaults from
/// ScalarKernel), and some other execution-related options.
// 在 Apache Arrow 的计算引擎中，VectorKernel 类是用于处理 向量函数（Vector Function） 的核心数据结构。它是 Kernel 基类的子类，专门针对那些输入和输出之间不一定是一一对应关系的复杂计算逻辑。
// 全局相关性：输出的某一行可能取决于输入的多个行或整列数据（例如：排序 Sort、去重 Unique）。
// 长度变化：输出数组的长度往往与输入不同（例如：过滤 Filter、分组聚合的前奏）。
// 状态累积：在处理分块数据（ChunkedArray）时，可能需要跨块维护中间状态（例如：哈希表）。
struct ARROW_EXPORT VectorKernel : public Kernel {
  /// \brief See VectorKernel::finalize member for usage
  // 在所有数据块处理完成后执行的收尾逻辑。
  // 某些算子（如基于哈希的算子）在处理分块输入时会累积状态，最后需要将这些中间状态转化为最终结果。此函数会直接修改 Datum 向量。
  using FinalizeFunc = std::function<Status(KernelContext*, std::vector<Datum>*)>;

  /// \brief Function for executing a stateful VectorKernel against a
  /// ChunkedArray input. Does not need to be defined for all VectorKernels
  // 分块执行函数
  // 一个函数指针，接收 ExecBatch。
  // 专门用于处理 ChunkedArray（分块数组）的执行入口。
  // 如果定义了此函数，引擎可以更高效地直接处理分块数据，而不需要先将分块数据合并成连续数组。
  using ChunkedExec = Status (*)(KernelContext*, const ExecBatch&, Datum* out);

  VectorKernel() = default;

  VectorKernel(std::vector<InputType> in_types, OutputType out_type, ArrayKernelExec exec,
               KernelInit init = NULLPTR, FinalizeFunc finalize = NULLPTR)
      : Kernel(std::move(in_types), std::move(out_type), std::move(init)),
        exec(exec),
        finalize(std::move(finalize)) {}

  VectorKernel(std::shared_ptr<KernelSignature> sig, ArrayKernelExec exec,
               KernelInit init = NULLPTR, FinalizeFunc finalize = NULLPTR)
      : Kernel(std::move(sig), std::move(init)),
        exec(exec),
        finalize(std::move(finalize)) {}

  /// \brief Perform a single invocation of this kernel. Any required state is
  /// managed through the KernelContext.
  // 单个原子操作的执行入口。
  // 继承自标量内核的执行签名，负责处理 ExecSpan（数据片段）。
  ArrayKernelExec exec;

  /// \brief Execute the kernel on a ChunkedArray. Does not need to be defined
  // 存储分块执行逻辑。默认为 NULLPTR。
  ChunkedExec exec_chunked = NULLPTR;

  /// \brief For VectorKernel, convert intermediate results into finalized
  /// results. Mutates input argument. Some kernels may accumulate state
  /// (example: hashing-related functions) through processing chunked inputs, and
  /// then need to attach some accumulated state to each of the outputs of
  /// processing each chunk of data.
  // 存储收尾回调函数。
  FinalizeFunc finalize;

  /// Since vector kernels generally are implemented rather differently from
  /// scalar/elementwise kernels (and they may not even yield arrays of the same
  /// size), so we make the developer opt-in to any memory preallocation rather
  /// than having to turn it off.
  // 空值处理策略。
  // 在向量函数中，框架不会自动通过按位与（AND）来计算空值位图。因为输出行和输入行没有固定对应关系，内核必须根据算法逻辑自行计算并分配位图。
  NullHandling::type null_handling = NullHandling::COMPUTED_NO_PREALLOCATE;
  // 框架不会预先分配结果内存。因为向量函数（如 Filter）在执行完之前通常不知道输出会有多少行，所以必须由内核在 exec 内部动态分配内存。
  MemAllocation::type mem_allocation = MemAllocation::NO_PREALLOCATE;

  /// \brief Writing execution results into larger contiguous allocations
  /// requires that the kernel be able to write into sliced output ArrayData*,
  /// including sliced output validity bitmaps. Some kernel implementations may
  /// not be able to do this, so setting this to false disables this
  /// functionality.
 // 是否允许将结果写入到大的连续内存的某个切片中。
  bool can_write_into_slices = true;

  /// Some vector kernels can do chunkwise execution using ExecSpanIterator,
  /// in some cases accumulating some state. Other kernels (like Take) need to
  /// be passed whole arrays and don't work on ChunkedArray inputs
  // 如果为 true，引擎可以使用迭代器逐块调用内核；如果为 false（例如某些全局排序算法），引擎必须先将输入的所有分块合并（Concatenate）成一个完整的数组再交给内核。
  bool can_execute_chunkwise = true;

  /// Some kernels (like unique and value_counts) yield non-chunked output from
  /// chunked-array inputs. This option controls how the results are boxed when
  /// returned from ExecVectorFunction
  ///
  /// true -> ChunkedArray
  /// false -> Array
  // true：即使输入是分块的，结果也返回为 ChunkedArray。
  // false：结果强制返回为单个 Array（适用于 unique 这种将多块输入压缩成一个去重集合的函数）。
  bool output_chunked = true;
};

// ----------------------------------------------------------------------
// ScalarAggregateKernel (for ScalarAggregateFunction)

using ScalarAggregateConsume = Status (*)(KernelContext*, const ExecSpan&);
using ScalarAggregateMerge = Status (*)(KernelContext*, KernelState&&, KernelState*);
// Finalize returns Datum to permit multiple return values
using ScalarAggregateFinalize = Status (*)(KernelContext*, Datum*);

/// \brief Kernel data structure for implementations of
/// ScalarAggregateFunction. The four necessary components of an aggregation
/// kernel are the init, consume, merge, and finalize functions.
///
/// * init: creates a new KernelState for a kernel.
/// * consume: processes an ExecSpan and updates the KernelState found in the
///   KernelContext.
/// * merge: combines one KernelState with another.
/// * finalize: produces the end result of the aggregation using the
///   KernelState in the KernelContext.
struct ARROW_EXPORT ScalarAggregateKernel : public Kernel {
  ScalarAggregateKernel(std::shared_ptr<KernelSignature> sig, KernelInit init,
                        ScalarAggregateConsume consume, ScalarAggregateMerge merge,
                        ScalarAggregateFinalize finalize, const bool ordered)
      : Kernel(std::move(sig), std::move(init)),
        consume(consume),
        merge(merge),
        finalize(finalize),
        ordered(ordered) {}

  ScalarAggregateKernel(std::vector<InputType> in_types, OutputType out_type,
                        KernelInit init, ScalarAggregateConsume consume,
                        ScalarAggregateMerge merge, ScalarAggregateFinalize finalize,
                        const bool ordered)
      : ScalarAggregateKernel(
            KernelSignature::Make(std::move(in_types), std::move(out_type)),
            std::move(init), consume, merge, finalize, ordered) {}

  /// \brief Merge a vector of KernelStates into a single KernelState.
  /// The merged state will be returned and will be set on the KernelContext.
  static Result<std::unique_ptr<KernelState>> MergeAll(
      const ScalarAggregateKernel* kernel, KernelContext* ctx,
      std::vector<std::unique_ptr<KernelState>> states);

  ScalarAggregateConsume consume;
  ScalarAggregateMerge merge;
  ScalarAggregateFinalize finalize;
  /// \brief Whether this kernel requires ordering
  /// Some aggregations, such as, "first", requires some kind of input order. The
  /// order can be implicit, e.g., the order of the input data, or explicit, e.g.
  /// the ordering specified with a window aggregation.
  /// The caller of the aggregate kernel is responsible for passing data in some
  /// defined order to the kernel. The flag here is a way for the kernel to tell
  /// the caller that data passed to the kernel must be defined in some order.
  bool ordered = false;
};

// ----------------------------------------------------------------------
// HashAggregateKernel (for HashAggregateFunction)

using HashAggregateResize = Status (*)(KernelContext*, int64_t);
using HashAggregateConsume = Status (*)(KernelContext*, const ExecSpan&);
using HashAggregateMerge = Status (*)(KernelContext*, KernelState&&, const ArrayData&);

// Finalize returns Datum to permit multiple return values
using HashAggregateFinalize = Status (*)(KernelContext*, Datum*);

/// \brief Kernel data structure for implementations of
/// HashAggregateFunction. The four necessary components of an aggregation
/// kernel are the init, consume, merge, and finalize functions.
///
/// * init: creates a new KernelState for a kernel.
/// * resize: ensure that the KernelState can accommodate the specified number of groups.
/// * consume: processes an ExecSpan (which includes the argument as well
///   as an array of group identifiers) and updates the KernelState found in the
///   KernelContext.
/// * merge: combines one KernelState with another.
/// * finalize: produces the end result of the aggregation using the
///   KernelState in the KernelContext.
struct ARROW_EXPORT HashAggregateKernel : public Kernel {
  HashAggregateKernel() = default;

  HashAggregateKernel(std::shared_ptr<KernelSignature> sig, KernelInit init,
                      HashAggregateResize resize, HashAggregateConsume consume,
                      HashAggregateMerge merge, HashAggregateFinalize finalize,
                      const bool ordered)
      : Kernel(std::move(sig), std::move(init)),
        resize(resize),
        consume(consume),
        merge(merge),
        finalize(finalize),
        ordered(ordered) {}

  HashAggregateKernel(std::vector<InputType> in_types, OutputType out_type,
                      KernelInit init, HashAggregateConsume consume,
                      HashAggregateResize resize, HashAggregateMerge merge,
                      HashAggregateFinalize finalize, const bool ordered)
      : HashAggregateKernel(
            KernelSignature::Make(std::move(in_types), std::move(out_type)),
            std::move(init), resize, consume, merge, finalize, ordered) {}

  HashAggregateResize resize;
  HashAggregateConsume consume;
  HashAggregateMerge merge;
  HashAggregateFinalize finalize;
  /// @brief whether the summarizer requires ordering
  /// This is similar to ScalarAggregateKernel. See ScalarAggregateKernel
  /// for detailed doc of this variable.
  bool ordered = false;
};

}  // namespace compute
}  // namespace arrow
