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

#include "arrow/array.h"
#include "arrow/extension_type.h"
#include "arrow/visitor_generate.h"

namespace arrow {

// 将一个通用的、基类的 Array 对象，在运行时根据它的 type_id，“变身”为具体的子类（如 Int32Array），然后交给 visitor 处理。
// 定义名为 ARRAY_VISIT_INLINE 的宏。
// 参数 TYPE_CLASS：这是一个类型前缀，比如传进来的是 Int32 或 String。
// 将 TYPE_CLASS 与 Type 字符串拼接。如果参数是 Int32，则生成 Int32Type::type_id。
// 调用访问者对象（即你之前看到的 RecordBatchSerializer 或其他 Visitor）的 Visit 重载方法。
// 将基类 const Array& 强制转换为具体的子类引用（如 const Int32Array&）。
// 使用 C++11 的完美转发（Perfect Forwarding）
#define ARRAY_VISIT_INLINE(TYPE_CLASS)                                                   \
  case TYPE_CLASS##Type::type_id:                                                        \
    return visitor->Visit(                                                               \
        internal::checked_cast<const typename TypeTraits<TYPE_CLASS##Type>::ArrayType&>( \
            array),                                                                      \
        std::forward<ARGS>(args)...);

/// \brief Apply the visitors Visit() method specialized to the array type
///
/// \tparam VISITOR Visitor type that implements Visit() for all array types.
/// \tparam ARGS Additional arguments, if any, will be passed to the Visit function after
/// the `arr` argument
/// \return Status
///
/// A visitor is a type that implements specialized logic for each Arrow type.
/// Example usage:
///
/// ```
/// class ExampleVisitor {
///   arrow::Status Visit(arrow::NumericArray<Int32Type> arr) { ... }
///   arrow::Status Visit(arrow::NumericArray<Int64Type> arr) { ... }
///   ...
/// }
/// ExampleVisitor visitor;
/// VisitArrayInline(some_array, &visitor);
/// ```
// array.type_id() 返回一个枚举值（如 INT32, STRING, LIST 等）。
// ARROW_GENERATE_FOR_ALL_TYPES 是一个**“高阶宏”**（一个以另一个宏作为参数的宏）。
// 工作原理：这个宏内部列出了 Arrow 支持的所有数据类型。它会对每一个类型（如 Int8, UInt8, Int16... 一直到 Dictionary, Struct 等）调用一次你传入的宏 ARRAY_VISIT_INLINE。
// 展开效果：编译器在处理这一行时，会自动展开生成几十个 case 分支，每个分支都像我们之前解读的那样，将 Array 强转为具体的子类并调用 visitor->Visit。
template <typename VISITOR, typename... ARGS>
inline Status VisitArrayInline(const Array& array, VISITOR* visitor, ARGS&&... args) {
  switch (array.type_id()) {
    ARROW_GENERATE_FOR_ALL_TYPES(ARRAY_VISIT_INLINE);
    default:
      break;
  }
  return Status::NotImplemented("Type not implemented");
}

#undef ARRAY_VISIT_INLINE

}  // namespace arrow
