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

#include <memory>
#include <string>
#include <vector>

#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/util/visibility.h"

namespace arrow {
namespace compute {

class Function;
class FunctionOptionsType;

/// \brief A mutable central function registry for built-in functions as well
/// as user-defined functions. Functions are implementations of
/// arrow::compute::Function.
///
/// Generally, each function contains kernels which are implementations of a
/// function for a specific argument signature. After looking up a function in
/// the registry, one can either execute it eagerly with Function::Execute or
/// use one of the function's dispatch methods to pick a suitable kernel for
/// lower-level function execution.
// 在 Apache Arrow 的计算（Compute）模块中，FunctionRegistry 类扮演着**“中央图书馆”或“目录系统”**的角色。它是所有可调用的计算函数（Functions）及其相关配置的存储库。
// FunctionRegistry 的主要作用是管理计算函数的生命周期和查找逻辑：
// 集中存储：它维护了一个从“函数名称”到“函数对象（Function）”的映射表。当你调用 add 或 filter 时，系统会先来这里查询。
// 层级结构：它支持“父子级”嵌套。例如，你可以有一个包含基础算子的全局注册表，以及一个包含特定业务逻辑的本地注册表。如果本地找不到，它会向上去父级查找。
// 别名管理：允许为一个函数设置多个名字（例如 add 的别名可以是 plus）。
// 配置类型管理：除了存储函数本身，它还存储了函数所需的配置选项类型（FunctionOptionsType），用于序列化和验证。
class ARROW_EXPORT FunctionRegistry {
 public:
  ~FunctionRegistry();

  /// \brief Construct a new registry.
  ///
  /// Most users only need to use the global registry.
  // 创建一个独立的、空的注册表。
  static std::unique_ptr<FunctionRegistry> Make();

  /// \brief Construct a new nested registry with the given parent.
  ///
  /// Most users only need to use the global registry. The returned registry never changes
  /// its parent, even when an operation allows overwriting.
  // 创建一个带有父级节点的嵌套注册表。
  // 在查找函数时，如果当前注册表没有，它会递归地向 parent 查询。这种设计实现了配置的继承和重写。
  static std::unique_ptr<FunctionRegistry> Make(FunctionRegistry* parent);

  /// \brief Check whether a new function can be added to the registry.
  ///
  /// \returns Status::KeyError if a function with the same name is already registered.
  // 预检查是否可以添加某个函数。
  Status CanAddFunction(std::shared_ptr<Function> function, bool allow_overwrite = false);

  /// \brief Add a new function to the registry.
  ///
  /// \returns Status::KeyError if a function with the same name is already registered.
  // 正式将函数对象存入注册表。
  Status AddFunction(std::shared_ptr<Function> function, bool allow_overwrite = false);

  /// \brief Check whether an alias can be added for the given function name.
  ///
  /// \returns Status::KeyError if the function with the given name is not registered.
  // 检查是否可以将 source_name 作为 target_name 的别名。
  Status CanAddAlias(const std::string& target_name, const std::string& source_name);

  /// \brief Add alias for the given function name.
  ///
  /// \returns Status::KeyError if the function with the given name is not registered.
  // 添加映射。例如将 sum 映射到 aggregate_sum。调用 GetFunction("sum") 时将返回真正的执行对象。
  Status AddAlias(const std::string& target_name, const std::string& source_name);

  /// \brief Check whether a new function options type can be added to the registry.
  ///
  /// \return Status::KeyError if a function options type with the same name is already
  /// registered.
  // 注册函数选项的类型。
  // 某些函数（如 Cast 或 Filter）需要额外的参数（Options）。注册这些类型是为了支持算子的动态构建和跨语言调用时的参数解析。
  Status CanAddFunctionOptionsType(const FunctionOptionsType* options_type,
                                   bool allow_overwrite = false);

  /// \brief Add a new function options type to the registry.
  ///
  /// \returns Status::KeyError if a function options type with the same name is already
  /// registered.
  Status AddFunctionOptionsType(const FunctionOptionsType* options_type,
                                bool allow_overwrite = false);

  /// \brief Retrieve a function by name from the registry.
  // 最常用的方法，根据名字获取函数对象。
  // 支持递归查找父级注册表。
  Result<std::shared_ptr<Function>> GetFunction(const std::string& name) const;

  /// \brief Return vector of all entry names in the registry.
  ///
  /// Helpful for displaying a manifest of available functions.
  // 获取当前及父级注册表中所有已注册函数的名称列表。常用于生成 API 文档或 UI 界面上的算子清单。
  std::vector<std::string> GetFunctionNames() const;

  /// \brief Retrieve a function options type by name from the registry.
  Result<const FunctionOptionsType*> GetFunctionOptionsType(
      const std::string& name) const;

  /// \brief The number of currently registered functions.
  int num_functions() const;

  /// \brief The cast function object registered in AddFunction.
  ///
  /// Helpful for get cast function as needed.
  const Function* cast_function() const;

 private:
  FunctionRegistry();

  // Use PIMPL pattern to not have std::unordered_map here
  // 真正的底层数据结构（通常是 std::unordered_map）被封装在 FunctionRegistryImpl 中。这样做是为了保证头文件的整洁，并减少编译依赖，同时也方便在不改变公开 API 的情况下修改底层存储逻辑。
  class FunctionRegistryImpl;
  std::unique_ptr<FunctionRegistryImpl> impl_;

  explicit FunctionRegistry(FunctionRegistryImpl* impl);
};

}  // namespace compute
}  // namespace arrow
