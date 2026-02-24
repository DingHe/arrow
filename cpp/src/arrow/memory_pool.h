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
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/type_fwd.h"
#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"

namespace arrow {

namespace internal {

///////////////////////////////////////////////////////////////////////
// Helper tracking memory statistics

/// \brief Memory pool statistics
///
/// 64-byte aligned so that all atomic values are on the same cache line.
// MemoryPoolStats 是一个高性能的辅助类，专门用于实时跟踪和统计内存分配器的运行状态。
// MemoryPoolStats 的核心作用是监控内存使用情况。它被嵌入在各种 MemoryPool 实现（如系统默认内存池、Jemalloc 或 Mimalloc 内存池）中，用于：
// 实时计数：记录当前已分配的内存字节数。
// 峰值监控：记录内存池自创建以来达到的最大内存负载（水位线）。
// 累计统计：统计历史累计分配的字节总数和分配次数，用于评估内存碎片或分配压力。
// 高性能并发支持：通过原子操作（Atomic）和缓存行对齐（Cache-line alignment），该类可以在多线程环境下并发更新，且不会造成严重的性能抖动。
// alignas(64)： 将类实例对齐到 64 字节（现代 CPU 常见的缓存行大小）
class alignas(64) MemoryPoolStats {
 private:
  // All atomics are updated according to Acquire-Release ordering.
  // https://en.cppreference.com/w/cpp/atomic/memory_order#Release-Acquire_ordering
  //
  // max_memory_, total_allocated_bytes_, and num_allocs_ only go up (they are
  // monotonically increasing) which can allow some optimizations.
  // 记录历史最高内存占用（Peak Memory usage）
  // 单调递增，只有在当前分配后的总量超过历史最大值时才会更新。
  std::atomic<int64_t> max_memory_{0};
  // 记录当前正在使用的内存字节数。
  // 动态波动。分配时增加，释放时减少。
  std::atomic<int64_t> bytes_allocated_{0};
  // 记录自内存池创建以来累计分配过的字节总数。
  // 单调递增。即使内存被释放，该值也不会减小。
  std::atomic<int64_t> total_allocated_bytes_{0};
  // 记录累计分配次数。
  // 单调递增。每执行一次 Allocate 操作，该计数器加 1。
  std::atomic<int64_t> num_allocs_{0};

 public:
  // 返回历史最高内存占用。
  int64_t max_memory() const { return max_memory_.load(std::memory_order_acquire); }
  // 返回当前内存池占用的字节数
  int64_t bytes_allocated() const {
    return bytes_allocated_.load(std::memory_order_acquire);
  }
  // 返回历史累计分配的总字节数。
  int64_t total_bytes_allocated() const {
    return total_allocated_bytes_.load(std::memory_order_acquire);
  }
  // 返回历史累计分配的次数
  int64_t num_allocations() const { return num_allocs_.load(std::memory_order_acquire); }
  // 在成功分配内存后更新所有统计指标。
  inline void DidAllocateBytes(int64_t size) {
    // Issue the load before everything else. max_memory_ is monotonically increasing,
    // so we can use a relaxed load before the read-modify-write.
    auto max_memory = max_memory_.load(std::memory_order_relaxed);
    const auto old_bytes_allocated =
        bytes_allocated_.fetch_add(size, std::memory_order_acq_rel);
    // Issue store operations on values that we don't depend on to proceed
    // with execution. When done, max_memory and old_bytes_allocated have
    // a higher chance of being available on CPU registers. This also has the
    // nice side-effect of putting 3 atomic stores close to each other in the
    // instruction stream.
    total_allocated_bytes_.fetch_add(size, std::memory_order_acq_rel);
    num_allocs_.fetch_add(1, std::memory_order_acq_rel);

    // If other threads are updating max_memory_ concurrently we leave the loop without
    // updating knowing that it already reached a value even higher than ours.
    const auto allocated = old_bytes_allocated + size;
    while (max_memory < allocated && !max_memory_.compare_exchange_weak(
                                         /*expected=*/max_memory, /*desired=*/allocated,
                                         std::memory_order_acq_rel)) {
    }
  }
  // 在释放内存后更新统计指标。
  inline void DidReallocateBytes(int64_t old_size, int64_t new_size) {
    if (new_size > old_size) {
      DidAllocateBytes(new_size - old_size);
    } else {
      DidFreeBytes(old_size - new_size);
    }
  }
  // 在进行重新分配（Reallocate）操作后更新指标。
  inline void DidFreeBytes(int64_t size) {
    bytes_allocated_.fetch_sub(size, std::memory_order_acq_rel);
  }
};

}  // namespace internal

/// Base class for memory allocation on the CPU.
///
/// Besides tracking the number of allocated bytes, the allocator also should
/// take care of the required 64-byte alignment.
// 在 Apache Arrow 项目中，MemoryPool（内存池）是整个库处理 CPU 内存分配的核心抽象。它不仅管理内存的申请与释放，还确保了数据布局符合高性能计算的需求。
// MemoryPool 是一个抽象基类，定义了 Arrow 中内存管理的核心标准：
// 强制对齐 (Alignment)：Arrow 依赖于 SIMD（单指令多数据）指令集来加速计算，而 SIMD 要求内存地址必须是对齐的。
// MemoryPool 默认确保所有分配的内存都是 64 字节对齐。
// 资源追踪 (Tracking)：实时监控内存的使用量、峰值以及分配次数，这对于防止内存泄漏和性能调优至关重要。
// 后端无关性 (Backend Agnostic)：通过该抽象类，Arrow 可以无缝切换底层分配器，例如使用系统默认分配器、jemalloc 或 mimalloc，而上层逻辑不需要任何改动。
// 内存碎片管理：通过统一的接口，内存池可以实现一些优化策略，如重用空闲块或将多余内存交还操作系统。

class ARROW_EXPORT MemoryPool {
 public:
  virtual ~MemoryPool() = default;

  /// \brief EXPERIMENTAL. Create a new instance of the default MemoryPool
  // 创建一个当前系统默认的内存池实例。
  // 这通常会根据编译选项返回系统分配器、jemalloc 或 mimalloc 的封装。
  static std::unique_ptr<MemoryPool> CreateDefault();

  /// Allocate a new memory region of at least size bytes.
  ///
  /// The allocated region shall be 64-byte aligned.
  // 分配至少 size 字节的内存。
  // 它是对带 alignment 参数版本的封装，默认使用 64 字节对齐（kDefaultBufferAlignment）。分配成功后的指针通过 out 返回。
  Status Allocate(int64_t size, uint8_t** out) {
    return Allocate(size, kDefaultBufferAlignment, out);
  }

  /// Allocate a new memory region of at least size bytes aligned to alignment.
  // 分配指定对齐大小的内存。这是子类必须实现的底层逻辑。
  virtual Status Allocate(int64_t size, int64_t alignment, uint8_t** out) = 0;

  /// Resize an already allocated memory section.
  ///
  /// As by default most default allocators on a platform don't support aligned
  /// reallocation, this function can involve a copy of the underlying data.
  // 调整已分配内存的大小。
  // 由于许多底层分配器不支持“对齐重分配”，该操作可能会涉及“申请新内存 -> 拷贝数据 -> 释放旧内存”的过程。
  virtual Status Reallocate(int64_t old_size, int64_t new_size, int64_t alignment,
                            uint8_t** ptr) = 0;
  Status Reallocate(int64_t old_size, int64_t new_size, uint8_t** ptr) {
    return Reallocate(old_size, new_size, kDefaultBufferAlignment, ptr);
  }

  /// Free an allocated region.
  ///
  /// @param buffer Pointer to the start of the allocated memory region
  /// @param size Allocated size located at buffer. An allocator implementation
  ///   may use this for tracking the amount of allocated bytes as well as for
  ///   faster deallocation if supported by its backend.
  /// @param alignment The alignment of the allocation. Defaults to 64 bytes.
  // 释放由该池分配的内存。
  // 除了指针外，还要求传入 size 和 alignment，这有助于某些分配器（如 jemalloc）更高效地回收内存并准确统计数据。
  virtual void Free(uint8_t* buffer, int64_t size, int64_t alignment) = 0;
  void Free(uint8_t* buffer, int64_t size) {
    Free(buffer, size, kDefaultBufferAlignment);
  }

  /// Return unused memory to the OS
  ///
  /// Only applies to allocators that hold onto unused memory.  This will be
  /// best effort, a memory pool may not implement this feature or may be
  /// unable to fulfill the request due to fragmentation.
  // 尽力而为（Best effort）地将池中持有的空闲内存归还给操作系统。这在防止内存碎片导致的虚高占用时非常有用。
  virtual void ReleaseUnused() {}

  /// Print statistics
  ///
  /// Print allocation statistics on stderr. The output format is
  /// implementation-specific. Not all memory pools implement this method.
  // 将内存分配的统计信息输出到 stderr。
  virtual void PrintStats() {}

  /// The number of bytes that were allocated and not yet free'd through
  /// this allocator.
  // 返回当前正在使用的内存字节数。
  virtual int64_t bytes_allocated() const = 0;

  /// Return peak memory allocation in this memory pool
  ///
  /// \return Maximum bytes allocated. If not known (or not implemented),
  /// returns -1
  // 返回自该池创建以来达到的内存峰值。如果无法追踪则返回 -1。
  virtual int64_t max_memory() const;

  /// The number of bytes that were allocated.
  // 返回该池自创建以来累计分配过的字节总数（不随释放而减小）
  virtual int64_t total_bytes_allocated() const = 0;

  /// The number of allocations or reallocations that were requested.
  // 返回累计发起的分配/重分配请求次数。
  virtual int64_t num_allocations() const = 0;

  /// The name of the backend used by this MemoryPool (e.g. "system" or "jemalloc").
  // 返回底层分配器的名称，如 "system"、"jemalloc" 或 "mimalloc"。
  virtual std::string backend_name() const = 0;

 protected:
  MemoryPool() = default;
};
// 在 Apache Arrow 项目中，LoggingMemoryPool 是一个典型的**装饰器模式（Decorator Pattern）**的应用。
// 它通过包装另一个 MemoryPool 实例，为内存分配行为添加了日志记录或调试功能。
class ARROW_EXPORT LoggingMemoryPool : public MemoryPool {
 public:
  explicit LoggingMemoryPool(MemoryPool* pool);
  ~LoggingMemoryPool() override = default;

  using MemoryPool::Allocate;
  using MemoryPool::Free;
  using MemoryPool::Reallocate;

  Status Allocate(int64_t size, int64_t alignment, uint8_t** out) override;
  Status Reallocate(int64_t old_size, int64_t new_size, int64_t alignment,
                    uint8_t** ptr) override;
  void Free(uint8_t* buffer, int64_t size, int64_t alignment) override;
  void ReleaseUnused() override;
  void PrintStats() override;

  int64_t bytes_allocated() const override;

  int64_t max_memory() const override;

  int64_t total_bytes_allocated() const override;

  int64_t num_allocations() const override;

  std::string backend_name() const override;

 private:
  MemoryPool* pool_;
};

/// Derived class for memory allocation.
///
/// Tracks the number of bytes and maximum memory allocated through its direct
/// calls. Actual allocation is delegated to MemoryPool class.
// ProxyMemoryPool 的核心作用是隔离统计信息。
// 当你有一个全局的 MemoryPool，但希望单独追踪某个特定子任务（例如某个特定的 Query 或特定的 DataSink）占用的内存时，你可以创建一个代理。
// 它会将内存分配请求转发给底层池，但会独立记录通过这个代理分配的字节数和峰值。
class ARROW_EXPORT ProxyMemoryPool : public MemoryPool {
 public:
  explicit ProxyMemoryPool(MemoryPool* pool);
  ~ProxyMemoryPool() override;

  using MemoryPool::Allocate;
  using MemoryPool::Free;
  using MemoryPool::Reallocate;

  Status Allocate(int64_t size, int64_t alignment, uint8_t** out) override;
  Status Reallocate(int64_t old_size, int64_t new_size, int64_t alignment,
                    uint8_t** ptr) override;
  void Free(uint8_t* buffer, int64_t size, int64_t alignment) override;
  void ReleaseUnused() override;
  void PrintStats() override;

  int64_t bytes_allocated() const override;

  int64_t max_memory() const override;

  int64_t total_bytes_allocated() const override;

  int64_t num_allocations() const override;

  std::string backend_name() const override;

 private:
  class ProxyMemoryPoolImpl;
  std::unique_ptr<ProxyMemoryPoolImpl> impl_;
};

/// EXPERIMENTAL MemoryPool wrapper with an upper limit
///
/// Checking for limits is not done in a fully thread-safe way, therefore
/// multi-threaded allocations might be able to go successfully above the
/// configured limit.
// CappedMemoryPool 是一个带上限保护的内存池。
// 它的主要作用是限制内存使用的软上限。如果在分配请求时发现当前已分配量加上新请求量将超过预设的 limit，它会拒绝分配并返回错误（OOM）。
class ARROW_EXPORT CappedMemoryPool : public MemoryPool {
 public:
  CappedMemoryPool(MemoryPool* wrapped_pool, int64_t bytes_allocated_limit)
      : wrapped_(wrapped_pool), bytes_allocated_limit_(bytes_allocated_limit) {}

  using MemoryPool::Allocate;
  using MemoryPool::Reallocate;

  Status Allocate(int64_t size, int64_t alignment, uint8_t** out) override;
  Status Reallocate(int64_t old_size, int64_t new_size, int64_t alignment,
                    uint8_t** ptr) override;
  void Free(uint8_t* buffer, int64_t size, int64_t alignment) override;

  void ReleaseUnused() override { wrapped_->ReleaseUnused(); }

  void PrintStats() override { wrapped_->PrintStats(); }

  int64_t bytes_allocated() const override { return wrapped_->bytes_allocated(); }

  int64_t max_memory() const override { return wrapped_->max_memory(); }

  int64_t total_bytes_allocated() const override {
    return wrapped_->total_bytes_allocated();
  }

  int64_t num_allocations() const override { return wrapped_->num_allocations(); }

  std::string backend_name() const override { return wrapped_->backend_name(); }

 private:
  Status OutOfMemory(int64_t current_allocated, int64_t requested) const;

  MemoryPool* wrapped_;
  const int64_t bytes_allocated_limit_;
};

/// \brief Return a process-wide memory pool based on the system allocator.
ARROW_EXPORT MemoryPool* system_memory_pool();

/// \brief Return a process-wide memory pool based on jemalloc.
///
/// May return NotImplemented if jemalloc is not available.
ARROW_EXPORT Status jemalloc_memory_pool(MemoryPool** out);

/// \brief Set jemalloc memory page purging behavior for future-created arenas
/// to the indicated number of milliseconds. See dirty_decay_ms and
/// muzzy_decay_ms options in jemalloc for a description of what these do. The
/// default is configured to 1000 (1 second) which releases memory more
/// aggressively to the operating system than the jemalloc default of 10
/// seconds. If you set the value to 0, dirty / muzzy pages will be released
/// immediately rather than with a time decay, but this may reduce application
/// performance.
ARROW_EXPORT
Status jemalloc_set_decay_ms(int ms);

/// \brief Get basic statistics from jemalloc's mallctl.
/// See the MALLCTL NAMESPACE section in jemalloc project documentation for
/// available stats.
ARROW_EXPORT
Result<int64_t> jemalloc_get_stat(const char* name);

/// \brief Reset the counter for peak bytes allocated in the calling thread to zero.
/// This affects subsequent calls to thread.peak.read, but not the values returned by
/// thread.allocated or thread.deallocated.
ARROW_EXPORT
Status jemalloc_peak_reset();

/// \brief Print summary statistics in human-readable form to stderr.
/// See malloc_stats_print documentation in jemalloc project documentation for
/// available opt flags.
ARROW_EXPORT
Status jemalloc_stats_print(const char* opts = "");

/// \brief Print summary statistics in human-readable form using a callback
/// See malloc_stats_print documentation in jemalloc project documentation for
/// available opt flags.
ARROW_EXPORT
Status jemalloc_stats_print(std::function<void(const char*)> write_cb,
                            const char* opts = "");

/// \brief Get summary statistics in human-readable form.
/// See malloc_stats_print documentation in jemalloc project documentation for
/// available opt flags.
ARROW_EXPORT
Result<std::string> jemalloc_stats_string(const char* opts = "");

/// \brief Return a process-wide memory pool based on mimalloc.
///
/// May return NotImplemented if mimalloc is not available.
ARROW_EXPORT Status mimalloc_memory_pool(MemoryPool** out);

/// \brief Return the names of the backends supported by this Arrow build.
ARROW_EXPORT std::vector<std::string> SupportedMemoryBackendNames();

}  // namespace arrow
