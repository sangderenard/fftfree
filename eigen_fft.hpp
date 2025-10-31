#ifndef EIGFFT_DEBUG
#define EIGFFT_DEBUG 0
#endif

#ifndef EIGFFT_TRACE_STOCKHAM
#define EIGFFT_TRACE_STOCKHAM 0
#endif

// eigen_fft.hpp (header-only)
#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cmath>
#include <complex>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <vector>
#include <type_traits>

// #if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
//   #include <xmmintrin.h>  // FTZ/DAZ
// #endif
#ifdef _OPENMP
#include <omp.h>
#endif

// Enforce OpenMP availability unless explicitly overridden for debug builds.
#if !defined(_OPENMP) && !defined(EIGFFT_ALLOW_SEQUENTIAL)
#error "fftfree requires OpenMP; rebuild with OpenMP enabled or define EIGFFT_ALLOW_SEQUENTIAL to opt-in to the sequential fallback."
#endif

namespace eigfft {

template<class T>
struct Plan;

struct KernelContext {
  virtual ~KernelContext() = default;
};

template <class T>
struct PlanArena {
  using Complex = std::complex<T>;
  Complex* twiddles = nullptr;
  int twiddle_count = 0;
  int* bitrev = nullptr;
  int bitrev_count = 0;

  Complex* baseline_a = nullptr;
  Complex* baseline_b = nullptr;
  Complex** baseline_columns = nullptr;
  int baseline_thread_capacity = 0;
  int baseline_lane_capacity = 0;

  Complex* stockham_ping = nullptr;
  Complex* stockham_pong = nullptr;
  std::ptrdiff_t* stockham_lane_bases = nullptr;
  int stockham_thread_capacity = 0;
  int stockham_lane_capacity = 0;

  Complex* nd_transpose = nullptr;
  std::size_t nd_transpose_capacity = 0;

  // Optional algorithm-specific special buffers. Fixed-size slot array for
  // fast, indexable hot-path access. Slots may be null if unused.
  static constexpr int kMaxSpecialBuffers = 8;
  Complex* special_buf[kMaxSpecialBuffers] = {};
  std::size_t special_capacity[kMaxSpecialBuffers] = {};
  std::ptrdiff_t special_stride[kMaxSpecialBuffers] = {};
};

template <class T>
struct PlanArenaShape {
  std::size_t twiddles = 0;
  std::size_t bitrev = 0;
  std::size_t baseline_complex = 0;   // count per buffer (a/b)
  std::size_t baseline_columns = 0;   // pointer slots
  std::size_t stockham_stage = 0;     // per ping/pong buffer
  std::size_t stockham_lane_bases = 0;
};

template <class T>
inline PlanArenaShape<T> compute_plan_arena_shape(int N, int thread_capacity, int lane_capacity) {
  PlanArenaShape<T> shape;
  const int threads = std::max(1, thread_capacity);
  const int lanes = std::max(1, lane_capacity);
  shape.twiddles = static_cast<std::size_t>(std::max(1, N / 2));
  shape.bitrev = static_cast<std::size_t>(N);
  shape.baseline_complex = static_cast<std::size_t>(threads) * static_cast<std::size_t>(lanes);
  shape.baseline_columns = static_cast<std::size_t>(threads) * static_cast<std::size_t>(lanes);
  shape.stockham_stage = static_cast<std::size_t>(threads) * static_cast<std::size_t>(lanes) * static_cast<std::size_t>(N);
  shape.stockham_lane_bases = static_cast<std::size_t>(threads) * static_cast<std::size_t>(lanes);
  return shape;
}

namespace detail {

#if defined(EIGFFT_ALLOW_SEQUENTIAL)
inline constexpr bool kAllowSequentialFallback = true;
#else
inline constexpr bool kAllowSequentialFallback = false;
#endif

#if defined(EIGFFT_ENABLE_EXTERNAL_KERNEL)
inline constexpr bool kExternalKernelAvailable = true;
#else
inline constexpr bool kExternalKernelAvailable = false;
#endif

class WorkerPool {
 public:
  WorkerPool() = default;

  explicit WorkerPool(int threads) { reset(threads); }

  ~WorkerPool() { shutdown(); }

  WorkerPool(const WorkerPool&) = delete;
  WorkerPool& operator=(const WorkerPool&) = delete;
  WorkerPool(WorkerPool&&) = delete;
  WorkerPool& operator=(WorkerPool&&) = delete;

  void reset(int threads) {
    const int requested = std::max(1, threads);
    if (workers_.empty()) {
      stop_ = false;
      job_.fn = nullptr;
      job_.total = 0;
      job_.chunk = 1;
      job_.chunk_count = 0;
      job_.next.store(0, std::memory_order_relaxed);
      job_.pending.store(0, std::memory_order_relaxed);
      job_.active = false;
      total_threads_ = 1;
      worker_count_ = 0;
    }

    if (requested <= total_threads_) {
      return;  // keep existing fleet alive; never shrink in the hot path
    }

    const int additional_workers = requested - total_threads_;
    if (additional_workers <= 0) return;

    workers_.reserve(static_cast<size_t>(worker_count_ + additional_workers));
    for (int i = 0; i < additional_workers; ++i) {
      const int worker_id = worker_count_ + 1;
      workers_.emplace_back([this, worker_id]() { worker_loop(worker_id); });
      ++worker_count_;
      ++total_threads_;
    }
  }

  int size() const { return total_threads_; }

  template <class Fn>
  void parallel_for(size_t total_items, size_t chunk, Fn&& fn) {
    if (total_items == 0) return;
    if (chunk == 0) chunk = 1;
    const size_t chunk_size = chunk;
    std::function<void(size_t, size_t, int)> wrapped = std::forward<Fn>(fn);
#if EIGFFT_TRACE_STOCKHAM
    std::cout << "[pool] parallel_for begin total=" << total_items
              << " chunk=" << chunk_size
              << " worker_count=" << worker_count_
              << " total_threads=" << total_threads_ << std::endl;
#endif
    if (total_threads_ <= 1 || worker_count_ == 0 || total_items <= chunk_size) {
      size_t start = 0;
      while (start < total_items) {
        const size_t end = std::min(total_items, start + chunk_size);
        wrapped(start, end, 0);
        start = end;
      }
#if EIGFFT_TRACE_STOCKHAM
      std::cout << "[pool] parallel_for completed inline" << std::endl;
#endif
      return;
    }

    const size_t chunk_count = (total_items + chunk_size - 1) / chunk_size;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      job_.fn = wrapped;
      job_.total = total_items;
      job_.chunk = chunk_size;
      job_.chunk_count = chunk_count;
      job_.next.store(0, std::memory_order_relaxed);
      // Count main + workers; everyone decrements once when done draining.
      job_.pending.store(worker_count_ + 1, std::memory_order_relaxed);
      job_.active = true;
    }
    cv_job_.notify_all();

    drain_chunks(0);
    // Main thread finished its share: decrement pending and possibly close job.
    {
      std::lock_guard<std::mutex> lock(mutex_);
      const int remaining = job_.pending.fetch_sub(1, std::memory_order_acq_rel);
      if (remaining == 1) {
        job_.active = false;
        cv_done_.notify_one();
      }
    }

    std::unique_lock<std::mutex> lock(mutex_);
    cv_done_.wait(lock, [&] { return !job_.active; });
    job_.fn = nullptr;
#if EIGFFT_TRACE_STOCKHAM
    std::cout << "[pool] parallel_for finished" << std::endl;
#endif
  }

 private:
  struct Job {
    std::function<void(size_t, size_t, int)> fn;
    size_t total = 0;
    size_t chunk = 1;
    size_t chunk_count = 0;
    std::atomic<size_t> next{0};
    std::atomic<int> pending{0};
    bool active = false;
  };

  void shutdown() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (workers_.empty()) {
        stop_ = true;
        job_.active = false;
      } else {
        stop_ = true;
        job_.active = false;
      }
    }
    cv_job_.notify_all();
    for (auto& worker : workers_) {
      if (worker.joinable()) worker.join();
    }
    workers_.clear();
    total_threads_ = 1;
    worker_count_ = 0;
  }

  void worker_loop(int worker_id) {
    for (;;) {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_job_.wait(lock, [&] { return stop_ || job_.active; });
      if (stop_) return;
#if EIGFFT_TRACE_STOCKHAM
  std::cout << "[pool] worker " << worker_id << " woke" << std::endl;
#endif
      lock.unlock();

      drain_chunks(worker_id);

      const int remaining = job_.pending.fetch_sub(1, std::memory_order_acq_rel);
      if (remaining == 1) {
        std::lock_guard<std::mutex> done_lock(mutex_);
        job_.active = false;
        cv_done_.notify_one();
      }
    }
  }

  void drain_chunks(int worker_id) {
    for (;;) {
      const size_t index = job_.next.fetch_add(1, std::memory_order_acq_rel);
      if (index >= job_.chunk_count) break;
      const size_t start = index * job_.chunk;
      const size_t end = std::min(job_.total, start + job_.chunk);
#if EIGFFT_TRACE_STOCKHAM
      std::cout << "[pool] worker " << worker_id << " processing chunk idx=" << index
            << " start=" << start << " end=" << end << std::endl;
#endif
      job_.fn(start, end, worker_id);
    }
  }

  int total_threads_ = 1;
  int worker_count_ = 0;
  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable cv_job_;
  std::condition_variable cv_done_;
  bool stop_ = false;
  Job job_{};
};

// Close the local `detail` namespace while including standalone butterfly
// headers. The butterfly headers declare their own `eigfft::detail` scope so
// include them at namespace scope to avoid nested `eigfft::detail::eigfft::detail`.
} // namespace detail
} // namespace eigfft

// Force the core to use the external butterfly types (compat header with
// aliases). You can set EIGFFT_USE_EXTERNAL_BFLY=0 to keep the old
// internal definitions in `eigen_fft.hpp` (not recommended).
#ifndef EIGFFT_USE_EXTERNAL_BFLY
#define EIGFFT_USE_EXTERNAL_BFLY 1
#endif

#include "butterfly_api.hpp"
#if EIGFFT_USE_EXTERNAL_BFLY
#include "butterfly_kernel.hpp" // compatibility header; defines aliases into detail
#else
// If external butterfly is disabled, optionally include internal lightweight
// implementation (kept separate to reduce file size). By default we prefer the
// external compatibility header.
#include "butterfly_impl_radix2.hpp"
#endif

namespace eigfft {
namespace detail {

template <class T>
struct StockhamState : KernelContext {
  using Complex = std::complex<T>;

  explicit StockhamState(Plan<T>& plan, PlanArena<T>& arena_ref)
      : arena(arena_ref),
        pool(plan.use_threads ? plan.effective_threads(plan.packet_cols) : 1),
        N(plan.N),
        plan_owner(&plan) {
    const int threads = std::max(1, pool.size());
    if (arena.stockham_thread_capacity < threads) {
      throw std::invalid_argument("PlanArena stockham_thread_capacity too small for requested threads");
    }
    if (arena.stockham_lane_capacity <= 0) {
      throw std::invalid_argument("PlanArena stockham_lane_capacity must be positive");
    }
    if (!arena.stockham_ping || !arena.stockham_pong || !arena.stockham_lane_bases) {
      throw std::invalid_argument("PlanArena missing Stockham scratch buffers");
    }
  }

  void ensure_lane_capacity(int requested) {
    const int desired = std::max(1, requested);
    if ((desired > arena.stockham_lane_capacity) && plan_owner && plan_owner->arena_resizer) {
      const int threads = std::max(1, pool.size());
      plan_owner->arena_resizer(arena, plan_owner->N, threads, desired);
    }
  }

  void ensure_threads(int threads) {
    int desired = std::max(1, threads);
    if ((desired > arena.stockham_thread_capacity) && plan_owner && plan_owner->arena_resizer) {
      const int lanes = std::max(1, arena.stockham_lane_capacity);
      plan_owner->arena_resizer(arena, plan_owner->N, desired, lanes);
    }
    const int clamped_threads = std::min(desired, arena.stockham_thread_capacity);
    if (clamped_threads > pool.size()) {
      pool.reset(clamped_threads);
    }
  }

  size_t thread_stride() const {
    return static_cast<size_t>(N) * static_cast<size_t>(std::max(1, arena.stockham_lane_capacity));
  }

  PlanArena<T>& arena;
  WorkerPool pool;
  int N = 0;
  Plan<T>* plan_owner = nullptr;
};

}  // namespace detail

enum class MetadataKind { Twiddle, LayoutMap, StageSnapshot, StageParams, ButterflyPairs, StageInvariant, TwiddleIndexMap };

template <class T>
struct MetadataRequest {
  MetadataKind kind = MetadataKind::Twiddle;
  std::complex<T>* complex_buffer = nullptr;
  std::size_t element_count = 0;
};

template <class T>
struct AxisLayout {
  using Complex = std::complex<T>;
  Complex* base = nullptr;
  Eigen::Index axis_size = 0;
  Eigen::Index batch_size = 0;
  Eigen::Index axis_stride = 1;
  Eigen::Index batch_stride = 0;
  const MetadataRequest<T>* metadata_requests = nullptr;
  int metadata_request_count = 0;

  Complex* column_ptr(Eigen::Index batch) const {
    return base + batch * batch_stride;
  }

  Complex* element_ptr(Eigen::Index axis_idx, Eigen::Index batch) const {
    return base + axis_idx * axis_stride + batch * batch_stride;
  }

  bool valid() const {
    return base != nullptr && axis_size > 0;
  }
};

namespace detail {

template <class T>
struct TwiddleMetadataWriter {
  const MetadataRequest<T>* requests = nullptr;
  int count = 0;
  int stages = 0;
  std::complex<T>* twiddle_buffer = nullptr;
  bool has_twiddle = false;
  std::size_t total_twiddles = 0;

  TwiddleMetadataWriter(const AxisLayout<T>& layout, int stage_count)
      : requests(layout.metadata_requests),
        count(layout.metadata_request_count),
        stages(stage_count) {
    if (!requests || count <= 0 || stages <= 0) return;
    total_twiddles = compute_total_twiddles(stages);
    for (int i = 0; i < count; ++i) {
      const MetadataRequest<T>& req = requests[i];
      if (req.kind != MetadataKind::Twiddle) continue;
      if (total_twiddles > 0) {
        if (!req.complex_buffer) {
          throw std::invalid_argument("Twiddle metadata requires complex buffer");
        }
        if (req.element_count < total_twiddles) {
          throw std::invalid_argument("Twiddle metadata buffer too small");
        }
      }
      twiddle_buffer = req.complex_buffer;
      has_twiddle = (twiddle_buffer != nullptr);
    }
  }

  bool enabled() const { return has_twiddle; }

  void record(int stage_idx, int twiddle_idx, const std::complex<T>& value) const {
    if (!has_twiddle) return;
    if (stage_idx < 0 || stage_idx >= stages) return;
    if (twiddle_idx < 0) return;
    const std::size_t per_stage = static_cast<std::size_t>(1) << stage_idx;
    if (static_cast<std::size_t>(twiddle_idx) >= per_stage) return;
    const std::size_t stage_offset = (static_cast<std::size_t>(1) << stage_idx) - 1;
    twiddle_buffer[stage_offset + static_cast<std::size_t>(twiddle_idx)] = value;
  }

  void record_plan_twiddles(const Plan<T>& plan) const {
    if (!enabled()) return;
    const int total_stages = std::min(stages, plan.lgN);
    for (int stage = 0; stage < total_stages; ++stage) {
      const int m = 1 << stage;
      const int distance = m << 1;
      const int tw_step = plan.N / distance;
      for (int j = 0; j < m; ++j) {
        record(stage, j, plan.W[j * tw_step]);
      }
    }
  }

 private:
  static std::size_t compute_total_twiddles(int stage_count) {
    if (stage_count <= 0) return 0;
    std::size_t total = 0;
    for (int i = 0; i < stage_count; ++i) {
      total += static_cast<std::size_t>(1) << i;
    }
    return total;
  }
};

template <class T>
struct LayoutMetadataWriter {
  const MetadataRequest<T>* requests = nullptr;
  int count = 0;
  std::complex<T>* layout_buffer = nullptr;
  std::size_t capacity = 0;
  std::size_t N = 0;

  LayoutMetadataWriter(const AxisLayout<T>& layout, std::size_t n)
      : requests(layout.metadata_requests),
        count(layout.metadata_request_count),
        N(n) {
    if (!requests || count <= 0 || N == 0) return;
    for (int i = 0; i < count; ++i) {
      const MetadataRequest<T>& req = requests[i];
      if (req.kind != MetadataKind::LayoutMap) continue;
      layout_buffer = req.complex_buffer;
      capacity = req.element_count;
      break;
    }
    if (layout_buffer && capacity < N) {
      throw std::invalid_argument("LayoutMap metadata buffer too small");
    }
  }

  bool enabled() const { return layout_buffer != nullptr && N > 0; }

  void set(std::size_t position, int source_index) const {
    if (!enabled() || position >= N) return;
    layout_buffer[position] = std::complex<T>(static_cast<T>(source_index), T(0));
  }
};

template <class T>
struct StageSnapshotWriter {
  const MetadataRequest<T>* reqs = nullptr;
  int count = 0;
  std::complex<T>* buf = nullptr;
  std::size_t N = 0;
  int stages = 0;

  StageSnapshotWriter(const AxisLayout<T>& layout, std::size_t n, int s)
      : reqs(layout.metadata_requests), count(layout.metadata_request_count),
        N(n), stages(s) {
    if (!reqs || count <= 0 || N == 0 || stages <= 0) return;
    for (int i = 0; i < count; ++i) {
      if (reqs[i].kind == MetadataKind::StageSnapshot) {
        buf = reqs[i].complex_buffer;
        if (!buf || reqs[i].element_count < N * (std::size_t)stages)
          throw std::invalid_argument("StageSnapshot buffer too small");
        break;
      }
    }
  }
  bool enabled() const { return buf && N && stages; }
  void set(int stage, int pos, const std::complex<T>& v) const {
    if (enabled() && stage >= 0 && stage < stages && pos >= 0 && (std::size_t)pos < N)
      buf[(std::size_t)stage * N + (std::size_t)pos] = v;
  }
  void copy_lane0(int stage, const std::complex<T>* src, int stride /*=1*/) const {
    if (!enabled()) return;
    for (std::size_t i = 0; i < N; ++i) set(stage, (int)i, src[i * (std::size_t)stride]);
  }
};

template <class T>
struct StageParamsWriter {
  const MetadataRequest<T>* reqs = nullptr;
  int count = 0;
  std::complex<T>* buf = nullptr;
  int stages = 0;
  StageParamsWriter(const AxisLayout<T>& layout, int s)
      : reqs(layout.metadata_requests), count(layout.metadata_request_count), stages(s) {
    if (!reqs || count <= 0 || stages <= 0) return;
    for (int i = 0; i < count; ++i) {
      if (reqs[i].kind == MetadataKind::StageParams) {
        buf = reqs[i].complex_buffer;
        if (!buf || reqs[i].element_count < (std::size_t)stages)
          throw std::invalid_argument("StageParams buffer too small");
        break;
      }
    }
  }
  bool enabled() const { return buf && stages > 0; }
  void set(int stage, int distance, int tw_step) const {
    if (!enabled() || stage < 0 || stage >= stages) return;
    buf[stage] = std::complex<T>(static_cast<T>(distance), static_cast<T>(tw_step));
  }
};

template <class T>
struct ButterflyPairsWriter {
  const MetadataRequest<T>* reqs = nullptr;
  int count = 0;
  std::complex<T>* buf = nullptr;
  std::size_t N = 0;
  int stages = 0;
  ButterflyPairsWriter(const AxisLayout<T>& layout, std::size_t n, int s)
      : reqs(layout.metadata_requests), count(layout.metadata_request_count), buf(nullptr), N(n), stages(s) {
    if (!reqs || count <= 0 || N == 0 || stages <= 0) return;
    for (int i = 0; i < count; ++i) {
      if (reqs[i].kind == MetadataKind::ButterflyPairs) {
        buf = reqs[i].complex_buffer;
        if (!buf || reqs[i].element_count < N * (std::size_t)stages)
          throw std::invalid_argument("ButterflyPairs buffer too small");
        break;
      }
    }
  }
  bool enabled() const { return buf && N && stages; }
  void set(int stage, int pos, int lhs, int rhs) const {
    if (!enabled() || stage < 0 || stage >= stages || pos < 0 || (std::size_t)pos >= N) return;
    buf[(std::size_t)stage * N + (std::size_t)pos] = std::complex<T>(static_cast<T>(lhs), static_cast<T>(rhs));
  }
};

template <class T>
struct StageInvariantWriter {
  const MetadataRequest<T>* reqs = nullptr;
  int count = 0;
  std::complex<T>* buf = nullptr;
  int stages = 0;
  StageInvariantWriter(const AxisLayout<T>& layout, int s)
      : reqs(layout.metadata_requests), count(layout.metadata_request_count), buf(nullptr), stages(s) {
    if (!reqs || count <= 0 || stages <= 0) return;
    for (int i = 0; i < count; ++i) {
      if (reqs[i].kind == MetadataKind::StageInvariant) {
        buf = reqs[i].complex_buffer;
        if (!buf || reqs[i].element_count < (std::size_t)stages)
          throw std::invalid_argument("StageInvariant buffer too small");
        break;
      }
    }
  }
  bool enabled() const { return buf && stages > 0; }
  void set(int stage, T max_r0, T max_r1) const {
    if (!enabled() || stage < 0 || stage >= stages) return;
    buf[stage] = std::complex<T>(max_r0, max_r1);
  }
};

template <class T>
struct TwiddleIndexWriter {
  const MetadataRequest<T>* reqs = nullptr;
  int count = 0;
  std::complex<T>* buf = nullptr;
  std::size_t N = 0;
  int stages = 0;
  TwiddleIndexWriter(const AxisLayout<T>& layout, std::size_t n, int s)
      : reqs(layout.metadata_requests), count(layout.metadata_request_count), buf(nullptr), N(n), stages(s) {
    if (!reqs || count <= 0 || N == 0 || stages <= 0) return;
    for (int i = 0; i < count; ++i) {
      if (reqs[i].kind == MetadataKind::TwiddleIndexMap) {
        buf = reqs[i].complex_buffer;
        if (!buf || reqs[i].element_count < N * (std::size_t)stages)
          throw std::invalid_argument("TwiddleIndexMap buffer too small");
        break;
      }
    }
  }
  bool enabled() const { return buf && N && stages; }
  void set(int stage, int pos, int tw_index) const {
    if (!enabled() || stage < 0 || stage >= stages || pos < 0 || (std::size_t)pos >= N) return;
    buf[(std::size_t)stage * N + (std::size_t)pos] = std::complex<T>(static_cast<T>(tw_index), T(0));
  }
};

}  // namespace detail

enum class KernelKind { Baseline, Stockham, External };

enum class KernelAccuracy { Default, HighPrecision, Reference };

namespace detail {

template<class T>
void baseline_execute_axis(const Plan<T>& P, const AxisLayout<T>& layout, KernelContext* ctx);

template<class T>
std::unique_ptr<KernelContext> baseline_create_state(Plan<T>& P);

template<class T>
void baseline_destroy_state(Plan<T>& P, KernelContext* ctx);

template<class T>
void stockham_execute_axis(const Plan<T>& P, const AxisLayout<T>& layout, KernelContext* ctx);

template<class T>
std::unique_ptr<KernelContext> stockham_create_state(Plan<T>& P);

template<class T>
void stockham_destroy_state(Plan<T>& P, KernelContext* ctx);

template<class T>
void external_execute_axis(const Plan<T>& P, const AxisLayout<T>& layout, KernelContext* ctx);

template<class T>
std::unique_ptr<KernelContext> external_create_state(Plan<T>& P);

template<class T>
void external_destroy_state(Plan<T>& P, KernelContext* ctx);

} // namespace detail

template<class T> struct Plan {
  using Complex = std::complex<T>;
  using RowBuffer = Eigen::Matrix<Complex, 1, Eigen::Dynamic, Eigen::RowMajor>;
  using ArenaResizer = void (*)(PlanArena<T>&, int N, int threads, int lanes);
  struct KernelDescriptor;

  int N;
  bool inverse;
  Complex* W = nullptr;      // twiddle factors (size N/2)
  int* bitrev = nullptr;     // bit-reversal indices (size N)
  int lgN;
  bool use_threads;
  int requested_threads;
  int packet_cols;  // auto from Eigen packets unless overridden

  // Per-plan default butterfly configs. These are hot-path stable pointers
  // (no allocations) that kernels will reference during execution. Users may
  // override these on the Plan before dispatch to change butterfly behavior
  // (e.g. select DIF vs DIT, set conjugation). Defaults preserve legacy
  // behavior (Radix2_DIT, no conjugation).
  ButterflyConfig<T> butterfly_default_baseline;
  ButterflyConfig<T> butterfly_default_stockham;
  ButterflyConfig<T> butterfly_default_external;

  struct Limits {
    static constexpr int kCompileTimeMaxThreads = 16;
    static constexpr int kDefaultRuntimeThreads = 4;
    static constexpr int kDefaultLaneCapacity = 2;

    static constexpr int compile_time_max_lane_capacity() {
      if constexpr (std::is_same_v<T, float>) {
        return 8;  // AVX-512 holds 8 complex<float>; AVX2 fits within this bound.
      } else {
        return 4;  // AVX-512 holds 4 complex<double>; AVX2 fits within this bound.
      }
    }
  };

  enum class ParallelDim { Auto, Columns, KBlocks };
  enum class Schedule { Auto, Static, Dynamic, Guided };
  struct Tuning {
    ParallelDim parallel_dim = ParallelDim::Auto;
    Schedule schedule = Schedule::Auto;
    int packet_step = 0;            // 0 => auto (=packet_cols), else multiple thereof
    int min_work_per_thread = 64;   // heuristics gate
    bool force_ftz_daz = true;
  } tuning;

  // Butterfly execution mode for Stockham: explicit, not implicit adapter.
  // - UseScatter: Stockham will call the out-of-place ButterflyScatter implementation.
  // - UseInplaceAdapter: Stockham will use PlanArena scratch and call an
  //   in-place kernel via the Inplace-to-Scatter adapter (compatibility fallback).
  enum class ButterflyMode { UseScatter = 0, UseInplaceAdapter = 1 };
  ButterflyMode butterfly_stockham_mode = ButterflyMode::UseScatter;

  struct Workspace {
    void bind(Plan& plan_ref) {
      owner = &plan_ref;
      arena = &plan_ref.arena;
    }

    void ensure(int threadCount, int cols) {
      if (!arena) {
        throw std::logic_error("Workspace not bound to PlanArena");
      }
      if (threadCount <= 0) threadCount = 1;
      if (cols <= 0) cols = 1;
      if ((cols > arena->baseline_lane_capacity ||
           threadCount > arena->baseline_thread_capacity) && owner && owner->arena_resizer) {
        owner->arena_resizer(*arena, owner->N, threadCount, cols);
      }
      const int clamped_threads = std::min(threadCount, arena->baseline_thread_capacity);
      const int clamped_cols = std::min(cols, arena->baseline_lane_capacity);
      threads = std::max(1, clamped_threads);
      capacity = std::max(1, clamped_cols);
    }

    Complex* row_a(int thread) const {
      return arena->baseline_a + static_cast<std::size_t>(thread) * arena->baseline_lane_capacity;
    }

    Complex* row_b(int thread) const {
      return arena->baseline_b + static_cast<std::size_t>(thread) * arena->baseline_lane_capacity;
    }

    Complex** column_row(int thread) const {
      return arena->baseline_columns + static_cast<std::size_t>(thread) * arena->baseline_lane_capacity;
    }

    void ensure_nd_buffer(Eigen::Index rows, Eigen::Index cols) {
      if (!arena) {
        throw std::logic_error("Workspace not bound to PlanArena");
      }
      if (rows <= 0 || cols <= 0) {
        nd_rows = nd_cols = 0;
        return;
      }
      const std::size_t required = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
      if (required > arena->nd_transpose_capacity) {
        if (owner && owner->arena_resizer) {
          owner->pending_nd_capacity = required;
          owner->arena_resizer(*arena, owner->N, std::max(1, threads), std::max(1, capacity));
        } else {
          throw std::invalid_argument("PlanArena nd_transpose capacity insufficient");
        }
      }
      if (required > arena->nd_transpose_capacity) {
        throw std::invalid_argument("PlanArena nd_transpose capacity insufficient after resize");
      }
      nd_rows = rows;
      nd_cols = cols;
      if (owner) {
        owner->pending_nd_capacity = 0;
      }
    }

    Complex* nd_buffer() const {
      return arena ? arena->nd_transpose : nullptr;
    }

    int capacity = 0;
    int threads = 0;
    Eigen::Index nd_rows = 0;
    Eigen::Index nd_cols = 0;

   private:
    PlanArena<T>* arena = nullptr;
    Plan* owner = nullptr;
  };

  mutable Workspace workspace;
  PlanArena<T>& arena;
  ArenaResizer arena_resizer = nullptr;
  mutable std::size_t pending_nd_capacity = 0;
  const KernelDescriptor* kernel_desc_ = nullptr;
  std::unique_ptr<KernelContext> kernel_state_;

  struct KernelDescriptor {
    KernelKind kind = KernelKind::Baseline;
    const char* name = nullptr;
    KernelAccuracy accuracy = KernelAccuracy::Default;
    bool realtime_safe = true;
    using CreateStateFn = std::unique_ptr<KernelContext> (*)(Plan&);
    using ExecuteFn = void (*)(const Plan&, const AxisLayout<T>&, KernelContext*);
    using DestroyStateFn = void (*)(Plan&, KernelContext*);
    CreateStateFn create_state = nullptr;
    ExecuteFn execute_axis = nullptr;
    DestroyStateFn destroy_state = nullptr;
  };

  // Generic, algorithm-agnostic request for extra plan workspace resources.
  struct AdvancedWorkspaceRequest {
    int min_lane_capacity = 0;              // desired per-thread lane capacity
    int min_thread_capacity = 0;            // desired thread capacity
    std::size_t min_nd_transpose_capacity = 0; // desired ND transpose capacity (elements)
    bool prefer_inplace_emulation = false;  // hint for allocator
    // Special buffer requests: preferred_slot >= 0 to request a fixed slot.
    struct SpecialRequest {
      int preferred_slot = -1; // -1 = not set; caller should prefer explicit slot for speed
      std::size_t elements = 0; // number of Complex elements requested
      bool operator==(const SpecialRequest& o) const noexcept {
        return preferred_slot == o.preferred_slot && elements == o.elements;
      }
    };
    std::vector<SpecialRequest> special_requests;

    bool operator==(const AdvancedWorkspaceRequest& o) const noexcept {
      if (min_lane_capacity != o.min_lane_capacity) return false;
      if (min_thread_capacity != o.min_thread_capacity) return false;
      if (min_nd_transpose_capacity != o.min_nd_transpose_capacity) return false;
      if (prefer_inplace_emulation != o.prefer_inplace_emulation) return false;
      if (special_requests.size() != o.special_requests.size()) return false;
      for (size_t i = 0; i < special_requests.size(); ++i) {
        if (!(special_requests[i] == o.special_requests[i])) return false;
      }
      return true;
    }
  };

  // Cached fulfilled advanced requests for this Plan. Durable for the Plan lifetime.
  mutable std::vector<AdvancedWorkspaceRequest> cached_advanced_requests;
  // Pending per-slot special capacities requested prior to calling arena_resizer.
  mutable std::array<std::size_t, PlanArena<T>::kMaxSpecialBuffers> pending_special_capacity{};
  // Mutex to protect cached requests and pending_special_capacity during reservation.
  mutable std::mutex advanced_reserve_mutex;

  // Attempt to reserve advanced workspace described by 'req'. Returns true if
  // the Plan (PlanArena) meets the request (either already satisfied or after
  // invoking arena_resizer). On success the request is cached for future
  // fast-path checks. This is conservative and durable: requests remain cached
  // for the Plan lifetime. The method may call arena_resizer if available.
  bool reserve_advanced_workspace(const AdvancedWorkspaceRequest& req) const {
    // Fast-path: already cached
    {
      std::lock_guard<std::mutex> g(advanced_reserve_mutex);
      for (const auto& r : cached_advanced_requests) {
        if (r == req) return true;
      }
    }

    // Try to satisfy ND transpose capacity first.
    if (req.min_nd_transpose_capacity > arena.nd_transpose_capacity) {
      if (arena_resizer) {
        // Set pending request and ask resizer to grow arena.
        const_cast<Plan*>(this)->pending_nd_capacity = req.min_nd_transpose_capacity;
        arena_resizer(const_cast<PlanArena<T>&>(arena), N, std::max(1, requested_threads), std::max(1, req.min_lane_capacity));
      }
      if (arena.nd_transpose_capacity < req.min_nd_transpose_capacity) return false;
    }

    // Check lane/thread capacity. If neither baseline nor stockham capacities
    // meet the requested lane count, attempt a resize.
    const bool lane_ok = (arena.baseline_lane_capacity >= req.min_lane_capacity) || (arena.stockham_lane_capacity >= req.min_lane_capacity);
    const bool thread_ok = (arena.baseline_thread_capacity >= req.min_thread_capacity) || (arena.stockham_thread_capacity >= req.min_thread_capacity);
    if (!lane_ok || !thread_ok) {
      if (arena_resizer) {
        arena_resizer(const_cast<PlanArena<T>&>(arena), N, std::max(req.min_thread_capacity, requested_threads), std::max(req.min_lane_capacity, arena.baseline_lane_capacity));
      }
      const bool lane_ok2 = (arena.baseline_lane_capacity >= req.min_lane_capacity) || (arena.stockham_lane_capacity >= req.min_lane_capacity);
      const bool thread_ok2 = (arena.baseline_thread_capacity >= req.min_thread_capacity) || (arena.stockham_thread_capacity >= req.min_thread_capacity);
      if (!lane_ok2 || !thread_ok2) return false;
    }

    // Handle special buffer requests (preferred_slot must be >= 0 for now).
    if (!req.special_requests.empty()) {
      if (!arena_resizer) return false; // cannot satisfy special requests without resizer
      // Lock while we update pending_special_capacity
      {
        std::lock_guard<std::mutex> g(advanced_reserve_mutex);
        for (const auto& s : req.special_requests) {
          if (s.preferred_slot < 0 || s.preferred_slot >= PlanArena<T>::kMaxSpecialBuffers) return false;
          const int slot = s.preferred_slot;
          // If arena already has sufficient capacity, nothing to request.
          if (arena.special_capacity[slot] >= s.elements) continue;
          // Otherwise, set pending capacity to requested size (or max of existing pending).
          pending_special_capacity[slot] = std::max(pending_special_capacity[slot], s.elements);
        }
      }
      // Call resizer to attempt to satisfy pending special capacities.
      arena_resizer(const_cast<PlanArena<T>&>(arena), N, std::max(req.min_thread_capacity, requested_threads), std::max(req.min_lane_capacity, arena.baseline_lane_capacity));
      // Verify capacities after resize.
      for (const auto& s : req.special_requests) {
        const int slot = s.preferred_slot;
        if (arena.special_capacity[slot] < s.elements) return false;
      }
    }

    // Success: cache and return true.
    {
      std::lock_guard<std::mutex> g(advanced_reserve_mutex);
      const_cast<std::vector<AdvancedWorkspaceRequest>&>(cached_advanced_requests).push_back(req);
    }
    return true;
  }

  Plan(int n, PlanArena<T>& arena_ref, bool inv=false, bool threads=true, int max_threads=0)
      : N(n), inverse(inv), lgN(0), use_threads(threads),
        requested_threads(max_threads), packet_cols(1), arena(arena_ref) {
#if !defined(EIGFFT_ALLOW_SEQUENTIAL)
    if (!use_threads) {
      throw std::invalid_argument(
          "Sequential execution has been disabled. Define EIGFFT_ALLOW_SEQUENTIAL at build time to permit opt-out.");
    }
#else
    if (!use_threads) {
#if EIGFFT_DEBUG
      std::cout << "Plan constructed with sequential override enabled." << std::endl;
#endif
    }
#endif
#if EIGFFT_DEBUG
    std::cout << "Plan constructor entered, N=" << N << std::endl;
#endif
    if (requested_threads <= 0) {
      requested_threads = Limits::kDefaultRuntimeThreads;
    }
    requested_threads = std::min(requested_threads, Limits::kCompileTimeMaxThreads);

    if (!arena.twiddles || !arena.bitrev) {
      throw std::invalid_argument("PlanArena must provide twiddle and bit-reversal buffers");
    }
    if (arena.twiddle_count < std::max(1, N / 2) || arena.bitrev_count < N) {
      throw std::invalid_argument("PlanArena buffers smaller than required for Plan");
    }

    W = arena.twiddles;
    bitrev = arena.bitrev;

    // power-of-two check
    int t = N;
    while ((t & 1) == 0) { ++lgN; t >>= 1; }
    if ((1 << lgN) != N) throw std::invalid_argument("N must be power of two");

    // twiddles
    const T sgn = inverse ? T(+1) : T(-1);
    const T tau = sgn * T(2 * std::acos(T(-1))) / T(N);
    for (int k = 0; k < N / 2; ++k) {
      const T ang = tau * T(k);
      const T re = static_cast<T>(std::cos(ang));
      const T im = static_cast<T>(std::sin(ang));
      W[k] = Complex(re, im);
    }

    // bit-reversal
    for (int i = 0; i < N; ++i) {
      unsigned x = static_cast<unsigned>(i);
      unsigned r = 0;
      for (int b = 0; b < lgN; ++b) {
        r = (r << 1) | (x & 1u);
        x >>= 1;
      }
      bitrev[i] = static_cast<int>(r);
    }

    if (!arena.baseline_a || !arena.baseline_b || !arena.baseline_columns) {
      throw std::invalid_argument("PlanArena missing baseline workspace buffers");
    }
    if (arena.baseline_thread_capacity <= 0 ||
        arena.baseline_thread_capacity > Limits::kCompileTimeMaxThreads) {
      throw std::invalid_argument("PlanArena baseline_thread_capacity out of range");
    }
    if (arena.baseline_lane_capacity <= 0 ||
        arena.baseline_lane_capacity > Limits::compile_time_max_lane_capacity()) {
      throw std::invalid_argument("PlanArena baseline_lane_capacity out of range");
    }

    if (!arena.stockham_ping || !arena.stockham_pong || !arena.stockham_lane_bases) {
      throw std::invalid_argument("PlanArena missing Stockham scratch buffers");
    }
    if (arena.stockham_thread_capacity <= 0 ||
        arena.stockham_thread_capacity > Limits::kCompileTimeMaxThreads) {
      throw std::invalid_argument("PlanArena stockham_thread_capacity out of range");
    }
    if (arena.stockham_lane_capacity <= 0 ||
        arena.stockham_lane_capacity > Limits::compile_time_max_lane_capacity()) {
      throw std::invalid_argument("PlanArena stockham_lane_capacity out of range");
    }

    requested_threads = std::min(requested_threads, arena.stockham_thread_capacity);
    if (arena.baseline_thread_capacity < requested_threads) {
      throw std::invalid_argument("PlanArena baseline_thread_capacity smaller than requested thread count");
    }

  int packet = static_cast<int>(Eigen::internal::packet_traits<Complex>::size);
  if (packet <= 0) packet = 1;
  const int arena_lane_cap = std::max(1, arena.baseline_lane_capacity);
  packet_cols = std::max(1, std::min(packet, arena_lane_cap));
#if EIGFFT_DEBUG
    std::cout << "Plan using packet_cols=" << packet_cols << std::endl;
#endif

  // Warm the workspace to avoid hot-path checks during the first dispatch.
  workspace.bind(*this);
  const int warm_threads = effective_threads(packet_cols);
  workspace.ensure(warm_threads, packet_cols);
    select_kernel(baseline_kernel());
  // initialize butterfly defaults per algorithm (no allocations in hot path)
  butterfly_default_baseline.radix = ButterflyRadix::Radix2;
  butterfly_default_baseline.method = ButterflyMethod::DIT;
  butterfly_default_baseline.tw_place = TwiddlePlacement::PreRHS;
  butterfly_default_baseline.forward = true;
  butterfly_default_baseline.conjugate_tw = false;
  butterfly_default_baseline.simd_width = 1;

  butterfly_default_stockham = butterfly_default_baseline;

  butterfly_default_external = butterfly_default_baseline;
  }

  ~Plan() {
    release_kernel();
  }

  // Optional: override SIMD coarsening (e.g., 2× packet for fat batches)
  void set_packet_step(int step) {
    tuning.packet_step = step;
  }

  // Enable/disable FTZ/DAZ at call sites
  static void set_ftz_daz(bool on) {
    // #if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    //   _MM_SET_FLUSH_ZERO_MODE(on ? _MM_FLUSH_ZERO_ON : _MM_FLUSH_ZERO_OFF);
    //   _MM_SET_DENORMALS_ZERO_MODE(on ? _MM_DENORMALS_ZERO_ON : _MM_DENORMALS_ZERO_OFF);
    // #else
    (void)on;
    // #endif
  }

  int effective_threads(int batch_cols) const {
  (void)batch_cols;
  if (!use_threads) {
#if defined(EIGFFT_ALLOW_SEQUENTIAL)
    return 1;
#else
    throw std::logic_error(
      "Sequential execution requested at runtime but disabled at build time.");
#endif
  }
  int limit = requested_threads;
#ifdef _OPENMP
  const int runtime_cap = std::max(1, omp_get_max_threads());
#else
  int hw = static_cast<int>(std::thread::hardware_concurrency());
  if (hw <= 0) hw = 1;
  const int runtime_cap = std::max(1, hw);
#endif
  if (limit <= 0) limit = runtime_cap;
  else limit = std::min(limit, runtime_cap);
  limit = std::min(limit, Limits::kCompileTimeMaxThreads);
  limit = std::min(limit, arena.stockham_thread_capacity);
  return std::max(1, limit);
  }

  void ensure_workspace(int threads, int cols) const {
    workspace.ensure(threads, cols);
  }

  void ensure_nd_workspace(Eigen::Index rows, Eigen::Index cols) const {
    workspace.ensure_nd_buffer(rows, cols);
  }

  Complex* transpose_buffer_data() const {
    return workspace.nd_buffer();
  }

  Eigen::Index transpose_rows() const { return workspace.nd_rows; }
  Eigen::Index transpose_cols() const { return workspace.nd_cols; }

  void set_arena_resizer(ArenaResizer r) { arena_resizer = r; }

  void select_kernel(const KernelDescriptor& descriptor) {
    if (kernel_desc_ == &descriptor) return;
    release_kernel();
    kernel_desc_ = &descriptor;
    if (kernel_desc_ && kernel_desc_->create_state) {
      kernel_state_ = kernel_desc_->create_state(*this);
    } else {
      kernel_state_.reset();
    }
  }

  const KernelDescriptor& kernel() const {
    return kernel_desc_ ? *kernel_desc_ : baseline_kernel();
  }

  bool use_kernel(KernelKind kind) {
    if (const KernelDescriptor* desc = find_kernel(kind)) {
      select_kernel(*desc);
      return true;
    }
    return false;
  }

  bool use_kernel(std::string_view name) {
    if (const KernelDescriptor* desc = find_kernel(name)) {
      select_kernel(*desc);
      return true;
    }
    return false;
  }

  bool kernel_realtime_safe() const { return kernel().realtime_safe; }
  KernelAccuracy kernel_accuracy() const { return kernel().accuracy; }

  KernelContext* kernel_state() const {
    return kernel_state_.get();
  }

  static const KernelDescriptor& baseline_kernel();
  static const KernelDescriptor& stockham_kernel();
  static const KernelDescriptor& external_kernel();
  static const std::array<const KernelDescriptor*, 3>& builtin_kernels();

  static constexpr bool has_external_kernel() {
    return detail::kExternalKernelAvailable;
  }

 private:
  void release_kernel() {
    if (kernel_desc_ && kernel_desc_->destroy_state && kernel_state_) {
      kernel_desc_->destroy_state(*this, kernel_state_.get());
    }
    kernel_state_.reset();
    kernel_desc_ = nullptr;
  }

  const KernelDescriptor* find_kernel(KernelKind kind) const {
    switch (kind) {
      case KernelKind::Baseline:
        return &baseline_kernel();
      case KernelKind::Stockham:
        return &stockham_kernel();
      case KernelKind::External:
        if (detail::kExternalKernelAvailable) {
          return &external_kernel();
        }
        return nullptr;
    }
    return nullptr;
  }

  const KernelDescriptor* find_kernel(std::string_view name) const {
    const auto& all = builtin_kernels();
    for (const auto* desc : all) {
      if (desc && desc->name && name == desc->name) {
        return desc;
      }
    }
    return nullptr;
  }
};

namespace detail {

template<class T>
inline void tiled_transpose_colmajor(const std::complex<T>* src, Eigen::Index rows,
                                     Eigen::Index cols, std::complex<T>* dst,
                                     Eigen::Index tile_rows, Eigen::Index tile_cols)
{
  if (rows <= 0 || cols <= 0 || src == nullptr || dst == nullptr) return;
  if (tile_rows <= 0) tile_rows = rows;
  if (tile_cols <= 0) tile_cols = cols;
  const Eigen::Index dest_rows = cols;

  for (Eigen::Index c0 = 0; c0 < cols; c0 += tile_cols) {
    const Eigen::Index width = std::min(tile_cols, cols - c0);
    for (Eigen::Index r0 = 0; r0 < rows; r0 += tile_rows) {
      const Eigen::Index height = std::min(tile_rows, rows - r0);
      for (Eigen::Index j = 0; j < width; ++j) {
        const std::complex<T>* src_col = src + (c0 + j) * rows + r0;
        for (Eigen::Index i = 0; i < height; ++i) {
          const Eigen::Index row = c0 + j;
          const Eigen::Index col = r0 + i;
          dst[row + col * dest_rows] = src_col[i];
        }
      }
    }
  }
  if (snap.enabled() && layout.batch_size > 0) {
    const auto* col0 = layout.base + 0 * layout.batch_stride; // batch 0
    snap.copy_lane0(stage_idx, col0, /*axis_stride*/ (int)layout.axis_stride);
  }
}

// Shared input validation for FFT axis execution (hot-path small inline)
template<class T>
inline void validate_fft_axis(const Plan<T>& P, const AxisLayout<T>& layout) {
  if (!layout.valid())
    throw std::invalid_argument("AxisLayout must reference valid data.");
  if (layout.axis_size != P.N)
    throw std::invalid_argument("AxisLayout axis_size must match Plan::N.");
}

// Prepare Plan workspace and compute lane/stride values used by execution paths.
template<class T>
inline void prepare_plan_workspace(const Plan<T>& P, const AxisLayout<T>& layout,
                                   int &out_B, int &out_threads,
                                   int &out_active_lane_cols, int &out_lane_cols,
                                   Eigen::Index &out_axis_stride, Eigen::Index &out_batch_stride,
                                   const typename Plan<T>::AdvancedWorkspaceRequest* adv_req = nullptr,
                                   bool* out_has_advanced_alloc = nullptr)
{
  out_B = static_cast<int>(layout.batch_size);
  out_threads = P.effective_threads(out_B);
  const int requested_lanes = (P.tuning.packet_step > 0) ? P.tuning.packet_step : P.packet_cols;
  out_lane_cols = std::max(1, requested_lanes);
  P.ensure_workspace(out_threads, out_lane_cols);
  out_active_lane_cols = std::max(1, std::min(out_lane_cols, P.workspace.capacity));
  if (P.tuning.force_ftz_daz) Plan<T>::set_ftz_daz(true);
  out_axis_stride = layout.axis_stride;
  out_batch_stride = layout.batch_stride;
  if (adv_req) {
    const bool ok = P.reserve_advanced_workspace(*adv_req);
    if (out_has_advanced_alloc) *out_has_advanced_alloc = ok;
  } else if (out_has_advanced_alloc) {
    *out_has_advanced_alloc = false;
  }
}

template<class T>
inline void baseline_execute_axis(const Plan<T>& P, const AxisLayout<T>& layout, KernelContext* ctx)
{
  (void)ctx;
  using Complex = typename Plan<T>::Complex;

  const int N = P.N;
  const int stages = P.lgN;
  validate_fft_axis(P, layout);
  const detail::TwiddleMetadataWriter<T> metadata(layout, stages);
  metadata.record_plan_twiddles(P);
  const detail::LayoutMetadataWriter<T> layout_writer(layout, static_cast<std::size_t>(N));
  if (layout_writer.enabled()) {
    for (int pos = 0; pos < N; ++pos) {
      layout_writer.set(static_cast<std::size_t>(pos), P.bitrev[pos]);
    }
  }
  const detail::StageSnapshotWriter<T> snap(layout, static_cast<std::size_t>(N), stages);
  const detail::StageParamsWriter<T> params(layout, stages);
  const detail::ButterflyPairsWriter<T> pairs(layout, static_cast<std::size_t>(N), stages);
  const detail::StageInvariantWriter<T> invariant(layout, stages);
  const detail::TwiddleIndexWriter<T> twmap(layout, static_cast<std::size_t>(N), stages);
  // Initialize invariant buffer to zero if enabled
  if (invariant.enabled()) {
    for (int s = 0; s < stages; ++s) invariant.set(s, T(0), T(0));
  }
  int B, threads, lane_cols, active_lane_cols;
  Eigen::Index axis_stride, batch_stride;
  prepare_plan_workspace(P, layout, B, threads, active_lane_cols, lane_cols, axis_stride, batch_stride);
  const bool parallel_enabled = P.use_threads && threads > 1;

  const int chunk_count = (B + active_lane_cols - 1) / active_lane_cols;
  // Print actual stride/ptr info used by Baseline (helpful for batch-pitch mismatch)
  std::cout << "[baseline] axis_stride=" << axis_stride
            << " batch_stride=" << batch_stride
            << " lane_cols=" << active_lane_cols << std::endl;
  for (int bi = 0; bi < std::min(3, B); ++bi) {
    const void* ptr = static_cast<const void*>(layout.base + static_cast<std::size_t>(bi) * (std::size_t)batch_stride);
    std::cout << "[baseline] base[" << bi << "]=" << ptr << std::endl;
  }
  const int* bitrev = P.bitrev;

  bool permuted_in_parallel = false;
#ifdef _OPENMP
  if (parallel_enabled) {
    permuted_in_parallel = true;
#pragma omp parallel num_threads(threads)
    {
      const int tid = omp_get_thread_num();
      Complex** columns = P.workspace.column_row(tid);
#pragma omp for schedule(static)
      for (int chunk = 0; chunk < chunk_count; ++chunk) {
        const int col = chunk * active_lane_cols;
        const int width = std::min(active_lane_cols, B - col);
        if (width <= 0) continue;
        for (int lane = 0; lane < width; ++lane) {
          const Eigen::Index batch_index = Eigen::Index(col + lane);
          columns[lane] = layout.base + batch_index * batch_stride;
        }
        for (int i = 0; i < N; ++i) {
          const int src = bitrev[i];
          if (src <= i) continue;
          const Eigen::Index dst_offset = Eigen::Index(i) * axis_stride;
          const Eigen::Index src_offset = Eigen::Index(src) * axis_stride;
          for (int lane = 0; lane < width; ++lane) {
            Complex* column_ptr = columns[lane];
            std::swap(column_ptr[dst_offset], column_ptr[src_offset]);
          }
        }
      }
    }
  }
#endif
  if (!permuted_in_parallel) {
    Complex** columns = P.workspace.column_row(0);
    for (int chunk = 0; chunk < chunk_count; ++chunk) {
      const int col = chunk * active_lane_cols;
      const int width = std::min(active_lane_cols, B - col);
      if (width <= 0) continue;
      for (int lane = 0; lane < width; ++lane) {
        const Eigen::Index batch_index = Eigen::Index(col + lane);
          columns[lane] = layout.base + batch_index * batch_stride;
      }
      for (int i = 0; i < N; ++i) {
        const int src = bitrev[i];
        if (src <= i) continue;
        const Eigen::Index dst_offset = Eigen::Index(i) * axis_stride;
        const Eigen::Index src_offset = Eigen::Index(src) * axis_stride;
        for (int lane = 0; lane < width; ++lane) {
            Complex* column_ptr = columns[lane];
          std::swap(column_ptr[dst_offset], column_ptr[src_offset]);
        }
      }
    }
  }

  for (int len = 2, stage_idx = 0; len <= N; len <<= 1, ++stage_idx) {
  const int half = len >> 1;
  const int step = N / len;
  const int blocks = N / len;
  if (params.enabled()) params.set(stage_idx, len, step);

    bool butterflies_in_parallel = false;
#ifdef _OPENMP
    if (parallel_enabled) {
      butterflies_in_parallel = true;
#pragma omp parallel num_threads(threads)
      {
        const int tid = omp_get_thread_num();
        Complex* a_ptr = P.workspace.row_a(tid);
        Complex* b_ptr = P.workspace.row_b(tid);
        Complex** columns = P.workspace.column_row(tid);
        const int work_items = half * blocks * chunk_count;
        if (work_items > 0) {
#pragma omp for schedule(static)
          for (int item = 0; item < work_items; ++item) {
            int tmp = item;
            const int chunk = tmp % chunk_count;
            tmp /= chunk_count;
            const int block = tmp % blocks;
            const int k = tmp / blocks;
            const int col = chunk * active_lane_cols;
            const int width = std::min(active_lane_cols, B - col);
            if (width <= 0) continue;
            for (int lane = 0; lane < width; ++lane) {
              const Eigen::Index batch_index = Eigen::Index(col + lane);
              columns[lane] = layout.base + batch_index * batch_stride;
            }
            const int a_index = block * len + k;
            const int b_index = a_index + half;
            const Eigen::Index a_offset = Eigen::Index(a_index) * axis_stride;
            const Eigen::Index b_offset = Eigen::Index(b_index) * axis_stride;
            const Complex w = P.W[k * step];
            Complex* a_in_local = P.workspace.row_a(threads > 1 ? tid : 0);
            Complex* b_in_local = P.workspace.row_b(threads > 1 ? tid : 0);
            for (int lane = 0; lane < width; ++lane) {
              Complex* column_ptr = columns[lane];
              a_ptr[lane] = column_ptr[a_offset];
              b_ptr[lane] = column_ptr[b_offset];
              a_in_local[lane] = a_ptr[lane];
              b_in_local[lane] = b_ptr[lane];
            }
            detail::ButterflyKernel<T>::apply(a_ptr, b_ptr, width, w, &P.butterfly_default_baseline);
            for (int lane = 0; lane < width; ++lane) {
              Complex* column_ptr = columns[lane];
              column_ptr[a_offset] = a_ptr[lane];
              column_ptr[b_offset] = b_ptr[lane];
              const int batch_index = col + lane;
              if (snap.enabled() && batch_index == 0) {
                const Complex y0 = column_ptr[a_offset];
                const Complex y1 = column_ptr[b_offset];
                snap.set(stage_idx, a_index, y0);
                snap.set(stage_idx, b_index, y1);
                if (twmap.enabled()) {
                  const int tw_index = k * step;
                  twmap.set(stage_idx, a_index, tw_index);
                  twmap.set(stage_idx, b_index, tw_index);
                }
                if (invariant.enabled()) {
                  const Complex a_in = a_in_local[static_cast<size_t>(lane)];
                  const Complex b_in = b_in_local[static_cast<size_t>(lane)];
                  const T r0 = std::abs((y0 + y1) - T(2) * a_in);
                  // energy invariant: (|y0|^2 + |y1|^2) - 2*(|a|^2 + |b|^2)
                  const T e = (std::norm(y0) + std::norm(y1)) - T(2) * (std::norm(a_in) + std::norm(b_in));
                  const T abs_e = std::abs(e);
                  const std::complex<T> prev = invariant.buf ? invariant.buf[stage_idx] : std::complex<T>(T(0), T(0));
                  const T prev_r0 = prev.real();
                  const T prev_e = prev.imag();
                  const T new_r0 = std::max(prev_r0, r0);
                  const T new_e = std::max(prev_e, abs_e);
                  invariant.buf[stage_idx] = std::complex<T>(new_r0, new_e);
                }
              }
              if (pairs.enabled() && batch_index == 0) {
                pairs.set(stage_idx, a_index, a_index, b_index);
                pairs.set(stage_idx, b_index, a_index, b_index);
              }
            }
          }
        }
      }
    }
#endif
    if (!butterflies_in_parallel) {
      Complex* a_ptr = P.workspace.row_a(0);
      Complex* b_ptr = P.workspace.row_b(0);
      Complex** columns = P.workspace.column_row(0);
      for (int k = 0; k < half; ++k) {
        const Complex w = P.W[k * step];
        for (int block = 0; block < blocks; ++block) {
          const int base = block * len;
          const int a_index = base + k;
          const int b_index = a_index + half;
          const Eigen::Index a_offset = Eigen::Index(a_index) * axis_stride;
          const Eigen::Index b_offset = Eigen::Index(b_index) * axis_stride;
          for (int chunk = 0; chunk < chunk_count; ++chunk) {
            const int col = chunk * active_lane_cols;
            const int width = std::min(active_lane_cols, B - col);
            if (width <= 0) continue;
            std::vector<Complex> a_in_local(static_cast<size_t>(width));
            std::vector<Complex> b_in_local(static_cast<size_t>(width));
            for (int lane = 0; lane < width; ++lane) {
              const Eigen::Index batch_index = Eigen::Index(col + lane);
              columns[lane] = layout.base + batch_index * batch_stride;
            }
            for (int lane = 0; lane < width; ++lane) {
              Complex* column_ptr = columns[lane];
              a_ptr[lane] = column_ptr[a_offset];
              b_ptr[lane] = column_ptr[b_offset];
              a_in_local[static_cast<size_t>(lane)] = a_ptr[lane];
              b_in_local[static_cast<size_t>(lane)] = b_ptr[lane];
            }
            detail::ButterflyKernel<T>::apply(a_ptr, b_ptr, width, w, &P.butterfly_default_baseline);
            for (int lane = 0; lane < width; ++lane) {
              Complex* column_ptr = columns[lane];
              column_ptr[a_offset] = a_ptr[lane];
              column_ptr[b_offset] = b_ptr[lane];
              const int batch_index = col + lane;
              if (snap.enabled() && batch_index == 0) {
                const Complex y0 = column_ptr[a_offset];
                const Complex y1 = column_ptr[b_offset];
                snap.set(stage_idx, a_index, y0);
                snap.set(stage_idx, b_index, y1);
                if (twmap.enabled()) {
                  const int tw_index = k * step;
                  twmap.set(stage_idx, a_index, tw_index);
                  twmap.set(stage_idx, b_index, tw_index);
                }
                if (invariant.enabled()) {
                  const Complex a_in = a_in_local[static_cast<size_t>(lane)];
                  const Complex b_in = b_in_local[static_cast<size_t>(lane)];
                  const T r0 = std::abs((y0 + y1) - T(2) * a_in);
                  // energy invariant: (|y0|^2 + |y1|^2) - 2*(|a|^2 + |b|^2)
                  const T e = (std::norm(y0) + std::norm(y1)) - T(2) * (std::norm(a_in) + std::norm(b_in));
                  const T abs_e = std::abs(e);
                  const std::complex<T> prev = invariant.buf ? invariant.buf[stage_idx] : std::complex<T>(T(0), T(0));
                  const T prev_r0 = prev.real();
                  const T prev_e = prev.imag();
                  const T new_r0 = std::max(prev_r0, r0);
                  const T new_e = std::max(prev_e, abs_e);
                  invariant.buf[stage_idx] = std::complex<T>(new_r0, new_e);
                }
              }
              if (pairs.enabled() && batch_index == 0) {
                pairs.set(stage_idx, a_index, a_index, b_index);
                pairs.set(stage_idx, b_index, a_index, b_index);
              }
            }
          }
        }
      }
    }
    // Baseline snapshots, pair records, twiddle-index, and invariants are captured at write-time above.
    if (invariant.enabled()) {
      // Nothing to do here because we updated per-write into the buffer; keep for clarity.
    }
  }

  if (P.inverse) {
    const T scale = T(1) / T(N);
    for (int col = 0; col < B; ++col) {
      const Eigen::Index batch_index = Eigen::Index(col);
      Complex* column_ptr = layout.base + batch_index * batch_stride;
      for (int i = 0; i < N; ++i) {
        column_ptr[Eigen::Index(i) * axis_stride] *= scale;
      }
    }
  }
  if (P.tuning.force_ftz_daz) Plan<T>::set_ftz_daz(false);
}

} // namespace detail

template<class T>
inline void fft_apply_axis(const Plan<T>& P, const AxisLayout<T>& layout)
{
  if (!layout.valid())
    throw std::invalid_argument("AxisLayout must reference valid data.");
  if (layout.axis_size != P.N)
    throw std::invalid_argument("AxisLayout axis_size must match Plan::N.");

  const auto& descriptor = P.kernel();
  if (descriptor.execute_axis) {
    descriptor.execute_axis(P, layout, P.kernel_state());
  } else {
    detail::baseline_execute_axis(P, layout, P.kernel_state());
  }
}

template<class T>
inline void dispatch_fft(Plan<T>& P, const typename Plan<T>::KernelDescriptor& descriptor,
                         const AxisLayout<T>& layout)
{
  P.select_kernel(descriptor);
  fft_apply_axis(P, layout);
}

template<class T>
inline void dispatch_fft(Plan<T>& P, KernelKind kind, const AxisLayout<T>& layout)
{
  switch (kind) {
    case KernelKind::Baseline:
      dispatch_fft(P, Plan<T>::baseline_kernel(), layout);
      break;
    case KernelKind::Stockham:
      dispatch_fft(P, Plan<T>::stockham_kernel(), layout);
      break;
    case KernelKind::External:
      dispatch_fft(P, Plan<T>::external_kernel(), layout);
      break;
    default:
      throw std::invalid_argument("Unsupported KernelKind.");
  }
}

template<class T>
inline void fft_inplace_batched_with_metadata(
    Eigen::Ref<Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> X,
    const Plan<T>& P,
    const MetadataRequest<T>* metadata_requests,
    int metadata_request_count)
{
  AxisLayout<T> layout;
  layout.base = X.data();
  layout.axis_size = P.N;
  layout.batch_size = static_cast<Eigen::Index>(X.cols());
  layout.axis_stride = 1;
  layout.batch_stride = static_cast<Eigen::Index>(X.rows());
  layout.metadata_requests = metadata_requests;
  layout.metadata_request_count = metadata_request_count;
  fft_apply_axis(P, layout);
}

template<class T>
inline void fft_inplace_batched(Eigen::Ref<Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> X,
                                const Plan<T>& P)
{
  fft_inplace_batched_with_metadata<T>(X, P, nullptr, 0);
}

template<class T>
inline void fft_inplace_2d(Eigen::Ref<Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> X,
                           const Plan<T>& axis0_plan,
                           const Plan<T>& axis1_plan,
                           Eigen::Index tile_rows = 64,
                           Eigen::Index tile_cols = 128)
{
  using Complex = typename Plan<T>::Complex;
  const Eigen::Index rows = X.rows();
  const Eigen::Index cols = X.cols();
  if (axis0_plan.N != rows) {
    throw std::invalid_argument("axis0_plan length must match matrix rows.");
  }
  if (axis1_plan.N != cols) {
    throw std::invalid_argument("axis1_plan length must match matrix cols.");
  }

  AxisLayout<T> axis0_layout;
  axis0_layout.base = X.data();
  axis0_layout.axis_size = rows;
  axis0_layout.batch_size = cols;
  axis0_layout.axis_stride = 1;
  axis0_layout.batch_stride = rows;
  fft_apply_axis(axis0_plan, axis0_layout);

  axis0_plan.ensure_nd_workspace(rows, cols);
  Complex* scratch = axis0_plan.transpose_buffer_data();
  detail::tiled_transpose_colmajor<T>(X.data(), rows, cols, scratch, tile_rows, tile_cols);

  AxisLayout<T> axis1_layout;
  axis1_layout.base = scratch;
  axis1_layout.axis_size = cols;
  axis1_layout.batch_size = rows;
  axis1_layout.axis_stride = 1;
  axis1_layout.batch_stride = cols;
  fft_apply_axis(axis1_plan, axis1_layout);

  detail::tiled_transpose_colmajor<T>(scratch, cols, rows, X.data(), tile_rows, tile_cols);
}

namespace detail {

template<class T>
std::unique_ptr<KernelContext> baseline_create_state(Plan<T>&) {
  return nullptr;
}

template<class T>
void baseline_destroy_state(Plan<T>&, KernelContext*) {}

template<class T>
void stockham_execute_axis(const Plan<T>& P, const AxisLayout<T>& layout, KernelContext* ctx) {
  using Complex = typename Plan<T>::Complex;
  validate_fft_axis(P, layout);
  auto* state = static_cast<StockhamState<T>*>(ctx);
  if (!state) {
    baseline_execute_axis(P, layout, nullptr);
    return;
  }

  const int N = P.N;
  const int stages = P.lgN;
  const detail::LayoutMetadataWriter<T> layout_writer(layout, static_cast<std::size_t>(N));
  if (layout_writer.enabled()) {
    std::vector<int> current_map(N);
    std::vector<int> next_map(N);
    for (int i = 0; i < N; ++i) current_map[i] = i;
    for (int stage = 0; stage < stages; ++stage) {
      
      const int m = 1 << stage;
      const int segments = N >> (stage + 1);  // N / (2 * m)
      const int halfN = N >> 1;
      for (int j = 0; j < m; ++j) {
        for (int g = 0; g < segments; ++g) {
          const int idx0 = (2 * j) * segments + g;
          const int idx1 = idx0 + segments;
          const int out0 = j * segments + g;
          const int out1 = out0 + halfN;
          next_map[out0] = current_map[idx0];
          next_map[out1] = current_map[idx1];
        }
      }
      current_map.swap(next_map);
    }
    for (int pos = 0; pos < N; ++pos) {
      layout_writer.set(static_cast<std::size_t>(pos), current_map[pos]);
    }
  }
  const detail::StageSnapshotWriter<T> snap(layout, static_cast<std::size_t>(N), stages);
  const detail::StageParamsWriter<T> params(layout, stages);
  const detail::ButterflyPairsWriter<T> pairs(layout, static_cast<std::size_t>(N), stages);
  const detail::StageInvariantWriter<T> invariant(layout, stages);
  const detail::TwiddleIndexWriter<T> twmap(layout, static_cast<std::size_t>(N), stages);
  const int B = static_cast<int>(layout.batch_size);
  if (B <= 0) {
    if (P.tuning.force_ftz_daz) Plan<T>::set_ftz_daz(false);
    return;
  }

  const detail::TwiddleMetadataWriter<T> metadata(layout, stages);
  metadata.record_plan_twiddles(P);

#if EIGFFT_TRACE_STOCKHAM
  static std::atomic<int> dispatch_seq{0};
  const int dispatch_id = dispatch_seq.fetch_add(1, std::memory_order_relaxed);
  std::cout << "[stockham] dispatch=" << dispatch_id
            << " plan=" << &P
            << " state=" << state
            << " layout.base=" << static_cast<const void*>(layout.base)
            << " N=" << N
            << " B=" << B
            << std::endl;
#endif

  const int batch_size_int = static_cast<int>(layout.batch_size);
  const int adv_threads = std::max(1, P.effective_threads(batch_size_int));
  const int adv_lane_request =
      std::max(1, (P.tuning.packet_step > 0) ? P.tuning.packet_step : P.packet_cols);
  int B2 = 0;
  int desired_threads = adv_threads;
  int desired_lanes = adv_lane_request;
  int active_lane_cols2 = 0;
  Eigen::Index axis_stride2 = layout.axis_stride;
  Eigen::Index batch_stride2 = layout.batch_stride;
  // Build an algorithm-agnostic advanced workspace request and pass it into
  // prepare_plan_workspace so the Plan can attempt to honor the reservation
  // as part of workspace preparation (may call arena_resizer).
  typename Plan<T>::AdvancedWorkspaceRequest adv_req{};
  adv_req.min_lane_capacity = adv_lane_request;
  adv_req.min_thread_capacity = adv_threads;
  bool adv_reserved = false;
  prepare_plan_workspace(P, layout, B2, desired_threads, active_lane_cols2, desired_lanes, axis_stride2, batch_stride2, &adv_req, &adv_reserved);
  const int workspace_threads = std::max(1, P.workspace.threads);
  const int workspace_lanes = std::max(1, P.workspace.capacity);

  state->ensure_threads(workspace_threads);
  state->ensure_lane_capacity(workspace_lanes);

  const int stockham_capacity = std::max(1, state->arena.stockham_lane_capacity);
  const int baseline_capacity = std::max(1, state->arena.baseline_lane_capacity);
  const int lane_capacity = stockham_capacity;
  const int lane_cols = std::max(1, std::min({desired_lanes, workspace_lanes, lane_capacity, baseline_capacity}));
#if EIGFFT_TRACE_STOCKHAM
  std::cout << "[stockham] lanes: requested=" << requested_lanes
            << " packet_cols=" << P.packet_cols
            << " lane_cols(final)=" << lane_cols
            << " capacity(stockham/baseline)=" << stockham_capacity << "/" << baseline_capacity
            << std::endl;
#endif
  const Eigen::Index axis_stride = layout.axis_stride;
  const Eigen::Index batch_stride = layout.batch_stride;
#if 1
  // Print Stockham stride/ptr info (single-shot)
  std::cout << "[stockham] axis_stride=" << axis_stride
            << " batch_stride=" << batch_stride
            << " lane_capacity=" << lane_capacity
            << " lane_cols(final)=" << lane_cols << std::endl;
  for (int bi = 0; bi < std::min(3, static_cast<int>(layout.batch_size)); ++bi) {
    const void* ptr = static_cast<const void*>(layout.base + static_cast<std::size_t>(bi) * (std::size_t)batch_stride);
    std::cout << "[stockham] base[" << bi << "]=" << ptr << std::endl;
  }
#endif
#if EIGFFT_TRACE_STOCKHAM
  std::cout << "[stockham] dispatch=" << dispatch_id
            << " desired_threads=" << desired_threads
            << " pool_size=" << state->pool.size()
            << " lane_cols=" << lane_cols
            << " lane_capacity=" << lane_capacity
            << " axis_stride=" << axis_stride
            << " batch_stride=" << batch_stride
            << std::endl;
  std::cout << "[stockham] arena ping=" << static_cast<const void*>(state->arena.stockham_ping)
            << " pong=" << static_cast<const void*>(state->arena.stockham_pong)
            << " lane_bases=" << static_cast<const void*>(state->arena.stockham_lane_bases)
            << " thread_capacity=" << state->arena.stockham_thread_capacity
            << " lane_capacity=" << state->arena.stockham_lane_capacity
            << std::endl;
#endif

  if (P.tuning.force_ftz_daz) Plan<T>::set_ftz_daz(true);

  auto process_chunk = [&](size_t start, size_t end, int worker_id) {
    if (start >= static_cast<size_t>(B)) return;
    const int width = static_cast<int>(end - start);
    if (width <= 0) return;
    if (width > lane_cols) {
      std::ostringstream oss;
      oss << "Stockham chunk width exceeds lane allocation: width=" << width
          << " lane_cols=" << lane_cols
          << " start=" << start << " end=" << end
          << " batch=" << B;
      throw std::runtime_error(oss.str());
    }
    // std::cerr << "[stockham debug] chunk start=" << start << " end=" << end
    //           << " worker=" << worker_id << " width=" << width << std::endl;
  #if EIGFFT_TRACE_STOCKHAM
    if (start == 0 && worker_id == 0) {
      std::cout << "[stockham] first chunk width=" << width
                << " threads=" << state->pool.size()
                << " lane_capacity=" << lane_capacity << std::endl;
    }
    #endif
    size_t slot = static_cast<size_t>(worker_id);
    if (slot >= static_cast<size_t>(state->arena.stockham_thread_capacity)) {
      std::ostringstream oss;
      oss << "Stockham worker slot exceeds arena thread capacity: worker_id=" << worker_id
          << " capacity=" << state->arena.stockham_thread_capacity
          << " pool=" << state->pool.size();
      throw std::runtime_error(oss.str());
    }
  const size_t lane_stride = static_cast<size_t>(lane_capacity);
    const size_t buffer_stride = state->thread_stride();
    Complex* stage_in = state->arena.stockham_ping + slot * buffer_stride;
    Complex* stage_out = state->arena.stockham_pong + slot * buffer_stride;
    std::ptrdiff_t* lane_bases = state->arena.stockham_lane_bases + slot * lane_stride;
    const ptrdiff_t max_offset = (layout.batch_size > 0 && layout.axis_size > 0)
                                     ? (ptrdiff_t(layout.batch_size - 1) *
                                            ptrdiff_t(layout.batch_stride) +
                                        ptrdiff_t(layout.axis_size - 1) *
                                            ptrdiff_t(layout.axis_stride))
                                     : 0;
    for (int lane = 0; lane < width; ++lane) {
      const ptrdiff_t base_idx = ptrdiff_t(start + lane) * ptrdiff_t(batch_stride);
      lane_bases[static_cast<size_t>(lane)] = base_idx;
      if (base_idx < 0 || base_idx > max_offset) {
        std::cerr << "[stockham debug] batch base out of bounds: lane=" << lane
                  << " start=" << start << " base_idx=" << base_idx
                  << " max_offset=" << max_offset << std::endl;
        return;
      }
    }
    const int safe_width = std::min(width, lane_cols);
    #if EIGFFT_TRACE_STOCKHAM
    if (start == 0 && worker_id == 0) {
      std::cout << "[stockham] load phase start" << std::endl;
    }
    #endif
    for (int i = 0; i < N; ++i) {
      const ptrdiff_t off = ptrdiff_t(i) * ptrdiff_t(axis_stride);
  Complex* dest = stage_in + i * lane_capacity;
      for (int lane = 0; lane < safe_width; ++lane) {
        const ptrdiff_t base_idx = lane_bases[static_cast<size_t>(lane)];
        const ptrdiff_t idx = base_idx + off;
        if (idx < 0 || idx > max_offset) {
          std::cerr << "[stockham] load OOB: lane=" << lane
                    << " i=" << i << " idx=" << idx
                    << " max=" << max_offset << std::endl;
          return;
        }
        dest[lane] = layout.base[idx];
      }
    }
    #if EIGFFT_TRACE_STOCKHAM
    if (start == 0 && worker_id == 0) {
      std::cout << "[stockham] load phase complete" << std::endl;
    }
    #endif

    Complex* in = stage_in;
    Complex* out = stage_out;
    #if EIGFFT_TRACE_STOCKHAM
    if (start == 0 && worker_id == 0) {
      std::cout << "[stockham] transform loop start" << std::endl;
    }
    #endif
    for (int stage = 0; stage < stages; ++stage) {
      const int m = 1 << stage;
      const int distance = m << 1;
      const int segments = N >> (stage + 1);  // N / (2 * m)
      const int halfN = N >> 1;
      const int tw_step = segments;  // == N / distance
      if (params.enabled()) {
        params.set(stage, distance, tw_step);
      }
      for (int j = 0; j < m; ++j) {
        const int tw_index = j * tw_step;
        const Complex w = P.W[tw_index];
        for (int g = 0; g < segments; ++g) {
          const int idx0 = (2 * j) * segments + g;
          const int idx1 = idx0 + segments;
          const int out0_idx = j * segments + g;
          const int out1_idx = out0_idx + halfN;
          const Complex* src0 = in + idx0 * lane_capacity;
          const Complex* src1 = in + idx1 * lane_capacity;
          Complex* dst0 = out + out0_idx * lane_capacity;
          Complex* dst1 = out + out1_idx * lane_capacity;
          // Use the Plan-level stockham config. If the user requested
          // in-place emulation (`prefer_inplace`), we provide preallocated
          // per-worker scratch buffers from the PlanArena baseline buffers
          // (no heap). Otherwise call the native scatter path.
          ButterflyScatter<T>::apply(src0, src1, dst0, dst1, safe_width, w, &P.butterfly_default_stockham);
          if (pairs.enabled()) {
            pairs.set(stage, out0_idx, idx0, idx1);
            pairs.set(stage, out1_idx, idx0, idx1);
          }
          if (twmap.enabled()) {
            twmap.set(stage, out0_idx, tw_index);
            twmap.set(stage, out1_idx, tw_index);
          }
          if (invariant.enabled()) {
            const Complex a_in = src0[0];
            const Complex b_in = src1[0];
            const Complex y0 = dst0[0];
            const Complex y1 = dst1[0];
            const T r0 = std::abs((y0 + y1) - T(2) * a_in);
            // energy invariant: (|y0|^2 + |y1|^2) - 2*(|a|^2 + |b|^2)
            const T e = (std::norm(y0) + std::norm(y1)) - T(2) * (std::norm(a_in) + std::norm(b_in));
            const T abs_e = std::abs(e);
            const std::complex<T> prev = invariant.buf ? invariant.buf[stage] : std::complex<T>(T(0), T(0));
            const T prev_r0 = prev.real();
            const T prev_e = prev.imag();
            const T new_r0 = std::max(prev_r0, r0);
            const T new_e = std::max(prev_e, abs_e);
            invariant.buf[stage] = std::complex<T>(new_r0, new_e);
          }
      }
    }
      if (snap.enabled() && start == 0 && safe_width > 0) {
        const ptrdiff_t base_idx = lane_bases[0];
        for (int i = 0; i < N; ++i) {
          const ptrdiff_t idx = base_idx + ptrdiff_t(i) * ptrdiff_t(axis_stride);
          snap.set(stage, i, layout.base[idx]);
        }
      }
      std::swap(in, out);
    }
    #if EIGFFT_TRACE_STOCKHAM
    if (start == 0 && worker_id == 0) {
      std::cout << "[stockham] transform loop complete" << std::endl;
    }
    #endif

    const Complex* final_buf = (stages % 2 == 0) ? stage_in : stage_out;
    if (P.inverse) {
      const T scale = T(1) / T(N);
      #if EIGFFT_TRACE_STOCKHAM
      if (start == 0 && worker_id == 0) {
        std::cout << "[stockham] store inverse phase start" << std::endl;
      }
      #endif
      for (int i = 0; i < N; ++i) {
        const ptrdiff_t off = ptrdiff_t(i) * ptrdiff_t(axis_stride);
  const Complex* src = final_buf + i * lane_capacity;
        for (int lane = 0; lane < safe_width; ++lane) {
          const ptrdiff_t base_idx = lane_bases[static_cast<size_t>(lane)];
          const ptrdiff_t idx = base_idx + off;
          if (idx < 0 || idx > max_offset) {
            std::cerr << "[stockham] store OOB(inv): lane=" << lane
                      << " i=" << i << " idx=" << idx
                      << " max=" << max_offset << std::endl;
            return;
          }
          layout.base[idx] = src[lane] * scale;
        }
      }
    } else {
      #if EIGFFT_TRACE_STOCKHAM
      if (start == 0 && worker_id == 0) {
        std::cout << "[stockham] store phase start" << std::endl;
      }
      #endif
      for (int i = 0; i < N; ++i) {
        const ptrdiff_t off = ptrdiff_t(i) * ptrdiff_t(axis_stride);
const Complex* src = final_buf + i * lane_capacity;
        for (int lane = 0; lane < safe_width; ++lane) {
          const ptrdiff_t base_idx = lane_bases[static_cast<size_t>(lane)];
          const ptrdiff_t idx = base_idx + off;
          if (idx < 0 || idx > max_offset) {
            std::cerr << "[stockham] store OOB: lane=" << lane
                      << " i=" << i << " idx=" << idx
                      << " max=" << max_offset << std::endl;
            return;
          }
          layout.base[idx] = src[lane];
        }
      }
    }
    #if EIGFFT_TRACE_STOCKHAM
    if (start == 0 && worker_id == 0) {
      std::cout << "[stockham] store phase complete" << std::endl;
    }
    #endif
  };

  state->pool.parallel_for(static_cast<size_t>(B), static_cast<size_t>(lane_cols), process_chunk);

#if EIGFFT_TRACE_STOCKHAM
  std::cout << "[stockham] dispatch=" << dispatch_id << " parallel_for complete" << std::endl;
#endif

  if (P.tuning.force_ftz_daz) Plan<T>::set_ftz_daz(false);
}

template<class T>
std::unique_ptr<KernelContext> stockham_create_state(Plan<T>& plan) {
  return std::unique_ptr<KernelContext>(new StockhamState<T>(plan, plan.arena));
}

template<class T>
void stockham_destroy_state(Plan<T>&, KernelContext*) {}

template<class T>
void external_execute_axis(const Plan<T>&, const AxisLayout<T>&, KernelContext*) {
  throw std::logic_error("External FFT kernel requested but no implementation supplied.");
}

template<class T>
std::unique_ptr<KernelContext> external_create_state(Plan<T>&) {
  return nullptr;
}

template<class T>
void external_destroy_state(Plan<T>&, KernelContext*) {}

} // namespace detail

template<class T>
const typename Plan<T>::KernelDescriptor& Plan<T>::baseline_kernel() {
  static const KernelDescriptor desc{
      KernelKind::Baseline,
      "baseline-cooleytukey",
      KernelAccuracy::Default,
      true,
      detail::baseline_create_state<T>,
      detail::baseline_execute_axis<T>,
      detail::baseline_destroy_state<T>
  };
  return desc;
}

template<class T>
const typename Plan<T>::KernelDescriptor& Plan<T>::stockham_kernel() {
  static const KernelDescriptor desc{
      KernelKind::Stockham,
      "stockham-autosort",
      KernelAccuracy::Default,
      true,
      detail::stockham_create_state<T>,
      detail::stockham_execute_axis<T>,
      detail::stockham_destroy_state<T>
  };
  return desc;
}

template<class T>
const typename Plan<T>::KernelDescriptor& Plan<T>::external_kernel() {
  static const KernelDescriptor desc{
      KernelKind::External,
      "external-provider",
      KernelAccuracy::HighPrecision,
      false,
      detail::kExternalKernelAvailable ? detail::external_create_state<T> : nullptr,
      detail::kExternalKernelAvailable ? detail::external_execute_axis<T> : nullptr,
      detail::kExternalKernelAvailable ? detail::external_destroy_state<T> : nullptr
  };
  return desc;
}

template<class T>
const std::array<const typename Plan<T>::KernelDescriptor*, 3>& Plan<T>::builtin_kernels() {
  static const std::array<const KernelDescriptor*, 3> list{
      &baseline_kernel(), &stockham_kernel(),
      detail::kExternalKernelAvailable ? &external_kernel() : nullptr};
  return list;
}

template<class T>
struct PlanProviderSelector {
  static typename detail::ButterflyRegistry<T>::ProviderInfo select(bool require_scatter, int min_simd_width, ButterflyMethod method_hint) {
    return detail::ButterflyRegistry<T>::negotiate(require_scatter, min_simd_width, method_hint);
  }
};

} // namespace eigfft
