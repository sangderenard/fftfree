#ifndef EIGFFT_DEBUG
#define EIGFFT_DEBUG 0
#endif

// eigen_fft.hpp (header-only)
#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <complex>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <atomic>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <vector>

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
    shutdown();
    const int requested = std::max(1, threads);
    total_threads_ = requested;
    worker_count_ = total_threads_ > 0 ? total_threads_ - 1 : 0;
    stop_ = false;
    job_.fn = nullptr;
    job_.total = 0;
    job_.chunk = 1;
    job_.chunk_count = 0;
    job_.next.store(0, std::memory_order_relaxed);
    job_.pending.store(0, std::memory_order_relaxed);
    job_.active = false;
    workers_.reserve(static_cast<size_t>(worker_count_));
    for (int i = 0; i < worker_count_; ++i) {
      workers_.emplace_back([this, worker_id = i + 1]() { worker_loop(worker_id); });
    }
  }

  int size() const { return total_threads_; }

  template <class Fn>
  void parallel_for(size_t total_items, size_t chunk, Fn&& fn) {
    if (total_items == 0) return;
    if (chunk == 0) chunk = 1;
    const size_t chunk_size = chunk;
    std::function<void(size_t, size_t, int)> wrapped = std::forward<Fn>(fn);
    if (total_threads_ <= 1 || worker_count_ == 0 || total_items <= chunk_size) {
      size_t start = 0;
      while (start < total_items) {
        const size_t end = std::min(total_items, start + chunk_size);
        wrapped(start, end, 0);
        start = end;
      }
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
      job_.pending.store(worker_count_, std::memory_order_relaxed);
      job_.active = true;
    }
    cv_job_.notify_all();

    drain_chunks(0);

    std::unique_lock<std::mutex> lock(mutex_);
    cv_done_.wait(lock, [&] { return !job_.active; });
    job_.fn = nullptr;
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

template <class T>
struct ButterflyKernel {
  using Complex = std::complex<T>;
  using Traits = Eigen::internal::packet_traits<Complex>;
  using Packet = typename Traits::type;
  static constexpr int PacketSize = Traits::size;

  static void apply(Complex* a, Complex* b, int width, const Complex& w) {
    int lane = 0;
    if constexpr (PacketSize > 1) {
      const Packet w_packet = Eigen::internal::pset1<Packet>(w);
      for (; lane + PacketSize <= width; lane += PacketSize) {
        const Packet a_pack = Eigen::internal::ploadu<Packet>(a + lane);
        const Packet b_pack = Eigen::internal::ploadu<Packet>(b + lane);
        const Packet b_twiddled = Eigen::internal::pmul(b_pack, w_packet);
        const Packet sum = Eigen::internal::padd(a_pack, b_twiddled);
        const Packet diff = Eigen::internal::psub(a_pack, b_twiddled);
        Eigen::internal::pstoreu<Complex, Packet>(a + lane, sum);
        Eigen::internal::pstoreu<Complex, Packet>(b + lane, diff);
      }
    }
    for (; lane < width; ++lane) {
      const Complex ai = a[lane];
      const Complex bt = b[lane] * w;
      a[lane] = ai + bt;
      b[lane] = ai - bt;
    }
  }
};

template <class T>
struct ButterflyScatter {
  using Complex = std::complex<T>;
  using Traits = Eigen::internal::packet_traits<Complex>;
  using Packet = typename Traits::type;
  static constexpr int PacketSize = Traits::size;

  static void apply(const Complex* a, const Complex* b, Complex* out0, Complex* out1,
                    int width, const Complex& w) {
    int lane = 0;
    if constexpr (PacketSize > 1) {
      const Packet w_packet = Eigen::internal::pset1<Packet>(w);
      for (; lane + PacketSize <= width; lane += PacketSize) {
        const Packet a_pack = Eigen::internal::ploadu<Packet>(a + lane);
        const Packet b_pack = Eigen::internal::ploadu<Packet>(b + lane);
        const Packet b_twiddled = Eigen::internal::pmul(b_pack, w_packet);
        const Packet sum = Eigen::internal::padd(a_pack, b_twiddled);
        const Packet diff = Eigen::internal::psub(a_pack, b_twiddled);
        Eigen::internal::pstoreu<Complex, Packet>(out0 + lane, sum);
        Eigen::internal::pstoreu<Complex, Packet>(out1 + lane, diff);
      }
    }
    for (; lane < width; ++lane) {
      const Complex ai = a[lane];
      const Complex bt = b[lane] * w;
      out0[lane] = ai + bt;
      out1[lane] = ai - bt;
    }
  }
};

template <class T>
struct StockhamState : KernelContext {
  using Complex = std::complex<T>;

  struct ThreadScratch {
    std::vector<Complex> ping;
    std::vector<Complex> pong;
    std::vector<Complex*> columns;
  };

  explicit StockhamState(Plan<T>& plan)
      : pool(plan.use_threads ? plan.effective_threads(plan.packet_cols) : 1),
        lane_capacity(std::max(1, (plan.tuning.packet_step > 0) ? plan.tuning.packet_step
                                                                : plan.packet_cols)),
        N(plan.N) {
    scratch.resize(static_cast<size_t>(std::max(1, pool.size())));
    resize_buffers();
  }

  void ensure_lane_capacity(int requested) {
    if (requested <= lane_capacity) return;
    lane_capacity = requested;
    resize_buffers();
  }

  void ensure_threads(int threads) {
    const int desired = std::max(1, threads);
    if (desired == pool.size()) return;
    pool.reset(desired);
    resize_buffers();
  }

  WorkerPool pool;
  std::vector<ThreadScratch> scratch;
  int lane_capacity = 1;
  int N = 0;

 private:
  void resize_buffers() {
    const size_t per_lane = static_cast<size_t>(N) * static_cast<size_t>(lane_capacity);
    if (per_lane == 0) return;
    scratch.resize(static_cast<size_t>(std::max(1, pool.size())));
    for (auto& ctx : scratch) {
      ctx.ping.resize(per_lane);
      ctx.pong.resize(per_lane);
      ctx.columns.resize(static_cast<size_t>(lane_capacity));
    }
  }
};

}  // namespace detail

template <class T>
struct AxisLayout {
  using Complex = std::complex<T>;
  Complex* base = nullptr;
  Eigen::Index axis_size = 0;
  Eigen::Index batch_size = 0;
  Eigen::Index axis_stride = 1;
  Eigen::Index batch_stride = 0;

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
  struct KernelDescriptor;

  int N;
  bool inverse;
  std::vector<Complex> W;   // base twiddles size N/2
  Eigen::VectorXi bitrev;
  int lgN;
  bool use_threads;
  int requested_threads;
  int packet_cols;  // auto from Eigen packets unless overridden

  enum class ParallelDim { Auto, Columns, KBlocks };
  enum class Schedule { Auto, Static, Dynamic, Guided };
  struct Tuning {
    ParallelDim parallel_dim = ParallelDim::Auto;
    Schedule schedule = Schedule::Auto;
    int packet_step = 0;            // 0 => auto (=packet_cols), else multiple thereof
    int min_work_per_thread = 64;   // heuristics gate
    bool force_ftz_daz = true;
  } tuning;

  struct Workspace {
    int capacity = 0;
    int threads = 0;
    std::vector<RowBuffer> a_cache;
    std::vector<RowBuffer> b_cache;
    std::vector<std::vector<Complex*>> column_ptrs;
    Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> nd_transpose;
    Eigen::Index nd_rows = 0;
    Eigen::Index nd_cols = 0;

    void ensure(int threadCount, int cols) {
      if (threadCount <= 0) threadCount = 1;
      if (threadCount != threads) {
        a_cache.resize(threadCount);
        b_cache.resize(threadCount);
        column_ptrs.resize(threadCount);
        threads = threadCount;
        capacity = 0;
#if EIGFFT_DEBUG
        std::cout << "Workspace::ensure resized caches for threads=" << threads
                  << std::endl;
#endif
      }
      if (cols > capacity) {
        for (auto& row : a_cache) row.resize(cols);
        for (auto& row : b_cache) row.resize(cols);
        for (auto& columns : column_ptrs) {
          columns.resize(cols);
          std::fill(columns.begin(), columns.end(), nullptr);
        }
        capacity = cols;
#if EIGFFT_DEBUG
        std::cout << "Workspace::ensure expanded capacity to cols=" << capacity
                  << std::endl;
#endif
      }
    }

    void ensure_nd_buffer(Eigen::Index rows, Eigen::Index cols) {
      if (rows <= 0 || cols <= 0) {
        nd_rows = nd_cols = 0;
        nd_transpose.resize(0, 0);
        return;
      }
      if (rows != nd_rows || cols != nd_cols) {
        nd_transpose.resize(cols, rows);
        nd_rows = rows;
        nd_cols = cols;
#if EIGFFT_DEBUG
        std::cout << "Workspace::ensure_nd_buffer resized transpose scratch to "
                  << nd_transpose.rows() << "x" << nd_transpose.cols() << std::endl;
#endif
      }
    }
  };

  mutable Workspace workspace;
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

  Plan(int n, bool inv=false, bool threads=true, int max_threads=0)
      : N(n), inverse(inv), W(n/2), bitrev(n), lgN(0), use_threads(threads),
        requested_threads(max_threads), packet_cols(1) {
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

    int packet = static_cast<int>(Eigen::internal::packet_traits<Complex>::size);
    if (packet <= 0) packet = 1;
    packet_cols = packet;
#if EIGFFT_DEBUG
    std::cout << "Plan using packet_cols=" << packet_cols << std::endl;
#endif
    select_kernel(baseline_kernel());
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
  return std::max(1, limit);
  }

  void ensure_workspace(int threads, int cols) const {
    workspace.ensure(threads, cols);
  }

  void ensure_nd_workspace(Eigen::Index rows, Eigen::Index cols) const {
    workspace.ensure_nd_buffer(rows, cols);
  }

  auto& transpose_buffer() const {
    return workspace.nd_transpose;
  }

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
inline void tiled_transpose(const std::complex<T>* src, Eigen::Index rows, Eigen::Index cols,
                            std::complex<T>* dst, Eigen::Index tile_rows, Eigen::Index tile_cols)
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
          dst[(c0 + j) + (r0 + i) * dest_rows] = src_col[i];
        }
      }
    }
  }
}

template<class T>
inline void baseline_execute_axis(const Plan<T>& P, const AxisLayout<T>& layout, KernelContext* ctx)
{
  (void)ctx;
  using Complex = typename Plan<T>::Complex;

  const int N = P.N;
  if (!layout.valid())
    throw std::invalid_argument("AxisLayout must reference valid data.");
  if (layout.axis_size != N)
    throw std::invalid_argument("AxisLayout axis_size must match Plan::N.");
  const int B = static_cast<int>(layout.batch_size);
  const int threads = P.effective_threads(B);
  const bool parallel_enabled = P.use_threads && threads > 1;
  const int requested_lanes =
      (P.tuning.packet_step > 0) ? P.tuning.packet_step : P.packet_cols;
  const int lane_cols = std::max(1, requested_lanes);
  P.ensure_workspace(threads, std::max(B, lane_cols));
  if (P.tuning.force_ftz_daz) Plan<T>::set_ftz_daz(true);

  const int chunk_count = (B + lane_cols - 1) / lane_cols;
  const Eigen::Index axis_stride = layout.axis_stride;
  const Eigen::Index batch_stride = layout.batch_stride;
  const int* bitrev = P.bitrev.data();

  bool permuted_in_parallel = false;
#ifdef _OPENMP
  if (parallel_enabled) {
    permuted_in_parallel = true;
#pragma omp parallel num_threads(threads)
    {
      const int tid = omp_get_thread_num();
      auto& columns = P.workspace.column_ptrs[tid];
#pragma omp for schedule(static)
      for (int chunk = 0; chunk < chunk_count; ++chunk) {
        const int col = chunk * lane_cols;
        const int width = std::min(lane_cols, B - col);
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
    auto& columns = P.workspace.column_ptrs[0];
    for (int chunk = 0; chunk < chunk_count; ++chunk) {
      const int col = chunk * lane_cols;
      const int width = std::min(lane_cols, B - col);
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

  for (int len = 2; len <= N; len <<= 1) {
    const int half = len >> 1;
    const int step = N / len;
    const int blocks = N / len;

    bool butterflies_in_parallel = false;
#ifdef _OPENMP
    if (parallel_enabled) {
      butterflies_in_parallel = true;
#pragma omp parallel num_threads(threads)
      {
        const int tid = omp_get_thread_num();
        auto& a_cache = P.workspace.a_cache[tid];
        auto& b_cache = P.workspace.b_cache[tid];
        auto& columns = P.workspace.column_ptrs[tid];
        Complex* a_ptr = a_cache.data();
        Complex* b_ptr = b_cache.data();
        const int work_items = half * blocks * chunk_count;
        if (work_items > 0) {
#pragma omp for schedule(static)
          for (int item = 0; item < work_items; ++item) {
            int tmp = item;
            const int chunk = tmp % chunk_count;
            tmp /= chunk_count;
            const int block = tmp % blocks;
            const int k = tmp / blocks;
            const int col = chunk * lane_cols;
            const int width = std::min(lane_cols, B - col);
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
            for (int lane = 0; lane < width; ++lane) {
              Complex* column_ptr = columns[lane];
              a_ptr[lane] = column_ptr[a_offset];
              b_ptr[lane] = column_ptr[b_offset];
            }
            detail::ButterflyKernel<T>::apply(a_ptr, b_ptr, width, w);
            for (int lane = 0; lane < width; ++lane) {
              Complex* column_ptr = columns[lane];
              column_ptr[a_offset] = a_ptr[lane];
              column_ptr[b_offset] = b_ptr[lane];
            }
          }
        }
      }
    }
#endif
    if (!butterflies_in_parallel) {
      auto& a_cache = P.workspace.a_cache[0];
      auto& b_cache = P.workspace.b_cache[0];
      auto& columns = P.workspace.column_ptrs[0];
      Complex* a_ptr = a_cache.data();
      Complex* b_ptr = b_cache.data();
      for (int k = 0; k < half; ++k) {
        const Complex w = P.W[k * step];
        for (int block = 0; block < blocks; ++block) {
          const int base = block * len;
          const int a_index = base + k;
          const int b_index = a_index + half;
          const Eigen::Index a_offset = Eigen::Index(a_index) * axis_stride;
          const Eigen::Index b_offset = Eigen::Index(b_index) * axis_stride;
          for (int chunk = 0; chunk < chunk_count; ++chunk) {
            const int col = chunk * lane_cols;
            const int width = std::min(lane_cols, B - col);
            if (width <= 0) continue;
            for (int lane = 0; lane < width; ++lane) {
              const Eigen::Index batch_index = Eigen::Index(col + lane);
              columns[lane] = layout.base + batch_index * batch_stride;
            }
            for (int lane = 0; lane < width; ++lane) {
              Complex* column_ptr = columns[lane];
              a_ptr[lane] = column_ptr[a_offset];
              b_ptr[lane] = column_ptr[b_offset];
            }
            detail::ButterflyKernel<T>::apply(a_ptr, b_ptr, width, w);
            for (int lane = 0; lane < width; ++lane) {
              Complex* column_ptr = columns[lane];
              column_ptr[a_offset] = a_ptr[lane];
              column_ptr[b_offset] = b_ptr[lane];
            }
          }
        }
      }
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
inline void fft_inplace_batched(Eigen::Ref<Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> X,
                                const Plan<T>& P)
{
  AxisLayout<T> layout;
  layout.base = X.data();
  layout.axis_size = P.N;
  layout.batch_size = static_cast<Eigen::Index>(X.cols());
  layout.axis_stride = 1;
  layout.batch_stride = static_cast<Eigen::Index>(X.rows());
  fft_apply_axis(P, layout);
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
  auto& scratch = axis0_plan.transpose_buffer();
  detail::tiled_transpose<T>(X.data(), rows, cols, scratch.data(), tile_rows, tile_cols);

  AxisLayout<T> axis1_layout;
  axis1_layout.base = scratch.data();
  axis1_layout.axis_size = cols;
  axis1_layout.batch_size = rows;
  axis1_layout.axis_stride = 1;
  axis1_layout.batch_stride = cols;
  fft_apply_axis(axis1_plan, axis1_layout);

  detail::tiled_transpose<T>(scratch.data(), cols, rows, X.data(), tile_rows, tile_cols);
}

namespace detail {

template<class T>
std::unique_ptr<KernelContext> baseline_create_state(Plan<T>&) {
  return nullptr;
}

template<class T>
void baseline_destroy_state(Plan<T>&, KernelContext*) {}

template<class T>
struct StockhamKernelState final : KernelContext {
  using Complex = std::complex<T>;
  struct ThreadBuffers {
    std::vector<Complex*> column_ptrs;
    std::vector<Complex> a_cache;
    std::vector<Complex> b_cache;
  };

  void ensure(int desired_threads, int desired_lane_capacity) {
    desired_threads = std::max(1, desired_threads);
    desired_lane_capacity = std::max(1, desired_lane_capacity);
    if (desired_threads != threads_) {
      threads_ = desired_threads;
      buffers_.resize(threads_);
      if (threads_ > 1) {
        pool_ = std::make_unique<ThreadPool>(threads_);
      } else {
        pool_.reset();
      }
      if (lane_capacity_ > 0) {
        for (auto& buf : buffers_) {
          resize_buffers(buf, lane_capacity_);
        }
      }
    }
    if (desired_lane_capacity > lane_capacity_) {
      lane_capacity_ = desired_lane_capacity;
      for (auto& buf : buffers_) {
        resize_buffers(buf, lane_capacity_);
      }
    }
  }

  int thread_count() const { return threads_; }
  int lane_capacity() const { return lane_capacity_; }
  ThreadPool* pool() const { return pool_.get(); }

  ThreadBuffers& buffers(int thread_id) { return buffers_[thread_id]; }

 private:
  static void resize_buffers(ThreadBuffers& buf, int lanes) {
    buf.column_ptrs.resize(lanes);
    buf.a_cache.resize(lanes);
    buf.b_cache.resize(lanes);
  }

  int threads_ = 1;
  int lane_capacity_ = 0;
  std::unique_ptr<ThreadPool> pool_;
  std::vector<ThreadBuffers> buffers_;
};

template<class T>
void stockham_execute_axis(const Plan<T>& P, const AxisLayout<T>& layout, KernelContext* ctx) {
  using Complex = typename Plan<T>::Complex;
  if (!layout.valid()) {
    throw std::invalid_argument("AxisLayout must reference valid data.");
  }
  if (layout.axis_size != P.N) {
    throw std::invalid_argument("AxisLayout axis_size must match Plan::N.");
  }
  auto* state = static_cast<StockhamState<T>*>(ctx);
  if (!state) {
    baseline_execute_axis(P, layout, nullptr);
    return;
  }

  const int N = P.N;
  const int stages = P.lgN;
  const int B = static_cast<int>(layout.batch_size);
  if (B <= 0) {
    if (P.tuning.force_ftz_daz) Plan<T>::set_ftz_daz(false);
    return;
  }

  const int desired_threads = P.effective_threads(B);
  state->ensure_threads(desired_threads);

  const int requested_lanes =
      (P.tuning.packet_step > 0) ? P.tuning.packet_step : P.packet_cols;
  const int lane_cols = std::max(1, requested_lanes);
  state->ensure_lane_capacity(lane_cols);
  const int lane_capacity = state->lane_capacity;
  const Eigen::Index axis_stride = layout.axis_stride;
  const Eigen::Index batch_stride = layout.batch_stride;

  if (P.tuning.force_ftz_daz) Plan<T>::set_ftz_daz(true);

  auto process_chunk = [&](size_t start, size_t end, int worker_id) {
    if (start >= static_cast<size_t>(B)) return;
    const int width = static_cast<int>(end - start);
    if (width <= 0) return;
    size_t slot = static_cast<size_t>(worker_id);
    if (slot >= state->scratch.size()) {
      slot = state->scratch.empty() ? size_t(0) : state->scratch.size() - 1;
    }
    auto& scratch = state->scratch[slot];
    auto* columns = scratch.columns.data();
    for (int lane = 0; lane < width; ++lane) {
      const Eigen::Index batch_index = Eigen::Index(start + lane);
      columns[lane] = layout.base + batch_index * batch_stride;
    }

    Complex* stage_in = scratch.ping.data();
    Complex* stage_out = scratch.pong.data();
    for (int i = 0; i < N; ++i) {
      const Eigen::Index offset = Eigen::Index(i) * axis_stride;
      Complex* dest = stage_in + i * lane_capacity;
      for (int lane = 0; lane < width; ++lane) {
        dest[lane] = columns[lane][offset];
      }
    }

    Complex* in = stage_in;
    Complex* out = stage_out;
    for (int stage = 0; stage < stages; ++stage) {
      const int m = 1 << stage;
      const int distance = m << 1;
      const int groups = N / distance;
      const int tw_step = N / distance;
      for (int j = 0; j < m; ++j) {
        const Complex w = P.W[j * tw_step];
        for (int g = 0; g < groups; ++g) {
          const int idx0 = g * distance + j;
          const int idx1 = idx0 + m;
          const int out_base = g * distance + (j << 1);
          const Complex* src0 = in + idx0 * lane_capacity;
          const Complex* src1 = in + idx1 * lane_capacity;
          Complex* dst0 = out + out_base * lane_capacity;
          Complex* dst1 = out + (out_base + 1) * lane_capacity;
          ButterflyScatter<T>::apply(src0, src1, dst0, dst1, width, w);
        }
      }
      std::swap(in, out);
    }

    const Complex* final_buf = (stages % 2 == 0) ? stage_in : stage_out;
    if (P.inverse) {
      const T scale = T(1) / T(N);
      for (int i = 0; i < N; ++i) {
        const Eigen::Index offset = Eigen::Index(i) * axis_stride;
        const Complex* src = final_buf + i * lane_capacity;
        for (int lane = 0; lane < width; ++lane) {
          columns[lane][offset] = src[lane] * scale;
        }
      }
    } else {
      for (int i = 0; i < N; ++i) {
        const Eigen::Index offset = Eigen::Index(i) * axis_stride;
        const Complex* src = final_buf + i * lane_capacity;
        for (int lane = 0; lane < width; ++lane) {
          columns[lane][offset] = src[lane];
        }
      }
    }
  };

  state->pool.parallel_for(static_cast<size_t>(B), static_cast<size_t>(lane_cols), process_chunk);

  if (P.tuning.force_ftz_daz) Plan<T>::set_ftz_daz(false);
}

template<class T>
std::unique_ptr<KernelContext> stockham_create_state(Plan<T>& plan) {
  return std::unique_ptr<KernelContext>(new StockhamState<T>(plan));
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

} // namespace eigfft
