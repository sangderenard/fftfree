// eigen_fft.hpp (header-only)
#pragma once
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <thread>
#include <vector>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  #include <xmmintrin.h>  // FTZ/DAZ
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

namespace eigfft {

template<class T> struct Plan {
  using Complex = std::complex<T>;
  using RowBuffer = Eigen::Matrix<Complex, 1, Eigen::Dynamic, Eigen::RowMajor>;
  using ColBuffer = Eigen::Matrix<Complex, Eigen::Dynamic, 1, Eigen::ColMajor>;

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
    std::vector<ColBuffer> perm_col;

    void ensure(int threadCount, int cols) {
      if (threadCount <= 0) threadCount = 1;
      if (threadCount != threads) {
        a_cache.resize(threadCount);
        b_cache.resize(threadCount);
        perm_col.resize(threadCount);
        capacity = 0;
        threads = threadCount;
      }
      if (cols > capacity) {
        for (auto& row : a_cache) row.resize(cols);
        for (auto& row : b_cache) row.resize(cols);
        capacity = cols;
      }
    }
  };

  mutable Workspace workspace;

  Plan(int n, bool inv=false, bool threads=true, int max_threads=0)
      : N(n), inverse(inv), W(n/2), bitrev(n), lgN(0), use_threads(threads), requested_threads(max_threads), packet_cols(1) {
    // power-of-two check
    int t = N;
    while ((t & 1) == 0) { ++lgN; t >>= 1; }
    if ((1 << lgN) != N) throw std::invalid_argument("N must be power of two");

    // twiddles
    const T sgn = inverse ? T(+1) : T(-1);
  const T tau = sgn * T(2 * std::acos(T(-1))) / T(N);
    for (int k = 0; k < N / 2; ++k) {
      const T ang = tau * T(k);
      W[k] = { std::cos(ang), std::sin(ang) };
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
  }

  // Optional: override SIMD coarsening (e.g., 2× packet for fat batches)
  void set_packet_step(int step) {
    tuning.packet_step = step;
  }

  // Enable/disable FTZ/DAZ at call sites
  static void set_ftz_daz(bool on) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    _MM_SET_FLUSH_ZERO_MODE(on ? _MM_FLUSH_ZERO_ON : _MM_FLUSH_ZERO_OFF);
    _MM_SET_DENORMALS_ZERO_MODE(on ? _MM_DENORMALS_ZERO_ON : _MM_DENORMALS_ZERO_OFF);
#else
    (void)on;
#endif
  }

  int effective_threads(int batch_cols) const {
    if (!use_threads) return 1;
    int limit = requested_threads;
#ifdef _OPENMP
    if (limit <= 0) limit = omp_get_max_threads();
    else limit = std::min(limit, omp_get_max_threads());
#else
    if (limit <= 0) limit = static_cast<int>(std::thread::hardware_concurrency());
    if (limit <= 0) limit = 1;
#endif
    limit = std::max(1, std::min(limit, batch_cols));
    return limit;
  }

  void ensure_workspace(int threads, int cols) const {
    workspace.ensure(threads, cols);
  }
};

// X: N x B (rows=time/freq, cols=batch); in-place
template<class T>
inline void fft_inplace_batched(Eigen::Ref<Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> X,
                                const Plan<T>& P)
{
  const int N = P.N;
  const int B = static_cast<int>(X.cols());
  const int threads = P.effective_threads(B);
  P.ensure_workspace(threads, B);
  if (P.tuning.force_ftz_daz) Plan<T>::set_ftz_daz(true);

  // Compute work geometry
  auto choose_parallel_dim = [&]() {
    if (P.tuning.parallel_dim != Plan<T>::ParallelDim::Auto) return P.tuning.parallel_dim;
    // Heuristic: if B >= 2*threads, prefer Columns; else KBlocks
    return (B >= 2*threads) ? Plan<T>::ParallelDim::Columns : Plan<T>::ParallelDim::KBlocks;
  };
  auto choose_schedule = [&](int work_items){
    if (P.tuning.schedule == Plan<T>::Schedule::Static) return 0;
    if (P.tuning.schedule == Plan<T>::Schedule::Dynamic) return 1;
    if (P.tuning.schedule == Plan<T>::Schedule::Guided) return 2;
    // Auto:
    return (work_items < P.tuning.min_work_per_thread*threads) ? 1 : 0; // dynamic for small tasks
  };
  const int packet_step = (P.tuning.packet_step>0) ? P.tuning.packet_step : P.packet_cols;

  // Bit-reversal with explicit temp to avoid aliasing
#ifdef _OPENMP
  #pragma omp parallel if(P.use_threads) num_threads(threads)
#endif
  {
#ifdef _OPENMP
    const int tid = P.use_threads ? omp_get_thread_num() : 0;
#else
    const int tid = 0;
#endif
    auto& tmp = P.workspace.perm_col[tid];
    tmp.resize(N);
#ifdef _OPENMP
    #pragma omp for schedule(static)
#endif
    for (int b = 0; b < B; ++b) {
      tmp = X.col(b)(P.bitrev);
      X.col(b) = tmp;
    }
  }

  for (int len = 2; len <= N; len <<= 1) {
    const int half = len >> 1;
    const int step = N / len;
    const int blocks = N / len;
    const auto par_dim = choose_parallel_dim();
    const int work_items = (par_dim==Plan<T>::ParallelDim::Columns) ? B : (half*blocks);
    const int sched = choose_schedule(work_items);

    // One parallel region per stage
#ifdef _OPENMP
    #pragma omp parallel if(P.use_threads) num_threads(threads)
#endif
    {
#ifdef _OPENMP
      const int tid2 = P.use_threads ? omp_get_thread_num() : 0;
#else
      const int tid2 = 0;
#endif
      auto& a_cache = P.workspace.a_cache[tid2];
      auto& b_cache = P.workspace.b_cache[tid2];

      if (par_dim == Plan<T>::ParallelDim::Columns) {
        auto process_column = [&](int col) {
          // process one column fully (good when B is big)
          for (int block = 0; block < blocks; ++block) {
            const int base = block * len;
            for (int k = 0; k < half; ++k) {
              const std::complex<T> w = P.W[k * step];
              auto a = X(base + k, col);
              auto b = X(base + k + half, col);
              if (k==0) {
                // twiddle = 1
                X(base + k, col)         = a + b;
                X(base + k + half, col)  = a - b;
              } else {
                const auto t = w * b;
                X(base + k, col)         = a + t;
                X(base + k + half, col)  = a - t;
              }
            }
          }
        };

        #pragma omp for schedule(static)
        for (int col = 0; col < B; ++col) process_column(col);
      } else {
        // Parallelize over k×blocks, vectorize across columns
        auto process_work = [&](int work) {
          const int k = work % half;
          const int block = work / half;
          const std::complex<T> w = P.W[k * step];
          const int base = block * len;
          auto a_row = X.row(base + k);
          auto b_row = X.row(base + k + half);
          for (int col = 0; col < B; col += packet_step) {
            const int width = std::min(packet_step, B - col);
            auto a_seg = a_row.segment(col, width);
            auto b_seg = b_row.segment(col, width);
            a_cache.head(width) = a_seg;
            b_cache.head(width) = b_seg;
            if (k==0) {
              a_seg = a_cache.head(width) + b_cache.head(width);
              b_seg = a_cache.head(width) - b_cache.head(width);
            } else {
              b_cache.head(width) *= w;
              a_seg = a_cache.head(width) + b_cache.head(width);
              b_seg = a_cache.head(width) - b_cache.head(width);
            }
          }
        };

        #pragma omp for schedule(static)
        for (int work = 0; work < half*blocks; ++work) process_work(work);
      }
    } // parallel region
  }
  if (P.inverse) X.array() /= T(N);
  if (P.tuning.force_ftz_daz) Plan<T>::set_ftz_daz(false);
}

} // namespace eigfft