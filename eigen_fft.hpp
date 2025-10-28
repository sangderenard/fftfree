#ifndef EIGFFT_DEBUG
#define EIGFFT_DEBUG 1
#endif
// eigen_fft.hpp (header-only)
#pragma once
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <thread>
#include <vector>
// #if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
//   #include <xmmintrin.h>  // FTZ/DAZ
// #endif
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
        std::vector<Complex> W;   // base twiddles size N/2
        // Debugging output
        #if EIGFFT_DEBUG
        std::cout << "Plan constructor entered, n=" << n << std::endl;
        #endif
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
    std::cout << "Plan constructor entered, n=" << n << std::endl;
    // power-of-two check
    int t = N;
    while ((t & 1) == 0) { ++lgN; t >>= 1; }
    if ((1 << lgN) != N) throw std::invalid_argument("N must be power of two");

    // twiddles
    const T sgn = inverse ? T(+1) : T(-1);
  const T tau = sgn * T(2 * std::acos(T(-1))) / T(N);
    for (int k = 0; k < N / 2; ++k) {
      const T ang = tau * T(k);
            #if EIGFFT_DEBUG
            std::cout << "Workspace::ensure called with threadCount=" << threadCount << " cols=" << cols << std::endl;
            #endif
      W[k] = { std::cos(ang), std::sin(ang) };
    }

    // bit-reversal
    for (int i = 0; i < N; ++i) {
      unsigned x = static_cast<unsigned>(i);
      unsigned r = 0;
              #if EIGFFT_DEBUG
              std::cout << "  Workspace resized: threads=" << threads << std::endl;
              #endif
      for (int b = 0; b < lgN; ++b) {
        r = (r << 1) | (x & 1u);
        x >>= 1;
      }
              #if EIGFFT_DEBUG
              std::cout << "  Workspace buffers resized to cols=" << cols << std::endl;
              #endif
      bitrev[i] = static_cast<int>(r);
    }
            #if EIGFFT_DEBUG
            std::cout << "  Workspace status: a_cache.size=" << a_cache.size() << " b_cache.size=" << b_cache.size() << " perm_col.size=" << perm_col.size() << " capacity=" << capacity << std::endl;
            #endif

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
    // #if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    //   _MM_SET_FLUSH_ZERO_MODE(on ? _MM_FLUSH_ZERO_ON : _MM_FLUSH_ZERO_OFF);
    //   _MM_SET_DENORMALS_ZERO_MODE(on ? _MM_DENORMALS_ZERO_ON : _MM_DENORMALS_ZERO_OFF);
    // #else
    (void)on;
    // #endif
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

  #if EIGFFT_DEBUG
    std::cout << "Workspace buffer sizes: perm_col=" << P.workspace.perm_col.size()
              << ", a_cache=" << P.workspace.a_cache.size()
              << ", b_cache=" << P.workspace.b_cache.size() << std::endl;
    std::cout << "Bitrev size: " << P.bitrev.size() << " N=" << N << " B=" << B << std::endl;
    if (P.workspace.perm_col.size() == 0 || P.workspace.a_cache.size() == 0 || P.workspace.b_cache.size() == 0) {
      std::cerr << "Workspace buffers not initialized!" << std::endl;
      throw std::runtime_error("Workspace buffers not initialized");
    }
    if (P.bitrev.size() != N) {
      std::cerr << "Bitrev size mismatch!" << std::endl;
      throw std::runtime_error("Bitrev size mismatch");
    }
  #endif

  // Bit-reversal with explicit temp to avoid aliasing
  {
    std::cout << "Bit-reversal start" << std::endl;
    auto& tmp = P.workspace.perm_col[0];
    tmp.resize(N);
    for (int b = 0; b < B; ++b) {
      tmp = X.col(b)(P.bitrev);
      X.col(b) = tmp;
    }
    std::cout << "Bit-reversal end" << std::endl;
  }

  for (int len = 2; len <= N; len <<= 1) {
    std::cout << "Stage len=" << len << std::endl;
    const int half = len >> 1;
    const int step = N / len;
    const int blocks = N / len;
    auto& a_cache = P.workspace.a_cache[0];
    auto& b_cache = P.workspace.b_cache[0];
    for (int k = 0; k < half; ++k) {
      const std::complex<T> w = P.W[k * step];
      for (int block = 0; block < blocks; ++block) {
        const int base = block * len;
        auto a_row = X.row(base + k);
        auto b_row = X.row(base + k + half);
        for (int col = 0; col < B; col += P.packet_cols) {
          const int width = std::min(P.packet_cols, B - col);
          #if EIGFFT_DEBUG
          std::cout << "  SIMD chunk: stage_len=" << len << " k=" << k << " block=" << block << " col=" << col << " width=" << width << " packet_cols=" << P.packet_cols << std::endl;
          #endif
          if (width <= 0) std::cout << "width=0 at col=" << col << std::endl;
          auto a_seg = a_row.segment(col, width);
          auto b_seg = b_row.segment(col, width);
          a_cache.head(width) = a_seg;
          b_cache.head(width) = b_seg;
          b_cache.head(width) *= w;
          a_seg = a_cache.head(width) + b_cache.head(width);
          b_seg = a_cache.head(width) - b_cache.head(width);
        }
      }
    }
    std::cout << "Stage len=" << len << " end" << std::endl;
  }
  if (P.inverse) X.array() /= T(N);
  if (P.tuning.force_ftz_daz) Plan<T>::set_ftz_daz(false);
}

} // namespace eigfft