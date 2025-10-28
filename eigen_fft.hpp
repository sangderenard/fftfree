#ifndef EIGFFT_DEBUG
#define EIGFFT_DEBUG 0
#endif

// eigen_fft.hpp (header-only)
#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
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

namespace detail {

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

}  // namespace detail

template<class T> struct Plan {
  using Complex = std::complex<T>;
  using RowBuffer = Eigen::Matrix<Complex, 1, Eigen::Dynamic, Eigen::RowMajor>;

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
    std::vector<Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>
        perm_block;

    void ensure(int threadCount, int cols) {
      if (threadCount <= 0) threadCount = 1;
      if (threadCount != threads) {
        a_cache.resize(threadCount);
        b_cache.resize(threadCount);
        perm_block.resize(threadCount);
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
        for (auto& block : perm_block) block.resize(0, 0);
        capacity = cols;
#if EIGFFT_DEBUG
        std::cout << "Workspace::ensure expanded capacity to cols=" << capacity
                  << std::endl;
#endif
      }
    }
  };

  mutable Workspace workspace;

  Plan(int n, bool inv=false, bool threads=true, int max_threads=0)
      : N(n), inverse(inv), W(n/2), bitrev(n), lgN(0), use_threads(threads),
        requested_threads(max_threads), packet_cols(1) {
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
#if EIGFFT_DEBUG
    std::cout << "Plan using packet_cols=" << packet_cols << std::endl;
#endif
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
  using Complex = typename Plan<T>::Complex;

  const int N = P.N;
  const int B = static_cast<int>(X.cols());
  const int threads = P.effective_threads(B);
  const int lane_cols = std::max(1, (P.tuning.packet_step > 0)
                                      ? P.tuning.packet_step
                                      : P.packet_cols);
  P.ensure_workspace(threads, std::max(B, lane_cols));
  if (P.tuning.force_ftz_daz) Plan<T>::set_ftz_daz(true);

  const int chunk_count = (B + lane_cols - 1) / lane_cols;

#ifdef _OPENMP
  if (P.use_threads && threads > 1) {
#pragma omp parallel num_threads(threads)
    {
      const int tid = omp_get_thread_num();
      auto& tmp = P.workspace.perm_block[tid];
      tmp.resize(N, lane_cols);
#pragma omp for schedule(static)
      for (int chunk = 0; chunk < chunk_count; ++chunk) {
        const int col = chunk * lane_cols;
        const int width = std::min(lane_cols, B - col);
        if (width <= 0) continue;
        auto block = X.block(0, col, N, width);
        for (int lane = 0; lane < width; ++lane) {
          tmp.col(lane) = block.col(lane)(P.bitrev);
        }
        block = tmp.leftCols(width);
      }
    }
  } else
#endif
  {
    auto& tmp = P.workspace.perm_block[0];
    tmp.resize(N, lane_cols);
    for (int chunk = 0; chunk < chunk_count; ++chunk) {
      const int col = chunk * lane_cols;
      const int width = std::min(lane_cols, B - col);
      if (width <= 0) continue;
      auto block = X.block(0, col, N, width);
      for (int lane = 0; lane < width; ++lane) {
        tmp.col(lane) = block.col(lane)(P.bitrev);
      }
      block = tmp.leftCols(width);
    }
  }

  for (int len = 2; len <= N; len <<= 1) {
    const int half = len >> 1;
    const int step = N / len;
    const int blocks = N / len;
#ifdef _OPENMP
    if (P.use_threads && threads > 1) {
#pragma omp parallel num_threads(threads)
      {
        const int tid = omp_get_thread_num();
        auto& a_cache = P.workspace.a_cache[tid];
        auto& b_cache = P.workspace.b_cache[tid];
        for (int k = 0; k < half; ++k) {
          const Complex w = P.W[k * step];
#pragma omp for collapse(2) schedule(static)
          for (int block = 0; block < blocks; ++block) {
            for (int chunk = 0; chunk < chunk_count; ++chunk) {
              const int col = chunk * lane_cols;
              const int width = std::min(lane_cols, B - col);
              if (width <= 0) continue;
              auto a_row = X.row(block * len + k);
              auto b_row = X.row(block * len + k + half);
              auto a_seg = a_row.segment(col, width);
              auto b_seg = b_row.segment(col, width);
              a_cache.head(width) = a_seg;
              b_cache.head(width) = b_seg;
              detail::ButterflyKernel<T>::apply(a_cache.data(), b_cache.data(), width, w);
              Complex* a_ptr = a_cache.data();
              Complex* b_ptr = b_cache.data();
              for (int lane = 0; lane < width; ++lane) {
                a_seg.coeffRef(lane) = a_ptr[lane];
                b_seg.coeffRef(lane) = b_ptr[lane];
              }
            }
          }
        }
      }
    } else
#endif
    {
      auto& a_cache = P.workspace.a_cache[0];
      auto& b_cache = P.workspace.b_cache[0];
      for (int k = 0; k < half; ++k) {
        const Complex w = P.W[k * step];
        for (int block = 0; block < blocks; ++block) {
          const int base = block * len;
          auto a_row = X.row(base + k);
          auto b_row = X.row(base + k + half);
          for (int chunk = 0; chunk < chunk_count; ++chunk) {
            const int col = chunk * lane_cols;
            const int width = std::min(lane_cols, B - col);
            if (width <= 0) continue;
            auto a_seg = a_row.segment(col, width);
            auto b_seg = b_row.segment(col, width);
            a_cache.head(width) = a_seg;
            b_cache.head(width) = b_seg;
            detail::ButterflyKernel<T>::apply(a_cache.data(), b_cache.data(), width, w);
            Complex* a_ptr = a_cache.data();
            Complex* b_ptr = b_cache.data();
            for (int lane = 0; lane < width; ++lane) {
              a_seg.coeffRef(lane) = a_ptr[lane];
              b_seg.coeffRef(lane) = b_ptr[lane];
            }
          }
        }
      }
    }
  }
  if (P.inverse) X.array() /= T(N);
  if (P.tuning.force_ftz_daz) Plan<T>::set_ftz_daz(false);
}

} // namespace eigfft

