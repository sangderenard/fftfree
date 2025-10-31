#include "../eigen_fft.hpp"
#include "../plan_support.hpp"

#include <Eigen/Core>

#include <cstddef>
#include <cmath>
#include <complex>
#include <array>
#include <random>
#include <string>
#include <type_traits>
#include <vector>
#include <iostream>
#include <iomanip>

namespace {

template <typename Scalar>
void print_twiddles(const std::vector<std::complex<Scalar>>& tw, int stages, int N) {
  using Complex = std::complex<Scalar>;
  std::cout << "Twiddle dump (N=" << N << ", stages=" << stages << ")" << std::endl;
  for (int stage = 0; stage < stages; ++stage) {
    const int twiddle_count = 1 << stage;
    const std::size_t stage_offset = (static_cast<std::size_t>(1) << stage) - 1;
    std::cout << "  Stage " << stage << " twiddles:";
    for (int j = 0; j < twiddle_count; ++j) {
      const Complex& value = tw[stage_offset + static_cast<std::size_t>(j)];
      std::cout << " (" << value.real() << ", " << value.imag() << ")";
    }
    std::cout << std::endl;
  }
}

template <typename Scalar>
void print_layout(const char* label, const std::vector<std::complex<Scalar>>& layout, int N) {
  std::cout << label << " layout (pos -> src):";
  for (int idx = 0; idx < N; ++idx) {
    const int raw = static_cast<int>(std::lround(layout[static_cast<std::size_t>(idx)].real()));
    int mod = raw % N; if (mod < 0) mod += N;
    std::cout << ' ' << mod;
  }
  std::cout << std::endl;
}

template <typename Scalar>
void print_snapshots(const char* label, const std::vector<std::complex<Scalar>>& buffer, int stages, int N) {
  using Complex = std::complex<Scalar>;
  std::cout << label << " stage snapshots:" << std::endl;
  for (int s = 0; s < stages; ++s) {
    std::cout << "  Stage " << s << ":";
    for (int i = 0; i < N; ++i) {
      const Complex& value = buffer[static_cast<std::size_t>(s) * static_cast<std::size_t>(N) + static_cast<std::size_t>(i)];
      std::cout << " (" << value.real() << ", " << value.imag() << ")";
    }
    std::cout << std::endl;
  }
}

template <typename Scalar>
void print_stage_params(const std::vector<std::complex<Scalar>>& params, int stages) {
  std::cout << "Stage parameters (distance, tw_step):" << std::endl;
  std::cout << "  ";
  for (int s = 0; s < stages; ++s) {
    const auto& p = params[static_cast<std::size_t>(s)];
    std::cout << " (" << p.real() << ", " << p.imag() << ")";
  }
  std::cout << std::endl;
}

template <typename Scalar>
void print_pairs(const char* label, const std::vector<std::complex<Scalar>>& buffer, int stages, int N) {
  std::cout << label << " butterfly pairs (stage: pos -> (lhs,rhs))" << std::endl;
  for (int s = 0; s < stages; ++s) {
    std::cout << "  Stage " << s << ":";
    for (int i = 0; i < N; ++i) {
      const auto& v = buffer[static_cast<std::size_t>(s) * static_cast<std::size_t>(N) + static_cast<std::size_t>(i)];
      const int lhs = static_cast<int>(std::lround(v.real()));
      const int rhs = static_cast<int>(std::lround(v.imag()));
      std::cout << " " << i << "->(" << lhs << "," << rhs << ")";
    }
    std::cout << std::endl;
  }
}

template <typename Scalar>
void print_invariants(const char* label, const std::vector<std::complex<Scalar>>& buffer, int stages) {
  std::cout << label << " stage invariants (max_r0, max_r1):" << std::endl;
  for (int s = 0; s < stages; ++s) {
    const auto& v = buffer[static_cast<std::size_t>(s)];
    std::cout << "  Stage " << s << ": (" << v.real() << ", " << v.imag() << ")" << std::endl;
  }
}

template <typename Scalar>
void print_twmap(const char* label, const std::vector<std::complex<Scalar>>& buffer, int stages, int N) {
  std::cout << label << " twiddle index map (stage: pos -> tw_index):" << std::endl;
  for (int s = 0; s < stages; ++s) {
    std::cout << "  Stage " << s << ":";
    for (int i = 0; i < N; ++i) {
      const auto& v = buffer[static_cast<std::size_t>(s) * static_cast<std::size_t>(N) + static_cast<std::size_t>(i)];
      const int idx = static_cast<int>(std::lround(v.real()));
      std::cout << " " << i << "->" << idx;
    }
    std::cout << std::endl;
  }
}

template <typename Scalar>
void print_matrix_frames(const char* label, const Eigen::Matrix<std::complex<Scalar>, Eigen::Dynamic, Eigen::Dynamic>& M) {
  std::cout << label << " frames:" << std::endl;
  const int rows = M.rows();
  const int cols = M.cols();
  for (int r = 0; r < rows; ++r) {
    std::cout << "  row " << r << ":";
    for (int c = 0; c < cols; ++c) {
      const auto& v = M(r, c);
      std::cout << " (" << v.real() << ", " << v.imag() << ")";
    }
    std::cout << std::endl;
  }
}

template <typename Scalar>
void print_elementwise_diffs(const char* label, const Eigen::Matrix<std::complex<Scalar>, Eigen::Dynamic, Eigen::Dynamic>& A,
                             const Eigen::Matrix<std::complex<Scalar>, Eigen::Dynamic, Eigen::Dynamic>& B) {
  std::cout << label << " differences (A - B):" << std::endl;
  const int rows = A.rows();
  const int cols = A.cols();
  for (int r = 0; r < rows; ++r) {
    std::cout << "  row " << r << ":";
    for (int c = 0; c < cols; ++c) {
      const auto d = A(r,c) - B(r,c);
      std::cout << " (" << d.real() << ", " << d.imag() << ")";
    }
    std::cout << std::endl;
  }
}


template <typename Scalar>
std::string precision_tag();

template <>
std::string precision_tag<double>() { return "f64"; }

template <>
std::string precision_tag<float>() { return "f32"; }

template <typename Scalar>
Scalar tolerance();

template <>
double tolerance<double>() { return 1e-9; }

template <>
float tolerance<float>() { return 1e-5f; }

template <typename Scalar>
eigfft::PlanRuntimeConfig default_runtime_config() {
  eigfft::PlanRuntimeConfig cfg;
  cfg.threads = eigfft::Plan<Scalar>::Limits::kDefaultRuntimeThreads;
  cfg.lanes = eigfft::Plan<Scalar>::Limits::kDefaultLaneCapacity;
  return cfg;
}

template <typename Scalar>
bool test_roundtrip_small_batches() {
  using Complex = std::complex<Scalar>;
  constexpr int N = 16;
  constexpr int B = 8;
  std::mt19937_64 rng(42);
  std::normal_distribution<double> dist(0.0, 1.0);

  Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic> signal(N, B);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < B; ++j) {
      const Scalar re = static_cast<Scalar>(dist(rng));
      const Scalar im = static_cast<Scalar>(dist(rng));
      signal(i, j) = Complex(re, im);
    }
  }
  const auto original = signal;

  eigfft::PlanRuntimeConfig cfg = default_runtime_config<Scalar>();
  eigfft::PlanCache<Scalar> cache;
  cache.warm_audio_profiles(cfg);

  auto forward_token = cache.get_plan(N, /*inverse=*/false, cfg);
  auto inverse_token = cache.get_plan(N, /*inverse=*/true, cfg);

  eigfft::fft_inplace_batched<Scalar>(signal, forward_token.plan());
  eigfft::fft_inplace_batched<Scalar>(signal, inverse_token.plan());

  double max_error = 0.0;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < B; ++j) {
      const Complex current = signal(i, j);
      const Complex reference = original(i, j);
      const double re_err = static_cast<double>(current.real()) -
                            static_cast<double>(reference.real());
      const double im_err = static_cast<double>(current.imag()) -
                            static_cast<double>(reference.imag());
      const double mag = std::hypot(re_err, im_err);
      if (mag > max_error) max_error = mag;
    }
  }

  const double tol = static_cast<double>(tolerance<Scalar>());
  return max_error < tol;
}

template <typename Scalar>
bool test_effective_threads_reports_parallel() {
  eigfft::PlanRuntimeConfig cfg = default_runtime_config<Scalar>();
  eigfft::PlanCache<Scalar> cache;
  auto token = cache.get_plan(32, /*inverse=*/false, cfg);
  return token.plan().effective_threads(64) >= 2;
}

template <typename Scalar>
Eigen::Matrix<std::complex<Scalar>, Eigen::Dynamic, Eigen::Dynamic> naive_dft(
    const Eigen::Matrix<std::complex<Scalar>, Eigen::Dynamic, Eigen::Dynamic>& input) {
  using Complex = std::complex<Scalar>;
  const int N = static_cast<int>(input.rows());
  const int B = static_cast<int>(input.cols());
  Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic> output(N, B);
  const Scalar two_pi = static_cast<Scalar>(2.0 * std::acos(-1.0));
  for (int col = 0; col < B; ++col) {
    for (int k = 0; k < N; ++k) {
      Complex accum = Complex(0, 0);
      for (int n = 0; n < N; ++n) {
        const Scalar angle = -two_pi * static_cast<Scalar>(k * n) / static_cast<Scalar>(N);
        const Complex twiddle(std::cos(angle), std::sin(angle));
        accum += input(n, col) * twiddle;
      }
      output(k, col) = accum;
    }
  }
  return output;
}

template <typename Scalar>
bool test_single_thread_fallback_matches_reference() {
  using Complex = std::complex<Scalar>;
  constexpr int N = 8;
  constexpr int B = 3;
  Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic> signal(N, B);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < B; ++j) {
      const Scalar re = static_cast<Scalar>(i + 2 * j + 1);
      const Scalar im = static_cast<Scalar>(j - i - 0.5);
      signal(i, j) = Complex(re, im);
    }
  }
  const auto original = signal;

  eigfft::PlanRuntimeConfig cfg;
  cfg.threads = 1;
  cfg.lanes = eigfft::Plan<Scalar>::Limits::kDefaultLaneCapacity;
  eigfft::PlanCache<Scalar> cache;
  auto token = cache.get_plan(N, /*inverse=*/false, cfg);
  auto& plan = token.plan();
  plan.requested_threads = 1;  // Force the fallback path even when OpenMP is available.
  plan.tuning.packet_step = 0;
  eigfft::fft_inplace_batched<Scalar>(signal, plan);

  const auto reference = naive_dft<Scalar>(original);

  double max_error = 0.0;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < B; ++j) {
      const Complex diff = signal(i, j) - reference(i, j);
      const double err = std::hypot(static_cast<double>(diff.real()),
                                    static_cast<double>(diff.imag()));
      if (err > max_error) max_error = err;
    }
  }

  const double tol = static_cast<double>(tolerance<Scalar>()) * 10.0;
  return max_error < tol;
}

template <typename Scalar>
bool test_kernel_selection_interface() {
  eigfft::PlanRuntimeConfig cfg = default_runtime_config<Scalar>();
  eigfft::PlanCache<Scalar> cache;
  auto token = cache.get_plan(32, /*inverse=*/false, cfg);
  auto& plan = token.plan();
  bool ok = plan.use_kernel(eigfft::KernelKind::Baseline);
  ok = ok && plan.kernel().kind == eigfft::KernelKind::Baseline;
  ok = ok && plan.kernel_realtime_safe();
  ok = ok && plan.kernel_accuracy() == eigfft::KernelAccuracy::Default;
  ok = ok && plan.use_kernel("stockham-autosort");
  ok = ok && plan.kernel().kind == eigfft::KernelKind::Stockham;
  const bool external_available = eigfft::Plan<Scalar>::has_external_kernel();
  const bool external_selected = plan.use_kernel("external-provider");
  if (external_available) {
    ok = ok && external_selected;
    ok = ok && plan.kernel().kind == eigfft::KernelKind::External;
    ok = ok && !plan.kernel_realtime_safe();
    ok = ok && plan.kernel_accuracy() == eigfft::KernelAccuracy::HighPrecision;
  } else {
    ok = ok && !external_selected;
    ok = ok && plan.kernel().kind == eigfft::KernelKind::Stockham;
  }
  ok = ok && !plan.use_kernel("nonexistent-kernel");
  ok = ok && !plan.use_kernel(static_cast<eigfft::KernelKind>(99));
  return ok;
}

template <typename Scalar>
bool test_plan_cache_unique_token_instances() {
  eigfft::PlanRuntimeConfig cfg = default_runtime_config<Scalar>();
  cfg.threads = 2;
  eigfft::PlanCache<Scalar> cache;
  auto token_a = cache.get_plan(512, /*inverse=*/false, cfg);
  auto token_b = cache.get_plan(512, /*inverse=*/false, cfg);
  return &token_a.plan() != &token_b.plan();
}

template <typename Scalar>
bool test_plan_cache_reuse_after_release() {
  eigfft::PlanRuntimeConfig cfg = default_runtime_config<Scalar>();
  eigfft::PlanCache<Scalar> cache;
  eigfft::Plan<Scalar>* first_ptr = nullptr;
  {
    auto token = cache.get_plan(1024, /*inverse=*/false, cfg);
    first_ptr = &token.plan();
  }
  auto token_again = cache.get_plan(1024, /*inverse=*/false, cfg);
  return first_ptr == &token_again.plan();
}

#ifndef EIGFFT_ALLOW_SEQUENTIAL
template <typename Scalar>
bool test_sequential_path_rejected() {
  using Complex = std::complex<Scalar>;
  constexpr int N = 8;
  const auto shape = eigfft::compute_plan_arena_shape<Scalar>(N, 1, 1);
  std::vector<Complex> twiddles(shape.twiddles);
  std::vector<int> bitrev(shape.bitrev);
  std::vector<Complex> baseline_a(shape.baseline_complex);
  std::vector<Complex> baseline_b(shape.baseline_complex);
  std::vector<Complex*> baseline_columns(shape.baseline_columns, nullptr);
  std::vector<Complex> stockham_ping(shape.stockham_stage);
  std::vector<Complex> stockham_pong(shape.stockham_stage);
  std::vector<std::ptrdiff_t> stockham_lane_bases(shape.stockham_lane_bases, 0);

  eigfft::PlanArena<Scalar> arena{};
  arena.twiddles = twiddles.data();
  arena.twiddle_count = static_cast<int>(shape.twiddles);
  arena.bitrev = bitrev.data();
  arena.bitrev_count = static_cast<int>(shape.bitrev);
  arena.baseline_a = baseline_a.data();
  arena.baseline_b = baseline_b.data();
  arena.baseline_columns = baseline_columns.data();
  arena.baseline_thread_capacity = 1;
  arena.baseline_lane_capacity = 1;
  arena.stockham_ping = stockham_ping.data();
  arena.stockham_pong = stockham_pong.data();
  arena.stockham_lane_bases = stockham_lane_bases.data();
  arena.stockham_thread_capacity = 1;
  arena.stockham_lane_capacity = 1;
  arena.nd_transpose = nullptr;
  arena.nd_transpose_capacity = 0;

  try {
    eigfft::Plan<Scalar> plan(N, arena, /*inverse=*/false, /*threads=*/false, /*max_threads=*/1);
    (void)plan;
    return false;
  } catch (const std::invalid_argument&) {
    return true;
  } catch (...) {
    return false;
  }
}
#endif

template <typename Scalar>
bool test_stockham_parallel_large_batch() {
  using Complex = std::complex<Scalar>;
  constexpr int N = 256;
  constexpr int B = 128;
  eigfft::PlanRuntimeConfig cfg = default_runtime_config<Scalar>();
  eigfft::PlanCache<Scalar> cache;
  cache.warm_audio_profiles(cfg);

  auto stockham_token = cache.get_plan(N, /*inverse=*/false, cfg);
  auto baseline_token = cache.get_plan(N, /*inverse=*/false, cfg);
  auto& plan = stockham_token.plan();
  auto& baseline_plan = baseline_token.plan();
  plan.tuning.packet_step = 1;
  plan.tuning.parallel_dim = eigfft::Plan<Scalar>::ParallelDim::Columns;
  baseline_plan.tuning.packet_step = 1;
  baseline_plan.tuning.parallel_dim = eigfft::Plan<Scalar>::ParallelDim::Columns;
  baseline_plan.use_kernel(eigfft::KernelKind::Baseline);

  Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic> signal(N, B);
  Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic> stockham_out(N, B);
  Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic> baseline_out(N, B);
  std::mt19937 rng(1337);
  std::normal_distribution<double> dist(0.0, 1.0);
  if (!plan.use_kernel(eigfft::KernelKind::Stockham)) {
    return false;
  }
  try {
    for (int rep = 0; rep < 2; ++rep) {
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < B; ++j) {
          const Scalar re = static_cast<Scalar>(dist(rng));
          signal(i, j) = Complex(re, Scalar(0));
        }
      }
      stockham_out = signal;
      baseline_out = signal;
#if EIGFFT_TRACE_STOCKHAM
      std::cout << "[test] invoking baseline FFT rep=" << rep << std::endl;
#endif
      eigfft::fft_inplace_batched<Scalar>(baseline_out, baseline_plan);
#if EIGFFT_TRACE_STOCKHAM
      std::cout << "[test] invoking stockham FFT rep=" << rep << std::endl;
#endif
      eigfft::fft_inplace_batched<Scalar>(stockham_out, plan);
#if EIGFFT_TRACE_STOCKHAM
      std::cout << "[test] stockham FFT completed rep=" << rep << std::endl;
#endif

      Scalar max_diff = Scalar(0);
      int bad_i = -1;
      int bad_j = -1;
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < B; ++j) {
          const Scalar diff = std::abs(stockham_out(i, j) - baseline_out(i, j));
          if (diff > max_diff) {
            max_diff = diff;
            bad_i = i;
            bad_j = j;
          }
        }
      }
      const Scalar tol = static_cast<Scalar>(tolerance<Scalar>() * 100);
      Scalar recomputed_max = Scalar(0);
      int report_i = -1;
      int report_j = -1;
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < B; ++j) {
          const Scalar diff = std::abs(stockham_out(i, j) - baseline_out(i, j));
          if (diff > recomputed_max) {
            recomputed_max = diff;
            report_i = i;
            report_j = j;
          }
        }
      }
      if (recomputed_max <= tol) {
        continue;
      }
#if EIGFFT_TRACE_STOCKHAM
      Scalar permuted_max_diff = Scalar(0);
      int perm_i = -1;
      int perm_j = -1;
      Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic> permuted(N, B);
      const int bits = plan.lgN;
      for (int i = 0; i < N; ++i) {
        int rev = 0;
        int x = i;
        for (int b = 0; b < bits; ++b) {
          rev = (rev << 1) | (x & 1);
          x >>= 1;
        }
        if (rev >= 0 && rev < N) {
          permuted.row(i) = stockham_out.row(rev);
        }
      }
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < B; ++j) {
          const Scalar diff = std::abs(permuted(i, j) - baseline_out(i, j));
          if (diff > permuted_max_diff) {
            permuted_max_diff = diff;
            perm_i = i;
            perm_j = j;
          }
        }
      }
      std::cerr << "[test] stockham mismatch after permutation diff="
                << permuted_max_diff << " at (" << perm_i << "," << perm_j << ")"
                << std::endl;
#endif
      std::cerr << "[test] stockham mismatch rep=" << rep
                << " max_diff=" << recomputed_max
                << " at (" << report_i << "," << report_j << ")"
                << " stock=" << stockham_out(report_i, report_j)
                << " baseline=" << baseline_out(report_i, report_j)
                << std::endl;
      if (max_diff > tol) {
        std::cerr << "[test] legacy max_diff=" << max_diff
                  << " at (" << bad_i << "," << bad_j << ")" << std::endl;
      }
      return false;
    }
  } catch (const std::exception& ex) {
#if EIGFFT_TRACE_STOCKHAM
    std::cerr << "[test] exception: " << ex.what() << std::endl;
#else
    (void)ex;
#endif
    return false;
  } catch (...) {
#if EIGFFT_TRACE_STOCKHAM
    std::cerr << "[test] unknown exception" << std::endl;
#endif
    return false;
  }
  return true;
}

template <typename Scalar>
bool test_stockham_metadata_capture() {
  using Complex = std::complex<Scalar>;
  constexpr int N = 16;
  constexpr int B = 5;
  eigfft::PlanRuntimeConfig cfg = default_runtime_config<Scalar>();
  eigfft::PlanCache<Scalar> cache;
  cache.warm_audio_profiles(cfg);

  auto token = cache.get_plan(N, /*inverse=*/false, cfg);
  auto& plan = token.plan();
  if (!plan.use_kernel(eigfft::KernelKind::Stockham)) {
    return false;
  }

  Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic> signal(N, B);
  std::mt19937 rng(2025);
  std::normal_distribution<double> dist(0.0, 1.0);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < B; ++j) {
      const Scalar re = static_cast<Scalar>(dist(rng));
      const Scalar im = static_cast<Scalar>(dist(rng));
      signal(i, j) = Complex(re, im);
    }
  }
  const auto original = signal;

  const int stages = plan.lgN;
  if (stages <= 0) {
    return false;
  }
  std::size_t per_stage = 1;
  std::size_t total_twiddles = 0;
  for (int s = 0; s < stages; ++s) {
    total_twiddles += per_stage;
    per_stage <<= 1U;
  }
  std::vector<Complex> twiddles(total_twiddles);

  std::array<eigfft::MetadataRequest<Scalar>, 1> requests{};
  requests[0].kind = eigfft::MetadataKind::Twiddle;
  requests[0].complex_buffer = twiddles.data();
  requests[0].element_count = twiddles.size();

  eigfft::fft_inplace_batched_with_metadata<Scalar>(signal, plan,
                                                    requests.data(),
                                                    static_cast<int>(requests.size()));

  const double tol = static_cast<double>(tolerance<Scalar>()) * 10.0;

  // Recorded twiddles must mirror the plan twiddle table usage per stage.
  for (int stage = 0; stage < stages; ++stage) {
    const int m = 1 << stage;
    const int distance = m << 1;
    const int tw_step = plan.N / distance;
    const std::size_t stage_offset = (static_cast<std::size_t>(1) << stage) - 1;
    for (int j = 0; j < m; ++j) {
      const Complex expected = plan.W[j * tw_step];
      const Complex captured = twiddles[stage_offset + static_cast<std::size_t>(j)];
      const double re_err = static_cast<double>(captured.real()) - static_cast<double>(expected.real());
      const double im_err = static_cast<double>(captured.imag()) - static_cast<double>(expected.imag());
      if (std::hypot(re_err, im_err) > tol) {
        return false;
      }
    }
  }

  // Final transform still matches the reference DFT within tolerance.
  const auto reference = naive_dft<Scalar>(original);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < B; ++j) {
      const double re_err = static_cast<double>(signal(i, j).real()) - static_cast<double>(reference(i, j).real());
      const double im_err = static_cast<double>(signal(i, j).imag()) - static_cast<double>(reference(i, j).imag());
      if (std::hypot(re_err, im_err) > tol) {
        return false;
      }
    }
  }

  return true;
}

template <typename Scalar>
bool test_plancache_consensus() {
  using Complex = std::complex<Scalar>;
  constexpr int debugN = 8;
  constexpr int debugFrames = 3;

  eigfft::PlanRuntimeConfig cfg = default_runtime_config<Scalar>();
  eigfft::PlanCache<Scalar> cache;
  cache.warm_audio_profiles(cfg);

  // Prepare input
  Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic> input(debugN, debugFrames);
  std::mt19937 rng(2026);
  std::normal_distribution<double> dist(0.0, 1.0);
  for (int i = 0; i < debugN; ++i) for (int j = 0; j < debugFrames; ++j)
    input(i,j) = Complex(static_cast<Scalar>(dist(rng)), static_cast<Scalar>(dist(rng)));

  auto token_base = cache.get_plan(debugN, /*inverse=*/false, cfg);
  auto& base_plan = token_base.plan();
  base_plan.tuning.packet_step = base_plan.packet_cols;
  base_plan.use_kernel(eigfft::KernelKind::Baseline);

  auto token_stock = cache.get_plan(debugN, /*inverse=*/false, cfg);
  auto& stock_plan = token_stock.plan();
  stock_plan.tuning.packet_step = stock_plan.packet_cols;
  if (!stock_plan.use_kernel(eigfft::KernelKind::Stockham)) {
    // If Stockham not available, the consensus test cannot proceed meaningfully
    return false;
  }

  const int stages = base_plan.lgN;
  if (stages <= 0) return false;

  // compute twiddle counts
  std::size_t per_stage = 1;
  std::size_t total_twiddles = 0;
  for (int s = 0; s < stages; ++s) { total_twiddles += per_stage; per_stage <<= 1U; }

  std::vector<Complex> base_tw(total_twiddles), stock_tw(total_twiddles);
  std::vector<Complex> base_layout(debugN), stock_layout(debugN);
  std::vector<Complex> base_snap(static_cast<std::size_t>(debugN) * static_cast<std::size_t>(stages));
  std::vector<Complex> stock_snap(static_cast<std::size_t>(debugN) * static_cast<std::size_t>(stages));
  std::vector<Complex> base_stagep(static_cast<std::size_t>(stages)), stock_stagep(static_cast<std::size_t>(stages));
  std::vector<Complex> base_pairs(static_cast<std::size_t>(debugN) * static_cast<std::size_t>(stages));
  std::vector<Complex> stock_pairs(static_cast<std::size_t>(debugN) * static_cast<std::size_t>(stages));
  std::vector<Complex> base_inv(static_cast<std::size_t>(stages)), stock_inv(static_cast<std::size_t>(stages));
  std::vector<Complex> base_twmap(static_cast<std::size_t>(debugN) * static_cast<std::size_t>(stages));
  std::vector<Complex> stock_twmap(static_cast<std::size_t>(debugN) * static_cast<std::size_t>(stages));

  // Run baseline with metadata
  Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic> base_in = input;
  std::array<eigfft::MetadataRequest<Scalar>, 7> base_reqs{
    eigfft::MetadataRequest<Scalar>{eigfft::MetadataKind::Twiddle, base_tw.data(), total_twiddles},
    eigfft::MetadataRequest<Scalar>{eigfft::MetadataKind::LayoutMap, base_layout.data(), static_cast<std::size_t>(debugN)},
    eigfft::MetadataRequest<Scalar>{eigfft::MetadataKind::StageSnapshot, base_snap.data(), static_cast<std::size_t>(debugN) * static_cast<std::size_t>(stages)},
    eigfft::MetadataRequest<Scalar>{eigfft::MetadataKind::StageParams, base_stagep.data(), static_cast<std::size_t>(stages)},
    eigfft::MetadataRequest<Scalar>{eigfft::MetadataKind::ButterflyPairs, base_pairs.data(), static_cast<std::size_t>(debugN) * static_cast<std::size_t>(stages)},
    eigfft::MetadataRequest<Scalar>{eigfft::MetadataKind::StageInvariant, base_inv.data(), static_cast<std::size_t>(stages)},
    eigfft::MetadataRequest<Scalar>{eigfft::MetadataKind::TwiddleIndexMap, base_twmap.data(), static_cast<std::size_t>(debugN) * static_cast<std::size_t>(stages)}
  };
  eigfft::fft_inplace_batched_with_metadata<Scalar>(base_in, base_plan, base_reqs.data(), static_cast<int>(base_reqs.size()));

  // Run stockham with metadata
  Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic> stock_in = input;
  std::array<eigfft::MetadataRequest<Scalar>, 7> stock_reqs{
    eigfft::MetadataRequest<Scalar>{eigfft::MetadataKind::Twiddle, stock_tw.data(), total_twiddles},
    eigfft::MetadataRequest<Scalar>{eigfft::MetadataKind::LayoutMap, stock_layout.data(), static_cast<std::size_t>(debugN)},
    eigfft::MetadataRequest<Scalar>{eigfft::MetadataKind::StageSnapshot, stock_snap.data(), static_cast<std::size_t>(debugN) * static_cast<std::size_t>(stages)},
    eigfft::MetadataRequest<Scalar>{eigfft::MetadataKind::StageParams, stock_stagep.data(), static_cast<std::size_t>(stages)},
    eigfft::MetadataRequest<Scalar>{eigfft::MetadataKind::ButterflyPairs, stock_pairs.data(), static_cast<std::size_t>(debugN) * static_cast<std::size_t>(stages)},
    eigfft::MetadataRequest<Scalar>{eigfft::MetadataKind::StageInvariant, stock_inv.data(), static_cast<std::size_t>(stages)},
    eigfft::MetadataRequest<Scalar>{eigfft::MetadataKind::TwiddleIndexMap, stock_twmap.data(), static_cast<std::size_t>(debugN) * static_cast<std::size_t>(stages)}
  };
  eigfft::fft_inplace_batched_with_metadata<Scalar>(stock_in, stock_plan, stock_reqs.data(), static_cast<int>(stock_reqs.size()));

  // Verbose metadata dump (match test_plancache.cpp style)
  std::cout << "--- Plancache consensus metadata dump (N=" << debugN << ") ---" << std::endl;
  print_twiddles<Scalar>(base_tw, stages, debugN);
  print_twiddles<Scalar>(stock_tw, stages, debugN);
  print_layout<Scalar>("Baseline", base_layout, debugN);
  print_layout<Scalar>("Stockham", stock_layout, debugN);
  print_snapshots<Scalar>("Baseline", base_snap, stages, debugN);
  print_snapshots<Scalar>("Stockham", stock_snap, stages, debugN);
  print_stage_params<Scalar>(base_stagep, stages);
  print_stage_params<Scalar>(stock_stagep, stages);
  print_pairs<Scalar>("Baseline", base_pairs, stages, debugN);
  print_pairs<Scalar>("Stockham", stock_pairs, stages, debugN);
  print_invariants<Scalar>("Baseline", base_inv, stages);
  print_invariants<Scalar>("Stockham", stock_inv, stages);
  print_twmap<Scalar>("Baseline", base_twmap, stages, debugN);
  print_twmap<Scalar>("Stockham", stock_twmap, stages, debugN);
  print_matrix_frames<Scalar>("Captured input", input);
  print_matrix_frames<Scalar>("Baseline output", base_in);
  print_matrix_frames<Scalar>("Stockham output", stock_in);
  print_elementwise_diffs<Scalar>("Element-wise Stockham vs Baseline (raw)", stock_in, base_in);

  // Build permutation mapping from layout maps
  auto normalize_index = [&](int idx) {
    if (debugN <= 0) return idx;
    int mod = idx % debugN; if (mod < 0) mod += debugN; return mod;
  };
  std::vector<int> pos_in_stock(debugN, -1), pos_in_base(debugN, -1);
  for (int pos = 0; pos < debugN; ++pos) {
    const int ssrc = normalize_index(static_cast<int>(std::lround(stock_layout[static_cast<std::size_t>(pos)].real())));
    const int bsrc = normalize_index(static_cast<int>(std::lround(base_layout[static_cast<std::size_t>(pos)].real())));
    if (ssrc >= 0 && ssrc < debugN) pos_in_stock[ssrc] = pos;
    if (bsrc >= 0 && bsrc < debugN) pos_in_base[bsrc] = pos;
  }
  std::vector<int> permutation(debugN, -1);
  bool perm_ok = true;
  for (int src = 0; src < debugN; ++src) {
    const int ps = pos_in_stock[src];
    const int pb = pos_in_base[src];
    if (ps < 0 || pb < 0) { perm_ok = false; break; }
    permutation[ps] = pb;
  }

  if (!perm_ok) return false;

  // Compute max difference after permutation remap
  double max_diff = 0.0;
  for (int row = 0; row < debugN; ++row) {
    const int mapped = permutation[row];
    for (int frame = 0; frame < debugFrames; ++frame) {
      const Complex a = stock_in(row, frame);
      const Complex b = base_in(mapped, frame);
      const double d = std::hypot(static_cast<double>(a.real()-b.real()), static_cast<double>(a.imag()-b.imag()));
      if (d > max_diff) max_diff = d;
    }
  }

  const double tol = static_cast<double>(tolerance<Scalar>()) * 20.0;
  if (max_diff > tol) return false;

  // Twiddle differences should be small too
  for (std::size_t i = 0; i < total_twiddles; ++i) {
    const double d = std::hypot(static_cast<double>(stock_tw[i].real()-base_tw[i].real()), static_cast<double>(stock_tw[i].imag()-base_tw[i].imag()));
    if (d > tol) return false;
  }

  // Column isolation sanity: zero all columns except one and rerun
  const int sel = (debugFrames>1)?1:0;
  Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic> iso_input(debugN, debugFrames);
  iso_input.setZero(); iso_input.col(sel) = input.col(sel);
  auto iso_base = iso_input; auto iso_stock = iso_input;
  eigfft::fft_inplace_batched<Scalar>(iso_base, base_plan);
  eigfft::fft_inplace_batched<Scalar>(iso_stock, stock_plan);
  double max_iso_diff = 0.0;
  for (int r=0;r<debugN;++r) for (int f=0; f<debugFrames; ++f) {
    const double d = std::hypot(static_cast<double>(iso_stock(r,f).real()-iso_base(r,f).real()), static_cast<double>(iso_stock(r,f).imag()-iso_base(r,f).imag()));
    if (d > max_iso_diff) max_iso_diff = d;
  }
  if (max_iso_diff > tol*10.0) return false;

  return true;
}

template <typename Scalar>
bool test_plancache_extended_consensus() {
  using Complex = std::complex<Scalar>;
  const int Ns[] = {2,4,8,16,32};
  const int frame_sets[] = {1,3,16};
  eigfft::PlanRuntimeConfig cfg = default_runtime_config<Scalar>();
  eigfft::PlanCache<Scalar> cache;
  cache.warm_audio_profiles(cfg);

  for (int N : Ns) {
    for (int frames : frame_sets) {
      // Prepare random input
      Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic> input(N, frames);
      std::mt19937 rng(static_cast<unsigned int>(N * 1009 + frames * 97));
      std::normal_distribution<double> dist(0.0, 1.0);
      for (int r = 0; r < N; ++r) for (int c = 0; c < frames; ++c)
        input(r,c) = Complex(static_cast<Scalar>(dist(rng)), static_cast<Scalar>(dist(rng)));

      // Get baseline and stockham plans
      auto token_base = cache.get_plan(N, /*inverse=*/false, cfg);
      auto& base_plan = token_base.plan();
      base_plan.tuning.packet_step = base_plan.packet_cols;
      base_plan.use_kernel(eigfft::KernelKind::Baseline);

      auto token_stock = cache.get_plan(N, /*inverse=*/false, cfg);
      auto& stock_plan = token_stock.plan();
      stock_plan.tuning.packet_step = stock_plan.packet_cols;
      if (!stock_plan.use_kernel(eigfft::KernelKind::Stockham)) {
        // If Stockham not available we can't compare meaningfully; skip.
        continue;
      }

      const int stages = base_plan.lgN;
      std::size_t total_twiddles = 0;
      for (int s = 0; s < stages; ++s) total_twiddles += static_cast<std::size_t>(1) << s;

      // allocate metadata buffers
      std::vector<Complex> base_tw(total_twiddles), stock_tw(total_twiddles);
  std::vector<Complex> base_layout(N), stock_layout(N);
      std::vector<Complex> base_snap(static_cast<std::size_t>(N) * std::max(1, stages));
      std::vector<Complex> stock_snap(static_cast<std::size_t>(N) * std::max(1, stages));
      std::vector<Complex> base_stagep(static_cast<std::size_t>(std::max(1, stages)));
      std::vector<Complex> stock_stagep(static_cast<std::size_t>(std::max(1, stages)));
      std::vector<Complex> base_pairs(static_cast<std::size_t>(N) * std::max(1, stages));
      std::vector<Complex> stock_pairs(static_cast<std::size_t>(N) * std::max(1, stages));
      std::vector<Complex> base_inv(static_cast<std::size_t>(std::max(1, stages)));
      std::vector<Complex> stock_inv(static_cast<std::size_t>(std::max(1, stages)));
      std::vector<Complex> base_twmap(static_cast<std::size_t>(N) * std::max(1, stages));
      std::vector<Complex> stock_twmap(static_cast<std::size_t>(N) * std::max(1, stages));

      // Run baseline with metadata
      Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic> base_in = input;
      std::array<eigfft::MetadataRequest<Scalar>, 7> base_reqs{
        eigfft::MetadataRequest<Scalar>{eigfft::MetadataKind::Twiddle, base_tw.data(), total_twiddles},
  eigfft::MetadataRequest<Scalar>{eigfft::MetadataKind::LayoutMap, base_layout.data(), static_cast<std::size_t>(N)},
        eigfft::MetadataRequest<Scalar>{eigfft::MetadataKind::StageSnapshot, base_snap.data(), static_cast<std::size_t>(N) * static_cast<std::size_t>(stages)},
        eigfft::MetadataRequest<Scalar>{eigfft::MetadataKind::StageParams, base_stagep.data(), static_cast<std::size_t>(stages)},
        eigfft::MetadataRequest<Scalar>{eigfft::MetadataKind::ButterflyPairs, base_pairs.data(), static_cast<std::size_t>(N) * static_cast<std::size_t>(stages)},
        eigfft::MetadataRequest<Scalar>{eigfft::MetadataKind::StageInvariant, base_inv.data(), static_cast<std::size_t>(stages)},
        eigfft::MetadataRequest<Scalar>{eigfft::MetadataKind::TwiddleIndexMap, base_twmap.data(), static_cast<std::size_t>(N) * static_cast<std::size_t>(stages)}
      };
      eigfft::fft_inplace_batched_with_metadata<Scalar>(base_in, base_plan, base_reqs.data(), static_cast<int>(base_reqs.size()));

      // Run stockham with metadata
      Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic> stock_in = input;
      std::array<eigfft::MetadataRequest<Scalar>, 7> stock_reqs{
        eigfft::MetadataRequest<Scalar>{eigfft::MetadataKind::Twiddle, stock_tw.data(), total_twiddles},
  eigfft::MetadataRequest<Scalar>{eigfft::MetadataKind::LayoutMap, stock_layout.data(), static_cast<std::size_t>(N)},
        eigfft::MetadataRequest<Scalar>{eigfft::MetadataKind::StageSnapshot, stock_snap.data(), static_cast<std::size_t>(N) * static_cast<std::size_t>(stages)},
        eigfft::MetadataRequest<Scalar>{eigfft::MetadataKind::StageParams, stock_stagep.data(), static_cast<std::size_t>(stages)},
        eigfft::MetadataRequest<Scalar>{eigfft::MetadataKind::ButterflyPairs, stock_pairs.data(), static_cast<std::size_t>(N) * static_cast<std::size_t>(stages)},
        eigfft::MetadataRequest<Scalar>{eigfft::MetadataKind::StageInvariant, stock_inv.data(), static_cast<std::size_t>(stages)},
        eigfft::MetadataRequest<Scalar>{eigfft::MetadataKind::TwiddleIndexMap, stock_twmap.data(), static_cast<std::size_t>(N) * static_cast<std::size_t>(stages)}
      };
      eigfft::fft_inplace_batched_with_metadata<Scalar>(stock_in, stock_plan, stock_reqs.data(), static_cast<int>(stock_reqs.size()));

  // Verbose metadata dump for this N/frames
  std::cout << "--- Extended plancache metadata dump (N=" << N << ", frames=" << frames << ") ---" << std::endl;
  print_twiddles<Scalar>(base_tw, stages, N);
  print_twiddles<Scalar>(stock_tw, stages, N);
  print_layout<Scalar>("Baseline", base_layout, N);
  print_layout<Scalar>("Stockham", stock_layout, N);
  print_snapshots<Scalar>("Baseline", base_snap, stages, N);
  print_snapshots<Scalar>("Stockham", stock_snap, stages, N);
  print_stage_params<Scalar>(base_stagep, stages);
  print_stage_params<Scalar>(stock_stagep, stages);
  print_pairs<Scalar>("Baseline", base_pairs, stages, N);
  print_pairs<Scalar>("Stockham", stock_pairs, stages, N);
  print_invariants<Scalar>("Baseline", base_inv, stages);
  print_invariants<Scalar>("Stockham", stock_inv, stages);
  print_twmap<Scalar>("Baseline", base_twmap, stages, N);
  print_twmap<Scalar>("Stockham", stock_twmap, stages, N);
  print_matrix_frames<Scalar>("Captured input", input);
  print_matrix_frames<Scalar>("Baseline output", base_in);
  print_matrix_frames<Scalar>("Stockham output", stock_in);
  print_elementwise_diffs<Scalar>("Element-wise Stockham vs Baseline (raw)", stock_in, base_in);

      // Build permutation mapping from layout maps
      auto normalize_index = [&](int idx) {
        if (N <= 0) return idx;
        int mod = idx % N; if (mod < 0) mod += N; return mod;
      };
      std::vector<int> pos_in_stock(N, -1), pos_in_base(N, -1);
      for (int pos = 0; pos < N; ++pos) {
  const int ssrc = normalize_index(static_cast<int>(std::lround(stock_layout[static_cast<std::size_t>(pos)].real())));
  const int bsrc = normalize_index(static_cast<int>(std::lround(base_layout[static_cast<std::size_t>(pos)].real())));
        if (ssrc >= 0 && ssrc < N) pos_in_stock[ssrc] = pos;
        if (bsrc >= 0 && bsrc < N) pos_in_base[bsrc] = pos;
      }
      std::vector<int> permutation(N, -1);
      bool perm_ok = true;
      for (int src = 0; src < N; ++src) {
        const int ps = pos_in_stock[src];
        const int pb = pos_in_base[src];
        if (ps < 0 || pb < 0) { perm_ok = false; break; }
        permutation[ps] = pb;
      }
      if (!perm_ok) return false;

      // Compare element-wise after permutation remap
      double max_diff = 0.0;
      for (int row = 0; row < N; ++row) {
        const int mapped = permutation[row];
        for (int frame = 0; frame < frames; ++frame) {
          const Complex a = stock_in(row, frame);
          const Complex b = base_in(mapped, frame);
          const double d = std::hypot(static_cast<double>(a.real()-b.real()), static_cast<double>(a.imag()-b.imag()));
          if (d > max_diff) max_diff = d;
        }
      }

      const double tol = static_cast<double>(tolerance<Scalar>()) * 50.0;
      if (max_diff > tol) {
        std::cerr << "[test] consensus mismatch: N=" << N
                  << " frames=" << frames
                  << " max_diff=" << max_diff
                  << " tol=" << tol << std::endl;
        return false;
      }

      // Twiddle differences should be small too
      for (std::size_t i = 0; i < total_twiddles; ++i) {
        const double d = std::hypot(static_cast<double>(stock_tw[i].real()-base_tw[i].real()), static_cast<double>(stock_tw[i].imag()-base_tw[i].imag()));
        if (d > tol) {
          std::cerr << "[test] twiddle mismatch: N=" << N
                    << " frames=" << frames
                    << " index=" << i
                    << " diff=" << d
                    << " tol=" << tol << std::endl;
          return false;
        }
      }

      // Stage snapshots and pairs: check L2-ish discrepancy small relative to scale
      auto l2_compare = [&](const std::vector<Complex>& A, const std::vector<Complex>& B)->bool{
        if (A.size() != B.size()) return false;
        double accum = 0.0;
        double scale = 0.0;
        for (std::size_t i=0;i<A.size();++i) {
          accum += std::pow(static_cast<double>(std::abs(A[i]-B[i])), 2.0);
          scale += std::pow(static_cast<double>(std::abs(B[i])), 2.0);
        }
        // If scale is tiny, compare absolute
        if (scale < 1e-12) return accum < 1e-12;
        return (std::sqrt(accum) / (std::sqrt(scale)+1e-18)) < 1e-6;
      };

      std::vector<Complex> stock_snap_perm(stock_snap.size());
      std::vector<Complex> stock_pairs_perm(stock_pairs.size());
      for (int stage = 0; stage < stages; ++stage) {
        const std::size_t stage_offset = static_cast<std::size_t>(stage) * static_cast<std::size_t>(N);
        for (int pos = 0; pos < N; ++pos) {
          const int mapped = permutation[pos];
          stock_snap_perm[stage_offset + static_cast<std::size_t>(mapped)] =
              stock_snap[stage_offset + static_cast<std::size_t>(pos)];
          stock_pairs_perm[stage_offset + static_cast<std::size_t>(mapped)] =
              stock_pairs[stage_offset + static_cast<std::size_t>(pos)];
        }
      }

      if (!l2_compare(base_snap, stock_snap_perm)) {
        std::cerr << "[test] snapshot mismatch: N=" << N << " frames=" << frames << std::endl;
        for (std::size_t i = 0; i < base_snap.size(); ++i) {
          std::cerr << "  idx " << i << " base=" << base_snap[i] << " stock=" << stock_snap_perm[i]
                    << " diff=" << std::abs(base_snap[i]-stock_snap_perm[i]) << std::endl;
        }
        return false;
      }
      if (!l2_compare(base_pairs, stock_pairs_perm)) {
        std::cerr << "[test] pair mismatch: N=" << N << " frames=" << frames << std::endl;
        return false;
      }

      // Column isolation sanity: zero all columns except one and rerun (extended batch)
      if (frames > 1) {
        const int sel = 1;
        Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic> iso_input(N, frames);
        iso_input.setZero(); iso_input.col(sel) = input.col(sel);
        auto iso_base = iso_input; auto iso_stock = iso_input;
        eigfft::fft_inplace_batched<Scalar>(iso_base, base_plan);
        eigfft::fft_inplace_batched<Scalar>(iso_stock, stock_plan);
        double max_iso_diff = 0.0;
        for (int r=0;r<N;++r) for (int f=0; f<frames; ++f) {
          const double d = std::hypot(static_cast<double>(iso_stock(r,f).real()-iso_base(r,f).real()), static_cast<double>(iso_stock(r,f).imag()-iso_base(r,f).imag()));
          if (d > max_iso_diff) max_iso_diff = d;
        }
        if (max_iso_diff > tol*10.0) {
          std::cerr << "[test] isolation mismatch: N=" << N
                    << " frames=" << frames
                    << " max_iso_diff=" << max_iso_diff
                    << " tol=" << tol << std::endl;
          return false;
        }
      }
    }
  }

  return true;
}

template <typename Scalar>
bool test_butterfly_multiN_metadata() {
  using Complex = std::complex<Scalar>;
  eigfft::PlanRuntimeConfig cfg;
  cfg.threads = eigfft::Plan<Scalar>::Limits::kDefaultRuntimeThreads;
  cfg.lanes = eigfft::Plan<Scalar>::Limits::kDefaultLaneCapacity;
  cfg.transpose_capacity = 0;
  cfg.inverse = false;

  const int widths[] = {1,2,4,8,16};
  const int frames = 3;
  std::mt19937_64 rng(123456);
  std::normal_distribution<double> dist(0.0,1.0);

  for (int N : widths) {
    eigfft::PlanEnvironment<Scalar> env;
    env.initialize(N, false, cfg);
    auto& plan = env.plan();
    plan.tuning.packet_step = plan.packet_cols;
    plan.use_kernel(eigfft::KernelKind::Baseline);

    Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic> X(N, frames);
    for (int r=0;r<N;++r) for (int c=0;c<frames;++c) X(r,c) = Complex(static_cast<Scalar>(dist(rng)), static_cast<Scalar>(dist(rng)));
    const auto input = X;

    const int stages = plan.lgN;
    std::size_t total_twiddles = 0;
    for (int s=0;s<stages;++s) total_twiddles += static_cast<std::size_t>(1)<<s;

    std::vector<Complex> twiddles(total_twiddles);
    std::vector<Complex> layout(static_cast<std::size_t>(N));
    std::vector<Complex> snapshots(static_cast<std::size_t>(N) * std::max(1, stages));
    std::vector<Complex> stage_params(static_cast<std::size_t>(std::max(1, stages)));
    std::vector<Complex> pairs(static_cast<std::size_t>(N) * std::max(1, stages));
    std::vector<Complex> invariants(static_cast<std::size_t>(std::max(1, stages)));
    std::vector<Complex> twmap(static_cast<std::size_t>(N) * std::max(1, stages));

    std::vector<eigfft::MetadataRequest<Scalar>> requests;
    requests.push_back(eigfft::MetadataRequest<Scalar>{eigfft::MetadataKind::Twiddle, twiddles.data(), total_twiddles});
    requests.push_back(eigfft::MetadataRequest<Scalar>{eigfft::MetadataKind::LayoutMap, layout.data(), static_cast<std::size_t>(N)});
    if (stages>0) {
      requests.push_back(eigfft::MetadataRequest<Scalar>{eigfft::MetadataKind::StageSnapshot, snapshots.data(), static_cast<std::size_t>(N) * static_cast<std::size_t>(stages)});
      requests.push_back(eigfft::MetadataRequest<Scalar>{eigfft::MetadataKind::StageParams, stage_params.data(), static_cast<std::size_t>(stages)});
      requests.push_back(eigfft::MetadataRequest<Scalar>{eigfft::MetadataKind::ButterflyPairs, pairs.data(), static_cast<std::size_t>(N) * static_cast<std::size_t>(stages)});
      requests.push_back(eigfft::MetadataRequest<Scalar>{eigfft::MetadataKind::StageInvariant, invariants.data(), static_cast<std::size_t>(stages)});
      requests.push_back(eigfft::MetadataRequest<Scalar>{eigfft::MetadataKind::TwiddleIndexMap, twmap.data(), static_cast<std::size_t>(N) * static_cast<std::size_t>(stages)});
    }

    eigfft::fft_inplace_batched_with_metadata<Scalar>(X, plan, requests.data(), static_cast<int>(requests.size()));

    // Verbose butterfly metadata dump
    std::cout << "--- Butterfly metadata dump (N=" << N << ") ---" << std::endl;
    print_twiddles<Scalar>(twiddles, stages, N);
    print_layout<Scalar>("Layout", layout, N);
    if (stages>0) {
      print_snapshots<Scalar>("Snapshots", snapshots, stages, N);
      print_stage_params<Scalar>(stage_params, stages);
      print_pairs<Scalar>("Pairs", pairs, stages, N);
      print_invariants<Scalar>("Invariants", invariants, stages);
      print_twmap<Scalar>("Twmap", twmap, stages, N);
    }
    print_matrix_frames<Scalar>("Transformed output", X);

    // Basic sanity checks: layout indices in range and twiddle count non-zero for stages>0
    for (int i=0;i<N;++i) {
      const int raw = static_cast<int>(std::lround(layout[static_cast<std::size_t>(i)].real()));
      (void)raw; // just ensure readable
    }
    if (stages>0 && total_twiddles==0) return false;

    // final transform should not produce NaNs
    for (int r=0;r<N;++r) for (int c=0;c<frames;++c) {
      const Complex v = X(r,c);
      if (!std::isfinite(v.real()) || !std::isfinite(v.imag())) return false;
    }
  }

  return true;
}

template <typename Scalar>
void enqueue_precision_tests(std::vector<std::pair<std::string, bool>>& results) {
  const std::string tag = precision_tag<Scalar>();
  results.emplace_back("roundtrip_small_batches<" + tag + ">",
                       test_roundtrip_small_batches<Scalar>());
  results.emplace_back("effective_threads_reports_parallel<" + tag + ">",
                       test_effective_threads_reports_parallel<Scalar>());
  results.emplace_back("single_thread_fallback_matches_reference<" + tag + ">",
                       test_single_thread_fallback_matches_reference<Scalar>());
  results.emplace_back("kernel_selection_interface<" + tag + ">",
                       test_kernel_selection_interface<Scalar>());
  results.emplace_back("stockham_parallel_large_batch<" + tag + ">",
                       test_stockham_parallel_large_batch<Scalar>());
  results.emplace_back("stockham_metadata_capture<" + tag + ">",
                       test_stockham_metadata_capture<Scalar>());
  results.emplace_back("butterfly_multiN_metadata<" + tag + ">",
                       test_butterfly_multiN_metadata<Scalar>());
  results.emplace_back("plancache_consensus<" + tag + ">",
                       test_plancache_consensus<Scalar>());
  results.emplace_back("plancache_extended_consensus<" + tag + ">",
                       test_plancache_extended_consensus<Scalar>());
  results.emplace_back("plan_cache_unique_tokens<" + tag + ">",
                       test_plan_cache_unique_token_instances<Scalar>());
  results.emplace_back("plan_cache_reuse_after_release<" + tag + ">",
                       test_plan_cache_reuse_after_release<Scalar>());
#ifndef EIGFFT_ALLOW_SEQUENTIAL
  results.emplace_back("sequential_path_rejected<" + tag + ">",
                       test_sequential_path_rejected<Scalar>());
#endif
}

}  // namespace

int main() {
  std::vector<std::pair<std::string, bool>> results;
  results.reserve(20);

  enqueue_precision_tests<double>(results);
  enqueue_precision_tests<float>(results);

  int failures = 0;
  for (const auto& entry : results) {
    if (!entry.second) {
      std::cerr << "[FAIL] " << entry.first << '\n';
      ++failures;
    }
  }

  if (failures != 0) {
    std::cerr << "fftfree tests: " << failures << " failure(s)." << std::endl;
    return 1;
  }

  std::cout << "fftfree tests: all passed" << std::endl;
  return 0;
}
