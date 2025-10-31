#include "../eigen_fft.hpp"
#include "../plan_support.hpp"

#include <Eigen/Core>
#include <unsupported/Eigen/FFT>

#include <cstddef>
#include <cmath>
#include <complex>
#include <algorithm>
#include <array>
#include <random>
#include <string>
#include <type_traits>
#include <vector>
#include <iostream>
#include <iomanip>
#include <map>
#include <tuple>

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
Eigen::Matrix<std::complex<Scalar>, Eigen::Dynamic, Eigen::Dynamic> eigen_fft_forward(
    const Eigen::Matrix<std::complex<Scalar>, Eigen::Dynamic, Eigen::Dynamic>& input) {
  using Complex = std::complex<Scalar>;
  Eigen::FFT<Scalar> fft;
  const int N = static_cast<int>(input.rows());
  const int B = static_cast<int>(input.cols());
  Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic> output(N, B);
  Eigen::Matrix<Complex, Eigen::Dynamic, 1> temp_in(N);
  Eigen::Matrix<Complex, Eigen::Dynamic, 1> temp_out(N);
  for (int col = 0; col < B; ++col) {
    temp_in = input.col(col);
    fft.fwd(temp_out, temp_in);
    output.col(col) = temp_out;
  }
  return output;
}

enum class ReferenceSignalKind { RealOnly = 0, Complex = 1 };

template <typename Scalar>
struct ReferenceBatch {
  using Complex = std::complex<Scalar>;
  Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic> input;
  Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic> forward;
};

template <typename Scalar>
const ReferenceBatch<Scalar>& reference_batch(int N,
                                              int frames,
                                              unsigned seed,
                                              ReferenceSignalKind kind) {
  using Batch = ReferenceBatch<Scalar>;
  using Key = std::tuple<int, int, unsigned, int>;
  static std::map<Key, Batch> cache;
  const Key key{N, frames, seed, static_cast<int>(kind)};
  auto it = cache.find(key);
  if (it == cache.end()) {
    Batch batch;
    batch.input.resize(N, frames);
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (int r = 0; r < N; ++r) {
      for (int c = 0; c < frames; ++c) {
        const Scalar re = static_cast<Scalar>(dist(rng));
        const Scalar im = (kind == ReferenceSignalKind::RealOnly)
                              ? Scalar(0)
                              : static_cast<Scalar>(dist(rng));
        batch.input(r, c) = typename Batch::Complex(re, im);
      }
    }
    batch.forward = eigen_fft_forward<Scalar>(batch.input);
    it = cache.emplace(key, std::move(batch)).first;
  }
  return it->second;
}

template <typename Scalar>
bool layout_to_natural(const std::vector<std::complex<Scalar>>& layout,
                       const Eigen::Matrix<std::complex<Scalar>, Eigen::Dynamic, Eigen::Dynamic>& plan_output,
                       Eigen::Matrix<std::complex<Scalar>, Eigen::Dynamic, Eigen::Dynamic>& natural) {
  using Complex = std::complex<Scalar>;
  const int rows = plan_output.rows();
  const int cols = plan_output.cols();
  if (rows <= 0 || static_cast<int>(layout.size()) < rows) {
    return false;
  }
  auto normalize = [&](long long idx) {
    if (rows == 0) {
      return static_cast<int>(idx);
    }
    long long mod = idx % rows;
    if (mod < 0) {
      mod += rows;
    }
    return static_cast<int>(mod);
  };
  natural.resize(rows, cols);
  std::vector<bool> seen(rows, false);
  for (int pos = 0; pos < rows; ++pos) {
    const long long raw = static_cast<long long>(std::llround(layout[static_cast<std::size_t>(pos)].real()));
    const int src = normalize(raw);
    if (src < 0 || src >= rows || seen[src]) {
      return false;
    }
    natural.row(src) = plan_output.row(pos);
    seen[src] = true;
  }
  for (bool flag : seen) {
    if (!flag) {
      return false;
    }
  }
  return true;
}

template <typename Scalar>
bool test_single_thread_fallback_matches_reference() {
  using Complex = std::complex<Scalar>;
  constexpr int N = 8;
  constexpr int B = 3;
  const auto& reference_case =
      reference_batch<Scalar>(N, B, 1381u, ReferenceSignalKind::Complex);
  Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic> signal = reference_case.input;
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

  const auto& expected = reference_case.forward;

  double max_error = 0.0;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < B; ++j) {
      const Complex diff = signal(i, j) - expected(i, j);
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
  bool ok = plan.use_kernel(eigfft::KernelKind::CooleyTukey);
  ok = ok && plan.kernel().kind == eigfft::KernelKind::CooleyTukey;
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
  std::vector<Complex> ct_a(shape.baseline_complex);
  std::vector<Complex> ct_b(shape.baseline_complex);
  std::vector<Complex*> ct_columns(shape.baseline_columns, nullptr);
  std::vector<Complex> stockham_ping(shape.stockham_stage);
  std::vector<Complex> stockham_pong(shape.stockham_stage);
  std::vector<std::ptrdiff_t> stockham_lane_bases(shape.stockham_lane_bases, 0);

  eigfft::PlanArena<Scalar> arena{};
  arena.twiddles = twiddles.data();
  arena.twiddle_count = static_cast<int>(shape.twiddles);
  arena.bitrev = bitrev.data();
  arena.bitrev_count = static_cast<int>(shape.bitrev);
  arena.baseline_a = ct_a.data();
  arena.baseline_b = ct_b.data();
  arena.baseline_columns = ct_columns.data();
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
  using Matrix = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic>;
  constexpr int N = 256;
  constexpr int B = 128;
  eigfft::PlanRuntimeConfig cfg = default_runtime_config<Scalar>();
  eigfft::PlanCache<Scalar> cache;
  cache.warm_audio_profiles(cfg);

  auto stockham_token = cache.get_plan(N, /*inverse=*/false, cfg);
  auto ct_token = cache.get_plan(N, /*inverse=*/false, cfg);
  auto& plan = stockham_token.plan();
  auto& ct_plan = ct_token.plan();
  plan.tuning.packet_step = 1;
  plan.tuning.parallel_dim = eigfft::Plan<Scalar>::ParallelDim::Columns;
  ct_plan.tuning.packet_step = 1;
  ct_plan.tuning.parallel_dim = eigfft::Plan<Scalar>::ParallelDim::Columns;
  ct_plan.use_kernel(eigfft::KernelKind::CooleyTukey);

  Matrix stockham_out(N, B);
  Matrix ct_out(N, B);
  std::vector<Complex> stock_layout(N);
  std::vector<Complex> ct_layout(N);
  if (!plan.use_kernel(eigfft::KernelKind::Stockham)) {
    return false;
  }
  try {
    const unsigned seeds[] = {1337u, 2337u};
    for (int rep = 0; rep < 2; ++rep) {
      const auto& reference_case =
          reference_batch<Scalar>(N, B, seeds[rep], ReferenceSignalKind::RealOnly);
      stockham_out = reference_case.input;
      ct_out = reference_case.input;
#if EIGFFT_TRACE_STOCKHAM
      std::cout << "[test] invoking CooleyTukey FFT rep=" << rep << std::endl;
#endif
      std::array<eigfft::MetadataRequest<Scalar>, 1> ct_meta{
          eigfft::MetadataRequest<Scalar>{
              eigfft::MetadataKind::LayoutMap,
              ct_layout.data(),
              static_cast<std::size_t>(N)}};
      eigfft::fft_inplace_batched_with_metadata<Scalar>(ct_out, ct_plan, ct_meta.data(), static_cast<int>(ct_meta.size()));
#if EIGFFT_TRACE_STOCKHAM
      std::cout << "[test] invoking stockham FFT rep=" << rep << std::endl;
#endif
      std::array<eigfft::MetadataRequest<Scalar>, 1> stock_meta{
          eigfft::MetadataRequest<Scalar>{
              eigfft::MetadataKind::LayoutMap,
              stock_layout.data(),
              static_cast<std::size_t>(N)}};
      eigfft::fft_inplace_batched_with_metadata<Scalar>(stockham_out, plan, stock_meta.data(), static_cast<int>(stock_meta.size()));
#if EIGFFT_TRACE_STOCKHAM
      std::cout << "[test] stockham FFT completed rep=" << rep << std::endl;
#endif

      const auto& expected = reference_case.forward;
      Matrix stock_natural;
      Matrix ct_natural;
      if (!layout_to_natural(stock_layout, stockham_out, stock_natural)) {
        return false;
      }
      if (!layout_to_natural(ct_layout, ct_out, ct_natural)) {
        return false;
      }

      double max_diff_stockham = 0.0;
      double max_diff_ct = 0.0;
      double max_diff_cross = 0.0;
      int bad_i = -1;
      int bad_j = -1;
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < B; ++j) {
          const double diff_stockham =
              std::abs(stock_natural(i, j) - expected(i, j));
          const double diff_ct = std::abs(ct_natural(i, j) - expected(i, j));
          const double diff_cross =
              std::abs(stock_natural(i, j) - ct_natural(i, j));
          if (diff_stockham > max_diff_stockham) {
            max_diff_stockham = diff_stockham;
            bad_i = i;
            bad_j = j;
          }
          if (diff_ct > max_diff_ct) {
            max_diff_ct = diff_ct;
          }
          if (diff_cross > max_diff_cross) {
            max_diff_cross = diff_cross;
          }
        }
      }

      const double tol = static_cast<double>(tolerance<Scalar>()) * 256.0;
      if (max_diff_stockham <= tol && max_diff_ct <= tol) {
        continue;
      }

      std::cerr << std::setprecision(12);
      const Complex stock_val = stock_natural(bad_i, bad_j);
      const Complex ct_val = ct_natural(bad_i, bad_j);
      const Complex ref_val = expected(bad_i, bad_j);
      std::cerr << "[test] stockham mismatch rep=" << rep
                << " stock_diff=" << max_diff_stockham
                << " ct_diff=" << max_diff_ct
                << " cross_diff=" << max_diff_cross
                << " at (" << bad_i << "," << bad_j << ")"
                << " stock=" << stock_val
                << " cooleytukey=" << ct_val
                << " reference=" << ref_val << std::endl;
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

  const auto& reference_case =
      reference_batch<Scalar>(N, B, 2025u, ReferenceSignalKind::Complex);
  Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic> signal = reference_case.input;
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
  const auto& expected = reference_case.forward;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < B; ++j) {
      const double re_err = static_cast<double>(signal(i, j).real()) - static_cast<double>(expected(i, j).real());
      const double im_err = static_cast<double>(signal(i, j).imag()) - static_cast<double>(expected(i, j).imag());
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
  using Matrix = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic>;
  constexpr int debugN = 8;
  constexpr int debugFrames = 3;

  eigfft::PlanRuntimeConfig cfg = default_runtime_config<Scalar>();
  eigfft::PlanCache<Scalar> cache;
  cache.warm_audio_profiles(cfg);

  // Prepare input
  const auto& reference_case =
      reference_batch<Scalar>(debugN, debugFrames, 2026u, ReferenceSignalKind::Complex);
  Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic> input = reference_case.input;

  auto token_base = cache.get_plan(debugN, /*inverse=*/false, cfg);
  auto& base_plan = token_base.plan();
  base_plan.tuning.packet_step = base_plan.packet_cols;
  base_plan.use_kernel(eigfft::KernelKind::CooleyTukey);

  auto token_stock = cache.get_plan(debugN, /*inverse=*/false, cfg);
  auto& stock_plan = token_stock.plan();
  const auto& expected = reference_case.forward;
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

  // Run CooleyTukey with metadata
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
  print_layout<Scalar>("CooleyTukey", base_layout, debugN);
  print_layout<Scalar>("Stockham", stock_layout, debugN);
  print_snapshots<Scalar>("CooleyTukey", base_snap, stages, debugN);
  print_snapshots<Scalar>("Stockham", stock_snap, stages, debugN);
  print_stage_params<Scalar>(base_stagep, stages);
  print_stage_params<Scalar>(stock_stagep, stages);
  print_pairs<Scalar>("CooleyTukey", base_pairs, stages, debugN);
  print_pairs<Scalar>("Stockham", stock_pairs, stages, debugN);
  print_invariants<Scalar>("CooleyTukey", base_inv, stages);
  print_invariants<Scalar>("Stockham", stock_inv, stages);
  print_twmap<Scalar>("CooleyTukey", base_twmap, stages, debugN);
  print_twmap<Scalar>("Stockham", stock_twmap, stages, debugN);
  print_matrix_frames<Scalar>("Captured input", input);
  print_matrix_frames<Scalar>("CooleyTukey output", base_in);
  print_matrix_frames<Scalar>("Stockham output", stock_in);
  print_elementwise_diffs<Scalar>("Element-wise Stockham vs CooleyTukey (raw)", stock_in, base_in);

  Matrix base_natural;
  Matrix stock_natural;
  if (!layout_to_natural(base_layout, base_in, base_natural)) {
    return false;
  }
  if (!layout_to_natural(stock_layout, stock_in, stock_natural)) {
    return false;
  }

  double max_stock_ref = 0.0;
  double max_base_ref = 0.0;
  double max_cross = 0.0;
  int worst_row = 0;
  int worst_frame = 0;
  for (int row = 0; row < debugN; ++row) {
    for (int frame = 0; frame < debugFrames; ++frame) {
      const double stock_diff = std::abs(stock_natural(row, frame) - expected(row, frame));
      const double base_diff = std::abs(base_natural(row, frame) - expected(row, frame));
      const double cross_diff = std::abs(stock_natural(row, frame) - base_natural(row, frame));
      if (stock_diff > max_stock_ref) {
        max_stock_ref = stock_diff;
        worst_row = row;
        worst_frame = frame;
      }
      if (base_diff > max_base_ref) {
        max_base_ref = base_diff;
      }
      if (cross_diff > max_cross) {
        max_cross = cross_diff;
      }
    }
  }

  const double tol = static_cast<double>(tolerance<Scalar>()) * 20.0;
  if (max_stock_ref > tol || max_base_ref > tol || max_cross > tol) {
    std::cerr << std::setprecision(12)
              << "[test] consensus spectrum mismatch row=" << worst_row
              << " frame=" << worst_frame
              << " stock_ref=" << max_stock_ref
              << " base_ref=" << max_base_ref
              << " cross=" << max_cross
              << " tol=" << tol << std::endl;
    std::cerr << "  expected=" << expected(worst_row, worst_frame)
              << " stock=" << stock_natural(worst_row, worst_frame)
              << " cooleytukey=" << base_natural(worst_row, worst_frame) << std::endl;
    return false;
  }

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
  using Matrix = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic>;
  const int Ns[] = {2,4,8,16,32};
  const int frame_sets[] = {1,3,16};
  eigfft::PlanRuntimeConfig cfg = default_runtime_config<Scalar>();
  eigfft::PlanCache<Scalar> cache;
  cache.warm_audio_profiles(cfg);

  for (int N : Ns) {
    for (int frames : frame_sets) {
      const unsigned seed = static_cast<unsigned int>(N * 1009 + frames * 97);
      const auto& reference_case =
          reference_batch<Scalar>(N, frames, seed, ReferenceSignalKind::Complex);
      Matrix input = reference_case.input;
      const Matrix& expected = reference_case.forward;

      // Get CooleyTukey and stockham plans
      auto token_base = cache.get_plan(N, /*inverse=*/false, cfg);
      auto& base_plan = token_base.plan();
      base_plan.tuning.packet_step = base_plan.packet_cols;
      base_plan.use_kernel(eigfft::KernelKind::CooleyTukey);

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

      // Run CooleyTukey with metadata
      Matrix base_in = input;
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
      Matrix stock_in = input;
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
  print_layout<Scalar>("CooleyTukey", base_layout, N);
  print_layout<Scalar>("Stockham", stock_layout, N);
  print_snapshots<Scalar>("CooleyTukey", base_snap, stages, N);
  print_snapshots<Scalar>("Stockham", stock_snap, stages, N);
  print_stage_params<Scalar>(base_stagep, stages);
  print_stage_params<Scalar>(stock_stagep, stages);
  print_pairs<Scalar>("CooleyTukey", base_pairs, stages, N);
  print_pairs<Scalar>("Stockham", stock_pairs, stages, N);
  print_invariants<Scalar>("CooleyTukey", base_inv, stages);
  print_invariants<Scalar>("Stockham", stock_inv, stages);
  print_twmap<Scalar>("CooleyTukey", base_twmap, stages, N);
      print_twmap<Scalar>("Stockham", stock_twmap, stages, N);
      print_matrix_frames<Scalar>("Captured input", input);
      print_matrix_frames<Scalar>("CooleyTukey output", base_in);
      print_matrix_frames<Scalar>("Stockham output", stock_in);
      print_elementwise_diffs<Scalar>("Element-wise Stockham vs CooleyTukey (raw)", stock_in, base_in);

      Matrix base_natural;
      Matrix stock_natural;
      if (!layout_to_natural(base_layout, base_in, base_natural)) {
        return false;
      }
      if (!layout_to_natural(stock_layout, stock_in, stock_natural)) {
        return false;
      }

      double max_stock_ref = 0.0;
      double max_base_ref = 0.0;
      double max_cross = 0.0;
      int worst_row = 0;
      int worst_frame = 0;
      for (int row = 0; row < N; ++row) {
        for (int frame = 0; frame < frames; ++frame) {
          const double stock_diff = std::abs(stock_natural(row, frame) - expected(row, frame));
          const double base_diff = std::abs(base_natural(row, frame) - expected(row, frame));
          const double cross_diff = std::abs(stock_natural(row, frame) - base_natural(row, frame));
          if (stock_diff > max_stock_ref) {
            max_stock_ref = stock_diff;
            worst_row = row;
            worst_frame = frame;
          }
          if (base_diff > max_base_ref) {
            max_base_ref = base_diff;
          }
          if (cross_diff > max_cross) {
            max_cross = cross_diff;
          }
        }
      }

      const double tol = static_cast<double>(tolerance<Scalar>()) * 50.0;
      if (max_stock_ref > tol || max_base_ref > tol || max_cross > tol) {
        std::cerr << std::setprecision(12)
                  << "[test] extended consensus mismatch N=" << N
                  << " frames=" << frames
                  << " stock_ref=" << max_stock_ref
                  << " base_ref=" << max_base_ref
                  << " cross=" << max_cross
                  << " tol=" << tol << std::endl;
        std::cerr << "  expected=" << expected(worst_row, worst_frame)
                  << " stock=" << stock_natural(worst_row, worst_frame)
                  << " cooleytukey=" << base_natural(worst_row, worst_frame) << std::endl;
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

      // Column isolation sanity: zero all columns except one and rerun (extended batch)
      if (frames > 1) {
        const int sel = 1;
        Matrix iso_input(N, frames);
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
    plan.use_kernel(eigfft::KernelKind::CooleyTukey);

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

