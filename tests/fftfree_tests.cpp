#include "../eigen_fft.hpp"

#include <Eigen/Core>

#include <cmath>
#include <complex>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

namespace {

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

  eigfft::Plan<Scalar> forward_plan(N, /*inverse=*/false, /*threads=*/true);
  eigfft::Plan<Scalar> inverse_plan(N, /*inverse=*/true, /*threads=*/true);

  eigfft::fft_inplace_batched<Scalar>(signal, forward_plan);
  eigfft::fft_inplace_batched<Scalar>(signal, inverse_plan);

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
  eigfft::Plan<Scalar> plan(32, /*inverse=*/false, /*threads=*/true);
  return plan.effective_threads(64) >= 2;
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

  eigfft::Plan<Scalar> plan(N, /*inverse=*/false, /*threads=*/true);
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
  eigfft::Plan<Scalar> plan(32, /*inverse=*/false, /*threads=*/true);
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

#ifndef EIGFFT_ALLOW_SEQUENTIAL
template <typename Scalar>
bool test_sequential_path_rejected() {
  try {
    eigfft::Plan<Scalar> plan(8, /*inverse=*/false, /*threads=*/false);
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
#ifndef EIGFFT_ALLOW_SEQUENTIAL
  results.emplace_back("sequential_path_rejected<" + tag + ">",
                       test_sequential_path_rejected<Scalar>());
#endif
}

}  // namespace

int main() {
  std::vector<std::pair<std::string, bool>> results;
  results.reserve(12);

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
