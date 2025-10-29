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
