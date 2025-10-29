#include "eigen_fft.hpp"

#include <Eigen/Core>

#include <chrono>
#include <cctype>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <algorithm>
#include <thread>
#include <vector>

namespace {

struct BenchmarkCase {
  int N;
  int B;
  int repeats;
};

static const std::vector<BenchmarkCase> kCases{
    {256, 128, 12},
    {1024, 64, 8},
    {2048, 96, 6},
    {4096, 32, 4},
};

inline std::string to_lower(std::string value) {
  for (char& ch : value) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  return value;
}

struct PrecisionConfig {
  std::string label;
  std::string cli_token;
};

template <typename Scalar>
struct PrecisionTraits;

template <>
struct PrecisionTraits<double> {
  static std::string label() { return "float64"; }
  static std::string cli_token() { return "f64"; }
};

template <>
struct PrecisionTraits<float> {
  static std::string label() { return "float32"; }
  static std::string cli_token() { return "f32"; }
};

template <>
struct PrecisionTraits<Eigen::half> {
  static std::string label() { return "float16"; }
  static std::string cli_token() { return "f16"; }
};

struct RealtimeOptions {
  bool enabled = false;
  double duration_seconds = 10.0;
  double sample_rate_hz = 48000.0;
  int window = 1024;
  int stride = 512;
  double delay_seconds = 0.0;
  double safety_margin = 0.85;
};

template <typename Scalar>
void run_realtime_simulation(std::mt19937_64 seed_rng, const RealtimeOptions& opts) {
  if (!opts.enabled) return;

  using Complex = std::complex<Scalar>;
  using MatrixXc = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic>;

  const double sample_rate = std::min(std::max(opts.sample_rate_hz, 1.0), 1'000'000.0);
  const int window = std::max(opts.window, 1);
  const int stride = std::max(opts.stride, 1);
  const double safety = std::clamp(opts.safety_margin, 1e-3, 0.999);
  const double duration = std::max(opts.duration_seconds, window / sample_rate);

  const std::size_t total_samples =
      static_cast<std::size_t>(std::ceil(sample_rate * duration));
  if (total_samples < static_cast<std::size_t>(window)) {
    std::cout << "Real-time probe skipped: total samples (" << total_samples
              << ") smaller than window (" << window << ")." << std::endl;
    return;
  }

  const std::size_t frames =
      1 + (total_samples - static_cast<std::size_t>(window)) / static_cast<std::size_t>(stride);
  if (frames == 0) {
    std::cout << "Real-time probe skipped: stride too large for the configured duration."
              << std::endl;
    return;
  }

  std::cout << "\n--- Real-time probe (" << PrecisionTraits<Scalar>::label()
            << ") ---" << std::endl;
  std::cout << "  sample_rate=" << sample_rate << " Hz, duration=" << duration
            << " s, frames=" << frames << std::endl;
  std::cout << "  window=" << window << ", stride=" << stride
            << ", safety=" << safety << ", delay=" << opts.delay_seconds << " s" << std::endl;

  std::vector<double> pcm(total_samples);
  std::normal_distribution<double> dist(0.0, 1.0);
  auto rng = seed_rng;
  for (std::size_t i = 0; i < total_samples; ++i) {
    pcm[i] = dist(rng);
  }

  MatrixXc frame(window, 1);
  eigfft::Plan<Scalar> plan(window, /*inverse=*/false, /*threads=*/true);
  plan.tuning.packet_step = 0;
  plan.tuning.parallel_dim = eigfft::Plan<Scalar>::ParallelDim::Columns;
  plan.tuning.min_work_per_thread = 32;
  plan.tuning.force_ftz_daz = false;

  const double frame_period = stride / sample_rate;
  double cumulative_time = 0.0;
  double worst_frame_time = 0.0;
  double worst_compute_time = 0.0;
  double total_frame_time = 0.0;
  double total_compute_time = 0.0;
  std::size_t overruns = 0;

  const auto delay_duration = std::chrono::duration<double>(opts.delay_seconds);

  for (std::size_t f = 0; f < frames; ++f) {
    const std::size_t start_idx = f * static_cast<std::size_t>(stride);
    for (int i = 0; i < window; ++i) {
      const Scalar real_sample = static_cast<Scalar>(pcm[start_idx + static_cast<std::size_t>(i)]);
      frame(i, 0) = Complex(real_sample, Scalar(0));
    }

    const auto compute_start = std::chrono::high_resolution_clock::now();
    eigfft::fft_inplace_batched<Scalar>(frame, plan);
    const auto compute_end = std::chrono::high_resolution_clock::now();
    const double compute_time =
        std::chrono::duration<double>(compute_end - compute_start).count();
    worst_compute_time = std::max(worst_compute_time, compute_time);
    total_compute_time += compute_time;

    if (opts.delay_seconds > 0.0) {
      std::this_thread::sleep_for(delay_duration);
    }
    const double frame_time = compute_time + opts.delay_seconds;
    worst_frame_time = std::max(worst_frame_time, frame_time);
    total_frame_time += frame_time;

    cumulative_time += frame_time;
    const double deadline = (static_cast<double>(f) + 1.0) * frame_period;
    if (cumulative_time > deadline) {
      ++overruns;
    }
  }

  const double frames_d = static_cast<double>(frames);
  const double avg_frame_time = total_frame_time / frames_d;
  const double avg_compute_time = total_compute_time / frames_d;

  const double required_frame_period = worst_frame_time / safety;
  const double sustainable_frame_rate = 1.0 / required_frame_period;
  const double sustainable_sample_rate = sustainable_frame_rate * stride;

  const double allowable_time = frame_period * safety;
  const bool meets_config = worst_frame_time <= allowable_time;
  const double slack = allowable_time - worst_frame_time;

  std::cout << "  avg frame time      : " << avg_frame_time << " s" << std::endl;
  std::cout << "  avg compute time    : " << avg_compute_time << " s" << std::endl;
  std::cout << "  worst frame time    : " << worst_frame_time << " s" << std::endl;
  std::cout << "  worst compute time  : " << worst_compute_time << " s" << std::endl;
  std::cout << "  deadline overruns   : " << overruns << std::endl;
  std::cout << "  provided frame period (stride/sample_rate): " << frame_period
            << " s" << std::endl;
  std::cout << "  allowable compute budget (safety * frame_period): " << allowable_time
            << " s" << std::endl;
  std::cout << "  meets configured rate? " << (meets_config ? "yes" : "no")
            << " (slack=" << slack << " s)" << std::endl;
  std::cout << "  max sustainable frame rate (with safety): " << sustainable_frame_rate
            << " Hz" << std::endl;
  std::cout << "  max sustainable sample rate (with safety): " << sustainable_sample_rate
            << " samples/s" << std::endl;
}

template <typename Scalar>
void run_suite_for_precision(std::mt19937_64 seed_rng, const RealtimeOptions& rt_opts) {
  using Complex = std::complex<Scalar>;
  using MatrixXc = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic>;

  std::cout << "\n=== Precision: " << PrecisionTraits<Scalar>::label() << " ===" << std::endl;

  std::normal_distribution<double> dist(0.0, 1.0);
  auto make_rng = [&](uint64_t seed) {
    std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
    return rng;
  };

  auto fill_random = [&](MatrixXc& mat, std::mt19937& rng) {
    for (int i = 0; i < mat.rows(); ++i) {
      for (int j = 0; j < mat.cols(); ++j) {
        const Scalar re = static_cast<Scalar>(dist(rng));
        const Scalar im = static_cast<Scalar>(dist(rng));
        mat(i, j) = Complex(re, im);
      }
    }
  };

  auto run_plan = [&](const MatrixXc& seed, eigfft::Plan<Scalar>& plan, int repeats) {
    double accum = 0.0;
    MatrixXc work(seed.rows(), seed.cols());
    for (int r = 0; r < repeats; ++r) {
      work = seed;
      const auto start = std::chrono::high_resolution_clock::now();
      eigfft::fft_inplace_batched<Scalar>(work, plan);
      const auto end = std::chrono::high_resolution_clock::now();
      accum += std::chrono::duration<double>(end - start).count();
    }
    return accum / static_cast<double>(repeats);
  };

  std::cout << std::fixed << std::setprecision(6);

  for (size_t idx = 0; idx < kCases.size(); ++idx) {
    const auto& task = kCases[idx];
    std::mt19937 rng = make_rng(seed_rng() + static_cast<uint64_t>(idx));

    std::cout << "\n--- Benchmark N=" << task.N << " B=" << task.B
              << " repeats=" << task.repeats << " ---" << std::endl;

    MatrixXc seed(task.N, task.B);
    fill_random(seed, rng);

    eigfft::Plan<Scalar> baseline_plan(task.N, /*inverse=*/false, /*threads=*/true);
    baseline_plan.tuning.packet_step = 1;
    baseline_plan.tuning.parallel_dim = eigfft::Plan<Scalar>::ParallelDim::Columns;
    baseline_plan.tuning.min_work_per_thread = 32;
    baseline_plan.tuning.force_ftz_daz = false;

    eigfft::Plan<Scalar> simd_plan(task.N, /*inverse=*/false, /*threads=*/true);
    simd_plan.tuning.packet_step = 0;
    simd_plan.tuning.parallel_dim = eigfft::Plan<Scalar>::ParallelDim::Columns;
    simd_plan.tuning.min_work_per_thread = 32;
    simd_plan.tuning.force_ftz_daz = false;

    const double baseline_time = run_plan(seed, baseline_plan, task.repeats);
    const double simd_time = run_plan(seed, simd_plan, task.repeats);

    std::cout << "  baseline (lanes=1)         : " << baseline_time << " s" << std::endl;
    std::cout << "  simd+threads (lanes=" << simd_plan.packet_cols
              << ", max threads=" << simd_plan.effective_threads(task.B)
              << ") : " << simd_time << " s" << std::endl;
    if (simd_time < baseline_time) {
      std::cout << "    speedup: " << (baseline_time / simd_time) << "x" << std::endl;
    } else {
      std::cout << "    slowdown: " << (simd_time / baseline_time) << "x" << std::endl;
    }
  }

  eigfft::Plan<Scalar> probe_plan(256, /*inverse=*/false, /*threads=*/true);
  std::cout << "\nHand-rolled Cooley-Tukey kernel uses Eigen packets (packet_cols="
            << probe_plan.packet_cols
            << ") and OpenMP for batched columns." << std::endl;

  run_realtime_simulation<Scalar>(seed_rng, rt_opts);
}

}  // namespace

int main(int argc, char** argv) {
  std::cout << "fftfree micro-benchmark" << std::endl;

  try {
    Eigen::setNbThreads(1);

    std::unordered_set<std::string> requested;
    RealtimeOptions realtime_opts;
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      const std::string prefix = "--precision=";
      if (arg.rfind(prefix, 0) == 0) {
        arg.erase(0, prefix.size());
        std::stringstream ss(arg);
        std::string token;
        while (std::getline(ss, token, ',')) {
          requested.insert(to_lower(token));
        }
      } else if (arg == "--realtime" || arg == "--rt") {
        realtime_opts.enabled = true;
      } else if (arg.rfind("--rt-sample-rate=", 0) == 0) {
        realtime_opts.enabled = true;
        realtime_opts.sample_rate_hz = std::stod(arg.substr(17));
      } else if (arg.rfind("--rt-window=", 0) == 0) {
        realtime_opts.enabled = true;
        realtime_opts.window = std::stoi(arg.substr(12));
      } else if (arg.rfind("--rt-stride=", 0) == 0) {
        realtime_opts.enabled = true;
        realtime_opts.stride = std::stoi(arg.substr(12));
      } else if (arg.rfind("--rt-delay-ms=", 0) == 0) {
        realtime_opts.enabled = true;
        realtime_opts.delay_seconds = std::stod(arg.substr(14)) / 1000.0;
      } else if (arg.rfind("--rt-safety=", 0) == 0) {
        realtime_opts.enabled = true;
        realtime_opts.safety_margin = std::stod(arg.substr(12));
      } else if (arg.rfind("--rt-duration=", 0) == 0) {
        realtime_opts.enabled = true;
        realtime_opts.duration_seconds = std::stod(arg.substr(14));
      } else if (arg == "--help" || arg == "-h") {
        std::cout << "Usage: fft_example [--precision=f64,f32,f16]\n"
                     "  f64 : std::complex<double>\n"
                     "  f32 : std::complex<float>\n"
                     "  f16 : std::complex<Eigen::half>\n"
                     "Default is to run all three precisions.\n"
                     "\nReal-time probe options (auto-enable realtime mode):\n"
                     "  --realtime | --rt                 Enable real-time simulation\n"
                     "  --rt-sample-rate=<Hz>             Input sample rate (default 48000, max 1e6)\n"
                     "  --rt-window=<samples>             FFT window size (default 1024)\n"
                     "  --rt-stride=<samples>             Hop size between frames (default 512)\n"
                     "  --rt-duration=<seconds>           PCM duration (default 10)\n"
                     "  --rt-delay-ms=<milliseconds>      Extra delay per frame (default 0)\n"
                     "  --rt-safety=<0-1)                 Fraction of frame period usable for compute (default 0.85)\n";
        return 0;
      } else {
        std::cerr << "Unrecognized argument: " << arg << std::endl;
        return 1;
      }
    }

    if (requested.empty()) {
      requested = {"f64", "f32", "f16"};
    }

    const std::vector<std::string> known = {"f64", "f32", "f16"};
    for (const auto& token : requested) {
      if (std::find(known.begin(), known.end(), token) == known.end()) {
        std::cerr << "Unknown precision token '" << token
                  << "'. Supported tokens: f64,f32,f16." << std::endl;
        return 1;
      }
    }

    std::mt19937_64 seed_rng(1337);

    if (requested.count("f64")) {
      run_suite_for_precision<double>(seed_rng, realtime_opts);
    }
    if (requested.count("f32")) {
      run_suite_for_precision<float>(seed_rng, realtime_opts);
    }
    if (requested.count("f16")) {
      run_suite_for_precision<Eigen::half>(seed_rng, realtime_opts);
    }

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return 1;
  }
}
