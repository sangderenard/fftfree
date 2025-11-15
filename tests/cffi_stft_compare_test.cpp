#include "../fft_cffi.hpp"
#include "../eigen_fft.hpp"
#include "../plan_support.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Complex = std::complex<float>;

constexpr float kTolerance = 1e-4f;
constexpr float kPiFloat = 3.14159265358979323846f;
constexpr double kStreamingTolerance = 5e-4;

std::vector<float> make_hann(int W) {
  constexpr double kPi = 3.14159265358979323846;
  std::vector<float> w(static_cast<std::size_t>(W));
  if (W <= 1) {
    std::fill(w.begin(), w.end(), 1.0f);
    return w;
  }
  const double denom = static_cast<double>(W - 1);
  for (int n = 0; n < W; ++n) {
    w[static_cast<std::size_t>(n)] =
        static_cast<float>(0.5 * (1.0 - std::cos(2.0 * kPi * static_cast<double>(n) / denom)));
  }
  return w;
}

std::vector<float> build_window_for_test(int kind, int W, float p1, float p2) {
  std::vector<float> w(static_cast<std::size_t>(std::max(W, 0)) , 1.0f);
  if (W <= 0) {
    return w;
  }
  switch (kind) {
    case FFT_WINDOW_RECT:
      std::fill(w.begin(), w.end(), 1.0f);
      break;
    case FFT_WINDOW_HANN: {
      const double denom = static_cast<double>(std::max(W - 1, 1));
      for (int n = 0; n < W; ++n) {
        w[static_cast<std::size_t>(n)] =
            static_cast<float>(0.5 * (1.0 - std::cos(2.0 * kPiFloat * static_cast<double>(n) / denom)));
      }
      break;
    }
    case FFT_WINDOW_HAMMING: {
      const double denom = static_cast<double>(std::max(W - 1, 1));
      for (int n = 0; n < W; ++n) {
        w[static_cast<std::size_t>(n)] =
            static_cast<float>(0.54 - 0.46 * std::cos(2.0 * kPiFloat * static_cast<double>(n) / denom));
      }
      break;
    }
    case FFT_WINDOW_BLACKMAN: {
      const double denom = static_cast<double>(std::max(W - 1, 1));
      const double a0 = 0.42;
      const double a1 = 0.5;
      const double a2 = 0.08;
      for (int n = 0; n < W; ++n) {
        const double ratio = static_cast<double>(n) / denom;
        w[static_cast<std::size_t>(n)] = static_cast<float>(
            a0 - a1 * std::cos(2.0 * kPiFloat * ratio) + a2 * std::cos(4.0 * kPiFloat * ratio));
      }
      break;
    }
    case FFT_WINDOW_TUKEY: {
      const double alpha = (p1 == 0.0f) ? 0.5 : static_cast<double>(p1);
      const double denom = static_cast<double>(std::max(W - 1, 1));
      for (int n = 0; n < W; ++n) {
        const double x = static_cast<double>(n) / denom;
        if (x < alpha / 2.0) {
          w[static_cast<std::size_t>(n)] =
              static_cast<float>(0.5 * (1.0 + std::cos(kPiFloat * (2.0 * x / alpha - 1.0))));
        } else if (x <= 1.0 - alpha / 2.0) {
          w[static_cast<std::size_t>(n)] = 1.0f;
        } else {
          w[static_cast<std::size_t>(n)] =
              static_cast<float>(0.5 * (1.0 + std::cos(kPiFloat * (2.0 * x / alpha - 2.0 / alpha + 1.0))));
        }
      }
      break;
    }
    case FFT_WINDOW_KAISER: {
      // Placeholder matches implementation in fft_cffi.cpp
      const double denom = static_cast<double>(std::max(W - 1, 1));
      for (int n = 0; n < W; ++n) {
        w[static_cast<std::size_t>(n)] =
            static_cast<float>(0.5 * (1.0 - std::cos(2.0 * kPiFloat * static_cast<double>(n) / denom)));
      }
      break;
    }
    default:
      break;
  }
  return w;
}

void normalize_window_for_test(std::vector<float>& w, int policy) {
  if (w.empty()) {
    return;
  }
  if (policy == FFT_WINDOW_NORM_L2) {
    double s2 = 0.0;
    for (float v : w) {
      s2 += static_cast<double>(v) * static_cast<double>(v);
    }
    s2 = std::sqrt(std::max(1e-30, s2));
    if (s2 > 0.0) {
      for (float& v : w) {
        v = static_cast<float>(static_cast<double>(v) / s2);
      }
    }
  } else if (policy == FFT_WINDOW_NORM_AREA) {
    double sum = 0.0;
    for (float v : w) {
      sum += static_cast<double>(v);
    }
    if (std::fabs(sum) > 1e-30) {
      for (float& v : w) {
        v = static_cast<float>(static_cast<double>(v) / sum);
      }
    }
  }
}

std::vector<Complex> run_eigen_fft(const std::vector<float>& pcm, int N) {
  Eigen::Matrix<Complex, Eigen::Dynamic, 1> column(static_cast<Eigen::Index>(N));
  for (int i = 0; i < N; ++i) {
    const float sample = (i < static_cast<int>(pcm.size())) ? pcm[static_cast<std::size_t>(i)] : 0.0f;
    column(static_cast<Eigen::Index>(i)) = Complex(sample, 0.0f);
  }

  eigfft::PlanRuntimeConfig cfg;
  cfg.threads = 1;
  cfg.lanes = 1;
  cfg.allow_inner_parallel = false;
  cfg.inner_threads = 0;

  eigfft::PlanCache<float> cache;
  auto token = cache.get_plan(N, /*inverse=*/false, cfg);
  auto& plan = token.plan();
  eigfft::fft_inplace_batched<float>(column, plan);

  std::vector<Complex> result(static_cast<std::size_t>(N));
  for (int i = 0; i < N; ++i) {
    result[static_cast<std::size_t>(i)] = column(static_cast<Eigen::Index>(i));
  }
  return result;
}

std::vector<Complex> run_eigen_stft(const std::vector<float>& pcm,
                                    int N,
                                    int W,
                                    int H,
                                    const std::vector<float>& window,
                                    int frames) {
  Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> matrix(
      static_cast<Eigen::Index>(N), static_cast<Eigen::Index>(frames));
  matrix.setZero();

  for (int f = 0; f < frames; ++f) {
    const std::size_t base = static_cast<std::size_t>(f) * static_cast<std::size_t>(H);
    for (int i = 0; i < W; ++i) {
      const std::size_t idx = base + static_cast<std::size_t>(i);
      float sample = 0.0f;
      if (idx < pcm.size()) {
        sample = pcm[idx];
      }
      float windowed = sample;
      if (!window.empty()) {
        windowed *= window[static_cast<std::size_t>(i)];
      }
      matrix(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(f)) = Complex(windowed, 0.0f);
    }
  }

  eigfft::PlanRuntimeConfig cfg;
  cfg.threads = 1;
  cfg.lanes = 1;
  cfg.allow_inner_parallel = false;
  cfg.inner_threads = 0;

  eigfft::PlanCache<float> cache;
  auto token = cache.get_plan(N, /*inverse=*/false, cfg);
  auto& plan = token.plan();
  eigfft::fft_inplace_batched<float>(matrix, plan);

  std::vector<Complex> result(static_cast<std::size_t>(frames) * static_cast<std::size_t>(N));
  for (int f = 0; f < frames; ++f) {
    for (int i = 0; i < N; ++i) {
      result[static_cast<std::size_t>(f) * static_cast<std::size_t>(N) + static_cast<std::size_t>(i)] =
          matrix(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(f));
    }
  }
  return result;
}

double max_abs_diff(const std::vector<Complex>& a, const std::vector<Complex>& b) {
  if (a.size() != b.size()) {
    return std::numeric_limits<double>::infinity();
  }
  double max_err = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double err = std::abs(a[i] - b[i]);
    if (err > max_err) {
      max_err = err;
    }
  }
  return max_err;
}

void fill_random(std::vector<float>& pcm, std::mt19937& rng) {
  std::normal_distribution<float> dist(0.0f, 1.0f);
  for (float& sample : pcm) {
    sample = dist(rng);
  }
}

std::vector<Complex> collect_cffi_fft(void* ctx, const std::vector<float>& pcm) {
  const std::size_t N = fft_ctx_size(ctx);
  std::vector<float> out_real(N);
  std::vector<float> out_imag(N);
  std::vector<float> out_mag(N);
  if (!fft_execute(ctx,
                   pcm.data(),
                   out_real.data(),
                   out_imag.data(),
                   out_mag.data(),
                   pcm.size())) {
    throw std::runtime_error("fft_execute failed");
  }
  std::vector<Complex> result(N);
  for (std::size_t i = 0; i < N; ++i) {
    result[i] = Complex(out_real[i], out_imag[i]);
  }
  return result;
}

std::vector<Complex> collect_cffi_batched(void* ctx,
                                          const std::vector<float>& pcm,
                                          int H,
                                          int& frames_out) {
  const std::size_t N = fft_ctx_size(ctx);
  const std::size_t bins = N;
  const std::size_t max_frames = (pcm.size() / static_cast<std::size_t>(std::max(1, H))) + 4;
  std::vector<float> out_real(max_frames * bins, 0.0f);
  std::vector<float> out_imag(max_frames * bins, 0.0f);
  std::vector<float> out_mag(max_frames * bins, 0.0f);

  const std::size_t produced = fft_execute_batched(ctx,
                                                   pcm.data(),
                                                   pcm.size(),
                                                   out_real.data(),
                                                   out_imag.data(),
                                                   out_mag.data(),
                                                   /*pad_mode=*/1,
                                                   /*enable_backup=*/0,
                                                   /*max_frames=*/0);
  if (produced == 0) {
    throw std::runtime_error("fft_execute_batched produced no frames");
  }
  frames_out = static_cast<int>(produced);
  std::vector<Complex> result(produced * bins);
  for (std::size_t f = 0; f < produced; ++f) {
    for (std::size_t i = 0; i < bins; ++i) {
      const std::size_t base = f * bins + i;
      result[base] = Complex(out_real[base], out_imag[base]);
    }
  }
  return result;
}

std::vector<Complex> collect_cffi_streaming(void* ctx,
                                            const std::vector<float>& pcm,
                                            int N,
                                            int W,
                                            int H,
                                            int total_frames,
                                            int chunk_frames) {
  const std::size_t bins = static_cast<std::size_t>(N);
  std::vector<Complex> aggregated(static_cast<std::size_t>(total_frames) * bins);
  std::size_t frames_done = 0;

  for (; frames_done < static_cast<std::size_t>(total_frames);) {
    const int remaining = total_frames - static_cast<int>(frames_done);
    const int frames_this = std::min(chunk_frames, remaining);
    const std::size_t start_sample = frames_done * static_cast<std::size_t>(H);
    const std::size_t needed_samples = static_cast<std::size_t>(W) + static_cast<std::size_t>(std::max(0, frames_this - 1)) * static_cast<std::size_t>(H);

    std::vector<float> chunk(needed_samples, 0.0f);
    for (std::size_t i = 0; i < needed_samples; ++i) {
      const std::size_t idx = start_sample + i;
      if (idx < pcm.size()) {
        chunk[i] = pcm[idx];
      }
    }

    const std::size_t max_chunk_frames = static_cast<std::size_t>(frames_this + 1);
    std::vector<float> out_real(max_chunk_frames * bins, 0.0f);
    std::vector<float> out_imag(max_chunk_frames * bins, 0.0f);
    std::vector<float> out_mag(max_chunk_frames * bins, 0.0f);

    const std::size_t produced = fft_execute_batched(ctx,
                                                     chunk.data(),
                                                     chunk.size(),
                                                     out_real.data(),
                                                     out_imag.data(),
                                                     out_mag.data(),
                                                     /*pad_mode=*/1,
                                                     /*enable_backup=*/0,
                                                     /*max_frames=*/0);
    if (produced != static_cast<std::size_t>(frames_this)) {
      throw std::runtime_error("Streaming STFT produced unexpected frame count");
    }

    for (std::size_t f = 0; f < produced; ++f) {
      for (std::size_t i = 0; i < bins; ++i) {
        const std::size_t src = f * bins + i;
        const std::size_t dst = (frames_done + f) * bins + i;
        aggregated[dst] = Complex(out_real[src], out_imag[src]);
      }
    }

    frames_done += produced;
  }
  return aggregated;
}

bool test_plain_fft_vs_eigen() {
  const int N = 1024;
  std::vector<float> pcm(static_cast<std::size_t>(N));
  std::mt19937 rng(1337);
  fill_random(pcm, rng);

  void* ctx = fft_init_full_v2(
      N,
      /*threads=*/1,
      /*lanes=*/1,
      /*inverse=*/0,
      /*kernel=*/0,
      /*radix=*/0,
      /*radix_pattern=*/nullptr,
      /*radix_pattern_len=*/0,
      /*pad_mode=*/1,
      /*window=*/0,
      /*hop=*/0,
      FFT_TRANSFORM_R2C,
      /*reduce_magnitude=*/0,
      /*store_polar=*/0,
      /*half_spectrum=*/0,
      /*allow_outer_parallel=*/0,
      /*allow_inner_parallel=*/0,
      /*inner_threads=*/0,
      /*save_crash_logs=*/0,
      /*silent_crash_reports=*/1,
      /*apply_windows=*/0,
      /*apply_ola=*/0,
      FFT_WINDOW_RECT,
      /*analysis_param1=*/0.0f,
      /*analysis_param2=*/0.0f,
      FFT_WINDOW_RECT,
      /*synthesis_param1=*/0.0f,
      /*synthesis_param2=*/0.0f,
      FFT_WINDOW_NORM_NONE,
      FFT_COLA_OFF);

  if (!ctx) {
    std::cerr << "plain FFT: failed to initialize context\n";
    return false;
  }

  bool ok = true;
  try {
    auto cffi = collect_cffi_fft(ctx, pcm);
    auto eigen = run_eigen_fft(pcm, N);
    const double err = max_abs_diff(cffi, eigen);
    std::cout << "No-window FFT max abs error=" << err << '\n';
    if (err > kTolerance) {
      std::cerr << "plain FFT mismatch (max abs error=" << err << ")\n";
      ok = false;
    }
  } catch (const std::exception& ex) {
    std::cerr << "plain FFT exception: " << ex.what() << '\n';
    ok = false;
  }

  fft_free(ctx);
  return ok;
}

bool test_batched_stft_vs_eigen(int& out_frames, std::vector<float>& pcm_out, std::vector<Complex>& eigen_out) {
  const int N = 512;
  const int W = 512;
  const int H = 128;
  const int planned_frames = 8;
  const std::size_t pcm_len = static_cast<std::size_t>(W) + static_cast<std::size_t>(planned_frames - 1) * static_cast<std::size_t>(H);

  pcm_out.assign(pcm_len, 0.0f);
  std::mt19937 rng(42);
  fill_random(pcm_out, rng);

  void* ctx = fft_init_full_v2(
      N,
      /*threads=*/1,
      /*lanes=*/1,
      /*inverse=*/0,
      /*kernel=*/0,
      /*radix=*/0,
      /*radix_pattern=*/nullptr,
      /*radix_pattern_len=*/0,
      /*pad_mode=*/1,
      /*window=*/W,
      /*hop=*/H,
      FFT_TRANSFORM_R2C,
      /*reduce_magnitude=*/0,
      /*store_polar=*/0,
      /*half_spectrum=*/0,
      /*allow_outer_parallel=*/0,
      /*allow_inner_parallel=*/0,
      /*inner_threads=*/0,
      /*save_crash_logs=*/0,
      /*silent_crash_reports=*/1,
      /*apply_windows=*/1,
      /*apply_ola=*/0,
      FFT_WINDOW_HANN,
      /*analysis_param1=*/0.0f,
      /*analysis_param2=*/0.0f,
      FFT_WINDOW_HANN,
      /*synthesis_param1=*/0.0f,
      /*synthesis_param2=*/0.0f,
      FFT_WINDOW_NORM_NONE,
      FFT_COLA_OFF);

  if (!ctx) {
    std::cerr << "batched STFT: failed to initialize context\n";
    return false;
  }

  bool ok = true;
  try {
    int frames = 0;
    auto cffi = collect_cffi_batched(ctx, pcm_out, H, frames);
    out_frames = frames;

    auto window = make_hann(W);
    eigen_out = run_eigen_stft(pcm_out, N, W, H, window, frames);
    const double err = max_abs_diff(cffi, eigen_out);
    std::cout << "Batched STFT max abs error=" << err << " (frames=" << frames << ")\n";
    if (err > kTolerance) {
      std::cerr << "batched STFT mismatch (max abs error=" << err << ")\n";
      ok = false;
    }
  } catch (const std::exception& ex) {
    std::cerr << "batched STFT exception: " << ex.what() << '\n';
    ok = false;
  }

  fft_free(ctx);
  return ok;
}

bool test_streaming_stft_vs_eigen(const std::vector<float>& pcm,
                                  int N,
                                  int W,
                                  int H,
                                  int total_frames,
                                  const std::vector<Complex>& eigen_reference) {
  void* ctx = fft_init_full_v2(
      N,
      /*threads=*/1,
      /*lanes=*/1,
      /*inverse=*/0,
      /*kernel=*/0,
      /*radix=*/0,
      /*radix_pattern=*/nullptr,
      /*radix_pattern_len=*/0,
      /*pad_mode=*/1,
      /*window=*/W,
      /*hop=*/H,
      FFT_TRANSFORM_R2C,
      /*reduce_magnitude=*/0,
      /*store_polar=*/0,
      /*half_spectrum=*/0,
      /*allow_outer_parallel=*/0,
      /*allow_inner_parallel=*/0,
      /*inner_threads=*/0,
      /*save_crash_logs=*/0,
      /*silent_crash_reports=*/1,
      /*apply_windows=*/1,
      /*apply_ola=*/0,
      FFT_WINDOW_HANN,
      /*analysis_param1=*/0.0f,
      /*analysis_param2=*/0.0f,
      FFT_WINDOW_HANN,
      /*synthesis_param1=*/0.0f,
      /*synthesis_param2=*/0.0f,
      FFT_WINDOW_NORM_NONE,
      FFT_COLA_OFF);

  if (!ctx) {
    std::cerr << "streaming STFT: failed to initialize context\n";
    return false;
  }

  bool ok = true;
  try {
    const int chunk_frames = 3;
    auto streaming = collect_cffi_streaming(ctx, pcm, N, W, H, total_frames, chunk_frames);
    const double err = max_abs_diff(streaming, eigen_reference);
    std::cout << "Streaming STFT max abs error=" << err << '\n';
    if (err > kTolerance) {
      std::cerr << "streaming STFT mismatch (max abs error=" << err << ")\n";
      ok = false;
    }
  } catch (const std::exception& ex) {
    std::cerr << "streaming STFT exception: " << ex.what() << '\n';
    ok = false;
  }

  fft_free(ctx);
  return ok;
}

struct StreamingRoundtripCase {
  const char* name;
  int N;
  int W;
  int H;
  int window_kind;
  int norm_policy;
};

bool run_streaming_inverse_roundtrip_case(const StreamingRoundtripCase& cfg) {
  if (cfg.N <= 0 || cfg.W <= 0 || cfg.H <= 0) {
    std::cerr << "streaming inverse: invalid dimensions in test case\n";
    return false;
  }
  if (cfg.W > cfg.N) {
    std::cerr << "streaming inverse: window larger than FFT size\n";
    return false;
  }

  const std::size_t total_samples = static_cast<std::size_t>(cfg.W) +
                                    3 * static_cast<std::size_t>(cfg.H);
  std::vector<float> pcm(total_samples, 0.0f);
  for (std::size_t i = 0; i < pcm.size(); ++i) {
    const float phase = static_cast<float>(i) * 2.0f * kPiFloat /
                        static_cast<float>(cfg.W + cfg.H + 17);
    const float phase2 = static_cast<float>(i) * 2.0f * kPiFloat /
                         static_cast<float>(cfg.H + 11);
    pcm[i] = std::sin(phase) + 0.35f * std::cos(phase2) + 0.1f * std::sin(phase * 0.5f);
  }
  const std::size_t taper = std::min<std::size_t>(pcm.size() / 4, static_cast<std::size_t>(cfg.W));
  if (taper > 0) {
    for (std::size_t i = 0; i < taper; ++i) {
      const float ramp = static_cast<float>(i) / static_cast<float>(taper);
      pcm[i] *= ramp;
      pcm[pcm.size() - 1 - i] *= ramp;
    }
  }

  void* forward = fft_init_full_v2(
      static_cast<std::size_t>(cfg.N),
      0,
      1,
      0,
      FFT_KERNEL_COOLEYTUKEY,
      0,
      nullptr,
      0,
      2,
      cfg.W,
      cfg.H,
      FFT_TRANSFORM_C2C,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      cfg.window_kind,
      0.0f,
      0.0f,
      cfg.window_kind,
      0.0f,
      0.0f,
      cfg.norm_policy,
      FFT_COLA_NORMALIZE);
  if (!forward) {
    std::cerr << "streaming inverse: failed to initialize forward context" << std::endl;
    return false;
  }

  void* inverse = fft_init_full_v2(
      static_cast<std::size_t>(cfg.N),
      0,
      1,
      1,
      FFT_KERNEL_COOLEYTUKEY,
      0,
      nullptr,
      0,
      2,
      cfg.W,
      cfg.H,
      FFT_TRANSFORM_C2C,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      cfg.window_kind,
      0.0f,
      0.0f,
      cfg.window_kind,
      0.0f,
      0.0f,
      cfg.norm_policy,
      FFT_COLA_NORMALIZE);
  if (!inverse) {
    std::cerr << "streaming inverse: failed to initialize inverse context" << std::endl;
    fft_free(forward);
    return false;
  }

  std::size_t frames_expected = 0;
  if (pcm.size() == 0) {
    frames_expected = 0;
  } else if (pcm.size() <= static_cast<std::size_t>(cfg.W)) {
    frames_expected = 1;
  } else {
    frames_expected = 1 + (pcm.size() - static_cast<std::size_t>(cfg.W)) /
                              static_cast<std::size_t>(cfg.H);
  }

  std::vector<float> spec_real;
  std::vector<float> spec_imag;
  spec_real.reserve(frames_expected * static_cast<std::size_t>(cfg.N));
  spec_imag.reserve(frames_expected * static_cast<std::size_t>(cfg.N));

  auto capture_frames = [&](std::size_t start, std::size_t count, int flush_mode) {
    const std::size_t max_frames = frames_expected + 2;
    std::vector<float> out_real(max_frames * static_cast<std::size_t>(cfg.N), 0.0f);
    std::vector<float> out_imag(max_frames * static_cast<std::size_t>(cfg.N), 0.0f);
    const float* chunk_ptr = (count > 0) ? pcm.data() + start : nullptr;
    const size_t produced = fft_stream_push_pcm(
        forward,
        chunk_ptr,
        count,
        out_real.data(),
        out_imag.data(),
        nullptr,
        max_frames,
        flush_mode);
    for (size_t f = 0; f < produced; ++f) {
      const float* r = out_real.data() + f * static_cast<std::size_t>(cfg.N);
      const float* im = out_imag.data() + f * static_cast<std::size_t>(cfg.N);
      spec_real.insert(spec_real.end(), r, r + cfg.N);
      spec_imag.insert(spec_imag.end(), im, im + cfg.N);
    }
    return produced;
  };

  const std::size_t chunk1 = std::min<std::size_t>(
      std::max<std::size_t>(1, static_cast<std::size_t>(cfg.H) - 1), pcm.size());
  const std::size_t chunk2 = std::min<std::size_t>(
      std::max<std::size_t>(1, static_cast<std::size_t>(cfg.H) + static_cast<std::size_t>(cfg.W) / 4),
      pcm.size() - chunk1);
  const std::size_t chunk3 = pcm.size() - chunk1 - chunk2;

  std::size_t frames_total = 0;
  frames_total += capture_frames(0, chunk1, FFT_STREAM_FLUSH_NONE);
  frames_total += capture_frames(chunk1, chunk2, FFT_STREAM_FLUSH_NONE);
  frames_total += capture_frames(chunk1 + chunk2, chunk3, FFT_STREAM_FLUSH_FINAL);

  const std::size_t bins = static_cast<std::size_t>(cfg.N);
  if (frames_total != frames_expected ||
      spec_real.size() != frames_total * bins ||
      spec_imag.size() != frames_total * bins) {
    std::cerr << "streaming inverse: unexpected frame count" << std::endl;
    fft_free(forward);
    fft_free(inverse);
    return false;
  }

  std::vector<float> analysis_win = build_window_for_test(cfg.window_kind, cfg.W, 0.0f, 0.0f);
  std::vector<float> synthesis_win = build_window_for_test(cfg.window_kind, cfg.W, 0.0f, 0.0f);
  normalize_window_for_test(analysis_win, cfg.norm_policy);
  normalize_window_for_test(synthesis_win, cfg.norm_policy);

  std::vector<float> reconstructed;
  std::vector<float> chunk_out(bins, 0.0f);
  size_t frame_idx = 0;
  while (frame_idx < frames_total) {
    const size_t remaining = frames_total - frame_idx;
    const size_t send = (frame_idx == 0) ? std::min<std::size_t>(static_cast<std::size_t>(2), remaining)
                                         : std::min<std::size_t>(static_cast<std::size_t>(1), remaining);
    const int flush = (frame_idx + send == frames_total) ? FFT_STREAM_FLUSH_FINAL : FFT_STREAM_FLUSH_NONE;
    const size_t produced = fft_stream_push_spectrum(
        inverse,
        spec_real.data() + frame_idx * bins,
        spec_imag.data() + frame_idx * bins,
        send,
        chunk_out.data(),
        bins,
        flush);
    if (produced > 0) {
      reconstructed.insert(reconstructed.end(), chunk_out.begin(), chunk_out.begin() + produced);
    }
    frame_idx += send;
  }

  if (fft_stream_pending_pcm(inverse) != 0) {
    std::vector<float> tail(bins, 0.0f);
    const size_t drained = fft_stream_push_spectrum(
        inverse,
        nullptr,
        nullptr,
        0,
        tail.data(),
        bins,
        FFT_STREAM_FLUSH_NONE);
    if (drained > 0) {
      reconstructed.insert(reconstructed.end(), tail.begin(), tail.begin() + drained);
    }
  }

  bool ok = true;
  if (reconstructed.size() != pcm.size()) {
    std::cerr << "streaming inverse: length mismatch (expected " << pcm.size()
              << ", got " << reconstructed.size() << ")" << std::endl;
    ok = false;
  } else {
    std::vector<double> cola_norm(reconstructed.size(), 0.0);
    for (std::size_t f = 0; f < frames_total; ++f) {
      const std::size_t start = f * static_cast<std::size_t>(cfg.H);
      for (int wi = 0; wi < cfg.W; ++wi) {
        const std::size_t idx = start + static_cast<std::size_t>(wi);
        if (idx >= cola_norm.size()) {
          break;
        }
        const float aw = (wi < static_cast<int>(analysis_win.size()))
                             ? analysis_win[static_cast<std::size_t>(wi)]
                             : 1.0f;
        const float sw = (wi < static_cast<int>(synthesis_win.size()))
                             ? synthesis_win[static_cast<std::size_t>(wi)]
                             : 1.0f;
        cola_norm[idx] += static_cast<double>(aw) * static_cast<double>(sw);
      }
    }

    double max_err = 0.0;
    std::size_t compared = 0;
    for (std::size_t i = 0; i < pcm.size(); ++i) {
      const double denom = (i < cola_norm.size()) ? cola_norm[i] : 0.0;
      if (denom < 1e-3) {
        continue;
      }
      ++compared;
      max_err = std::max(max_err, static_cast<double>(std::fabs(pcm[i] - reconstructed[i])));
    }
    if (compared == 0) {
      for (std::size_t i = 0; i < pcm.size(); ++i) {
        max_err = std::max(max_err, static_cast<double>(std::fabs(pcm[i] - reconstructed[i])));
      }
    }
    std::cout << cfg.name << " max abs error=" << max_err << '\n';
    if (max_err > kStreamingTolerance) {
      std::cerr << "streaming inverse: reconstruction error exceeds tolerance" << std::endl;
      ok = false;
    }
  }

  fft_free(forward);
  fft_free(inverse);
  return ok;
}

bool test_streaming_inverse_roundtrip() {
  const std::vector<StreamingRoundtripCase> cases = {
      {"Rectangular window (no taper)", 64, 64, 16, FFT_WINDOW_RECT, FFT_WINDOW_NORM_NONE},
      {"Hann window 50% overlap", 128, 128, 64, FFT_WINDOW_HANN, FFT_WINDOW_NORM_NONE},
      {"Hamming window quarter-hop (L2)", 128, 128, 32, FFT_WINDOW_HAMMING, FFT_WINDOW_NORM_L2},
      {"Blackman window heavy overlap (area)", 128, 128, 96, FFT_WINDOW_BLACKMAN, FFT_WINDOW_NORM_AREA},
  };

  bool ok = true;
  for (const auto& cfg : cases) {
    if (!run_streaming_inverse_roundtrip_case(cfg)) {
      ok = false;
    }
  }
  return ok;
}

}  // namespace

int main() {
  bool ok = true;

  if (!test_plain_fft_vs_eigen()) {
    ok = false;
  }

  int frames = 0;
  std::vector<float> pcm;
  std::vector<Complex> eigen_ref;
  if (!test_batched_stft_vs_eigen(frames, pcm, eigen_ref)) {
    ok = false;
  }

  if (!eigen_ref.empty() && frames > 0) {
    if (!test_streaming_stft_vs_eigen(pcm, /*N=*/512, /*W=*/512, /*H=*/128, frames, eigen_ref)) {
      ok = false;
    }
  } else {
    std::cerr << "Skipping streaming STFT test due to missing reference" << '\n';
    ok = false;
  }

  if (!test_streaming_inverse_roundtrip()) {
    ok = false;
  }

  if (ok) {
    std::cout << "All CFFI vs Eigen comparisons passed." << std::endl;
  }
  return ok ? 0 : 1;
}
