// phase_infer.hpp
#pragma once

#include <vector>
#include <cmath>

namespace phaseinfer {

struct PhaseConfig {
  int N = 0;            // plan size (power of two preferred)
  int hop = 0;          // hop size between frames
  bool half_spectrum = true; // input magnitude uses N/2+1 bins if true, else N
  int mode = 0;         // 0=linear (deterministic), reserved for future modes
  int iterations = 0;   // reserved for future iterative solvers (e.g., Griffin–Lim)
};

// Simple, non-iterative phase inference that propagates a linear phase advance
// per bin across frames: phi[f,k] = phi0[k] + f * (2*pi*k*hop/N). The initial
// phase phi0 is zero. This preserves inter-frame coherence for stationary bins
// and avoids checkerboard artifacts seen with per-frame zero phase.
//
// Inputs:
//  - mag: frame-major magnitudes of shape (frames, bins_per_frame)
//  - frames: number of frames (columns)
//  - cfg: plan/inference configuration
// Outputs:
//  - out_real,out_imag: frame-major complex arrays sized frames * bins_per_frame
//
// Bins per frame is N when cfg.half_spectrum==false, or N/2+1 otherwise.
inline void infer_linear(const float* mag,
                         size_t frames,
                         const PhaseConfig& cfg,
                         float* out_real,
                         float* out_imag) {
  const int N = cfg.N;
  const int hop = (cfg.hop > 0) ? cfg.hop : N;
  const int bins = cfg.half_spectrum ? (N/2 + 1) : N;
  if (N <= 0 || bins <= 0 || frames == 0) return;
  const double two_pi = 6.28318530717958647692;
  // Per-bin phase increment
  std::vector<double> dphi(static_cast<size_t>(bins));
  for (int k = 0; k < bins; ++k) {
    // Clamp k for half-spectrum edge cases
    const int kk = (k < N) ? k : (k % N);
    dphi[static_cast<size_t>(k)] = two_pi * static_cast<double>(kk) * static_cast<double>(hop) / static_cast<double>(N);
  }
  // Accumulated phase per bin
  std::vector<double> phi(static_cast<size_t>(bins), 0.0);
  for (size_t f = 0; f < frames; ++f) {
    const size_t base = f * static_cast<size_t>(bins);
    for (int k = 0; k < bins; ++k) {
      const double m = static_cast<double>(mag[base + static_cast<size_t>(k)]);
      const double c = std::cos(phi[static_cast<size_t>(k)]);
      const double s = std::sin(phi[static_cast<size_t>(k)]);
      out_real[base + static_cast<size_t>(k)] = static_cast<float>(m * c);
      out_imag[base + static_cast<size_t>(k)] = static_cast<float>(m * s);
      phi[static_cast<size_t>(k)] += dphi[static_cast<size_t>(k)];
    }
  }
}

} // namespace phaseinfer

