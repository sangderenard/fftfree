// fft_cffi.cpp

#include "fft_cffi.hpp"
#include "eigen_fft.hpp"
#include "plan_support.hpp"
#include <vector>
#include <complex>

extern "C" {
void fft_pcm_to_channels(const float* in_pcm, float* out_real, float* out_imag, float* out_mag, size_t n) {
    using namespace eigfft;
    using Complex = std::complex<float>;
    // Copy PCM to complex input (imag=0)
    std::vector<Complex> input(n);
    for (size_t i = 0; i < n; ++i) {
        input[i] = Complex(in_pcm[i], 0.0f);
    }
    // Set up FFT plan
    PlanRuntimeConfig cfg;
    cfg.threads = 1;
    cfg.lanes = 1;
    PlanEnvironment<float> env;
    env.initialize(static_cast<int>(n), false, cfg);
    auto& plan = env.plan();
    // Run FFT in-place
    Eigen::Map<Eigen::Matrix<Complex, Eigen::Dynamic, 1>> data(input.data(), n);
    fft_inplace_batched<float>(data, plan);
    // Output real, imag, mag as binary
    for (size_t i = 0; i < n; ++i) {
        out_real[i] = data(i).real();
        out_imag[i] = data(i).imag();
        out_mag[i] = std::abs(data(i));
    }
}
}
