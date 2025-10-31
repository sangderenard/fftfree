// fft_cffi.cpp

#include "fft_cffi.hpp"
#include "eigen_fft.hpp"
#include "plan_support.hpp"
#include <vector>
#include <complex>
#if defined(_WIN32)
#include <windows.h>
#endif

namespace {
struct FftContext {
    eigfft::PlanCache<float> cache;
    eigfft::PlanRuntimeConfig cfg{};
    int N = 0;
    bool inverse = false;
    int kernel = 0;   // 0=auto,1=CT,2=Stockham
    int pad_mode = 0; // 0=auto,1=always,2=never
};
}

extern "C" {
void fft_pcm_to_channels(const float* in_pcm, float* out_real, float* out_imag, float* out_mag, size_t n, int threads) {
    using namespace eigfft;
    using Complex = std::complex<float>;
    if (!in_pcm || !out_real || !out_imag || !out_mag || n == 0) {
        return;
    }
    try {
        // Copy PCM to complex input (imag=0)
        std::vector<Complex> input(n);
        for (size_t i = 0; i < n; ++i) {
            input[i] = Complex(in_pcm[i], 0.0f);
        }
        // Set up FFT plan
        PlanRuntimeConfig cfg;
        cfg.threads = (threads > 0) ? threads : 1;
        cfg.lanes = 1;
        PlanEnvironment<float> env;
        env.initialize(static_cast<int>(n), false, cfg);
        auto& plan = env.plan();
        // Run FFT in-place
        Eigen::Map<Eigen::Matrix<Complex, Eigen::Dynamic, 1>> data(input.data(), static_cast<Eigen::Index>(n));
        fft_inplace_batched<float>(data, plan);
        // Output real, imag, mag as binary
        for (size_t i = 0; i < n; ++i) {
            out_real[i] = data(static_cast<Eigen::Index>(i)).real();
            out_imag[i] = data(static_cast<Eigen::Index>(i)).imag();
            out_mag[i] = std::abs(data(static_cast<Eigen::Index>(i)));
        }
    } catch (const std::exception& ex) {
        // Do not let exceptions escape the C boundary; zero outputs as a safe fallback.
        for (size_t i = 0; i < n; ++i) {
            out_real[i] = 0.0f;
            out_imag[i] = 0.0f;
            out_mag[i] = 0.0f;
        }
#if defined(_WIN32)
        // Best-effort signal to debugger / stderr for diagnostics.
        OutputDebugStringA("fft_pcm_to_channels: exception caught; outputs zeroed.\n");
#endif
        (void)ex; // suppress unused warning when no logging available
    } catch (...) {
        for (size_t i = 0; i < n; ++i) {
            out_real[i] = 0.0f;
            out_imag[i] = 0.0f;
            out_mag[i] = 0.0f;
        }
#if defined(_WIN32)
        OutputDebugStringA("fft_pcm_to_channels: unknown exception; outputs zeroed.\n");
#endif
    }
}

static inline bool is_power_of_two(size_t v) {
    if (v == 0) return false;
    return (v & (v - 1)) == 0;
}

static inline size_t next_power_of_two(size_t v) {
    if (v <= 1) return 1;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
#if INTPTR_MAX == INT64_MAX
    v |= v >> 32;
#endif
    v++;
    return v;
}

void* fft_init(size_t n, int threads, int lanes, int inverse, int kernel, int pad_mode) {
    try {
        if (n == 0) return nullptr;
        auto* ctx = new FftContext();
        ctx->N = static_cast<int>(n);
        ctx->inverse = (inverse != 0);
        ctx->kernel = kernel;
        ctx->pad_mode = pad_mode;
        ctx->cfg.threads = (threads > 0) ? threads : eigfft::Plan<float>::Limits::kDefaultRuntimeThreads;
        ctx->cfg.lanes = (lanes > 0) ? lanes : 0; // 0 => auto inside plan
        ctx->cfg.inverse = ctx->inverse;
        // Resolve kernel requirement: currently both built-ins require power-of-two.
        const bool requires_pow2 = true; // both CT and Stockham are radix-2 here

        size_t plan_n = static_cast<size_t>(ctx->N);
        if (requires_pow2) {
            const bool is_pow2 = is_power_of_two(plan_n);
            if (!is_pow2) {
                if (pad_mode == 2 /*never*/) {
                    delete ctx;
                    return nullptr;
                }
                // auto/always pad up
                plan_n = next_power_of_two(plan_n);
            } else if (pad_mode == 1 /*always*/) {
                // already a power of two: keep as is
            }
        } else if (pad_mode == 1 /*always*/) {
            // Kernel does not require power-of-two, but ALWAYS pad requested.
            plan_n = next_power_of_two(plan_n);
        }
        ctx->N = static_cast<int>(plan_n);
        // Touch the cache to precreate at least `threads` plan instances.
        {
            auto tok = ctx->cache.get_plan(ctx->N, ctx->inverse, ctx->cfg);
            (void)tok; // creating the token ensures pool is sized; token releases on scope end
        }
        return static_cast<void*>(ctx);
    } catch (...) {
        return nullptr;
    }
}

int fft_execute(void* handle,
                const float* in_pcm,
                float* out_real,
                float* out_imag,
                float* out_mag,
                size_t n) {
    if (!handle || !in_pcm || !out_real || !out_imag || !out_mag || n == 0) return 0;
    FftContext* ctx = static_cast<FftContext*>(handle);
    if (ctx->pad_mode == 2 /*never*/ && static_cast<int>(n) != ctx->N) return 0;
    if (n > static_cast<size_t>(ctx->N)) return 0; // cannot exceed plan size
    try {
        using Complex = std::complex<float>;
        const size_t plan_n = static_cast<size_t>(ctx->N);
        std::vector<Complex> input(plan_n);
        // Copy provided samples; zero-pad remainder when allowed
        size_t i = 0;
        for (; i < n; ++i) input[i] = Complex(in_pcm[i], 0.0f);
        for (; i < plan_n; ++i) input[i] = Complex(0.0f, 0.0f);
        // Acquire a plan instance from cache
        auto token = ctx->cache.get_plan(ctx->N, ctx->inverse, ctx->cfg);
        auto& plan = token.plan();
        // Select kernel if requested
        if (ctx->kernel == 1) {
            plan.use_kernel(eigfft::KernelKind::CooleyTukey);
        } else if (ctx->kernel == 2) {
            plan.use_kernel(eigfft::KernelKind::Stockham);
        }
        Eigen::Map<Eigen::Matrix<Complex, Eigen::Dynamic, 1>> data(input.data(), static_cast<Eigen::Index>(plan_n));
        eigfft::fft_inplace_batched<float>(data, plan);
        for (size_t j = 0; j < plan_n; ++j) {
            const auto v = data(static_cast<Eigen::Index>(j));
            out_real[j] = v.real();
            out_imag[j] = v.imag();
            out_mag[j] = std::abs(v);
        }
        return 1;
    } catch (...) {
        return 0;
    }
}

size_t fft_ctx_size(void* handle) {
    if (!handle) return 0;
    FftContext* ctx = static_cast<FftContext*>(handle);
    return static_cast<size_t>(ctx->N);
}

void fft_free(void* handle) {
    if (!handle) return;
    FftContext* ctx = static_cast<FftContext*>(handle);
    delete ctx;
}
}
