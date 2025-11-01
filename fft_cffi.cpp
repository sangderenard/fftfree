// fft_cffi.cpp

#include "fft_cffi.hpp"
#include "eigen_fft.hpp"
#include "plan_support.hpp"
#include "crash_handler.hpp"
#include <vector>
#include <complex>
#include <algorithm>
#include <functional>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#if defined(_WIN32)
#include <windows.h>
#endif

#if defined(_WIN32)
// windows.h may define macros min/max; avoid interfering with std::min/std::max
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif
#endif

namespace {
struct FftContext {
    eigfft::PlanCache<float> cache;
    eigfft::PlanRuntimeConfig cfg{};
    int N = 0;
    bool inverse = false;
    int kernel = 0;   // 0=auto,1=CT,2=Stockham
    int radix = 0;    // 0=unspecified, otherwise 2/4/8/16
    int pad_mode = 0; // 0=auto,1=always,2=never
    // Copied mixed-radix pattern from outer API (values like 2,4,...)
    std::vector<int> radix_pattern;
    // STFT/windowing settings
    int window = 0;   // analysis window (W). 0 => use plan N
    int hop = 0;      // hop/stride between windows. 0 => window
    int stft_mode = 0; // 0=disabled,1=batched helper,2=streaming (reserved)
    // Transform/output shape
    int transform = 0;        // 0=C2C,1=R2C,2=C2R,3=R2R
    int reduce_magnitude = 0; // bool-like
    int store_polar = 0;      // bool-like
    int half_spectrum = 0;    // bool-like
    int allow_outer_parallel = 1;
    int allow_inner_parallel = 0;
    int inner_threads = 0;
    // Persistent worker pool (outer parallelism only). Use the canonical
    // WorkerPool implementation from `eigen_fft.hpp` to avoid duplication.
    std::unique_ptr<eigfft::WorkerPool> pool;
};

static eigfft::PlanRuntimeConfig compute_effective_runtime(const FftContext& ctx) {
    eigfft::PlanRuntimeConfig cfg = ctx.cfg;
    int outer_threads = (ctx.cfg.threads > 0) ? ctx.cfg.threads : 1;
    if ((ctx.allow_outer_parallel != 0) && ctx.pool) {
        const int pool_threads = ctx.pool->size();
        if (pool_threads > 0) {
            outer_threads = pool_threads;
        }
    }
    const bool allow_inner = (ctx.allow_inner_parallel != 0) && outer_threads <= 1;
    int inner_budget = 0;
    if (allow_inner) {
        if (ctx.inner_threads > 0) {
            inner_budget = ctx.inner_threads;
        } else {
            int hw = static_cast<int>(std::thread::hardware_concurrency());
            inner_budget = (hw > 0) ? hw : 1;
        }
    }
    int plan_threads = outer_threads;
    if (allow_inner) {
        plan_threads = std::max(plan_threads, inner_budget);
    }
    plan_threads = std::max(plan_threads, 1);
    cfg.threads = plan_threads;
    cfg.allow_inner_parallel = allow_inner;
    cfg.inner_threads = allow_inner ? ctx.inner_threads : 0;
    return cfg;
}

// Dispatcher that forwards jobs to the context's outer WorkerPool, or runs inline if unavailable.
struct PoolDispatcher : public eigfft::JobDispatcher {
    eigfft::WorkerPool* pool = nullptr;
    void parallel_for(size_t total, size_t chunk, const Fn& fn) override {
        if (!pool || total == 0) {
            eigfft::InlineDispatcher::instance().parallel_for(total, chunk, fn);
            return;
        }
        pool->parallel_for(total, (chunk==0)?1:chunk, fn);
    }
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
        // Determine plan size: auto-pad to next power-of-two so this convenience
        // function behaves consistently with the fft_init/fft_execute path.
        size_t plan_n = n;
        // Simple helper in this translation unit lives below; declare inline logic here
        auto is_pow2 = [](size_t v) {
            return v != 0 && (v & (v - 1)) == 0;
        };
        auto next_pow2 = [](size_t v) {
            if (v <= 1) return static_cast<size_t>(1);
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
        };

        if (!is_pow2(plan_n)) plan_n = next_pow2(plan_n);

        std::vector<Complex> input(plan_n);
        // copy provided samples and zero-pad remainder
        size_t i = 0;
        for (; i < n; ++i) input[i] = Complex(in_pcm[i], 0.0f);
        for (; i < plan_n; ++i) input[i] = Complex(0.0f, 0.0f);

        // Set up FFT plan
        PlanRuntimeConfig cfg;
        cfg.threads = (threads > 0) ? threads : 1;
        cfg.lanes = 1;
        PlanEnvironment<float> env;
        env.initialize(static_cast<int>(plan_n), false, cfg);
        auto& plan = env.plan();
        // Run FFT in-place
        Eigen::Map<Eigen::Matrix<Complex, Eigen::Dynamic, 1>> data(input.data(), static_cast<Eigen::Index>(plan_n));
        fft_inplace_batched<float>(data, plan);
        // Output real, imag, mag as binary
        // Only write out the first `n` samples (match previous behavior)
        for (size_t j = 0; j < n; ++j) {
            const auto v = data(static_cast<Eigen::Index>(j));
            out_real[j] = v.real();
            out_imag[j] = v.imag();
            out_mag[j] = std::abs(v);
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

void* fft_init(size_t n, int threads, int lanes, int inverse, int kernel, int radix, const int* radix_pattern, size_t radix_pattern_len, int pad_mode) {
    // legacy init forwards to extended initializer with default window/hop/mode
    return fft_init_ex(n, threads, lanes, inverse, kernel, radix, radix_pattern, radix_pattern_len, pad_mode, 0, 0, 0);
}

void* fft_init_ex(size_t n, int threads, int lanes, int inverse, int kernel, int radix, const int* radix_pattern, size_t radix_pattern_len, int pad_mode, int window, int hop, int stft_mode) {
    try {
        // Ensure we install the crash handler for processes that call the C API
        fftfree::install_crash_handler();
        if (n == 0) return nullptr;
        auto* ctx = new FftContext();
        ctx->N = static_cast<int>(n);
        ctx->inverse = (inverse != 0);
        ctx->kernel = kernel;
        ctx->radix = radix;
        ctx->pad_mode = pad_mode;
        ctx->window = window;
        ctx->hop = hop;
        ctx->stft_mode = stft_mode;
        // copy supplied radix pattern (if any)
        if (radix_pattern && radix_pattern_len > 0) {
            ctx->radix_pattern.assign(radix_pattern, radix_pattern + radix_pattern_len);
        }
        ctx->cfg.threads = (threads > 0) ? threads : eigfft::Plan<float>::Limits::kDefaultRuntimeThreads;
        ctx->cfg.lanes = (lanes > 0) ? lanes : 0; // 0 => auto inside plan
        ctx->cfg.radix = ctx->radix;
        // Propagate any supplied pattern into runtime config so PlanEnvironment
        // can copy it into the created Plan before allocation.
        if (!ctx->radix_pattern.empty()) ctx->cfg.radix_pattern = ctx->radix_pattern;
        ctx->cfg.inverse = ctx->inverse;
    // Decide effective radix and whether the plan requires a power-of-two size.
    // Priority: explicit caller radix -> algorithm default radix (if kernel specified) -> fallback to 2.
    int effective_radix = ctx->radix;
    if (effective_radix == 0 && ctx->kernel != 0) {
        try {
            eigfft::PlanRuntimeConfig probe_cfg = ctx->cfg;
            probe_cfg.threads = 1;
            probe_cfg.lanes = 1;
            probe_cfg.radix = 0;
            eigfft::PlanEnvironment<float> probe_env;
            probe_env.initialize(2, ctx->inverse, probe_cfg);
            auto& probe_plan = probe_env.plan();
            using BR = eigfft::ButterflyRadix;
            BR br = BR::Radix2;
            if (ctx->kernel == 1) br = probe_plan.butterfly_default_cooleytukey.radix;
            else if (ctx->kernel == 2) br = probe_plan.butterfly_default_stockham.radix;
            switch (br) {
              case BR::Radix2: effective_radix = 2; break;
              case BR::Radix4: effective_radix = 4; break;
              case BR::Radix8: effective_radix = 8; break;
              case BR::Radix16: effective_radix = 16; break;
              default: effective_radix = 2; break;
            }
        } catch (...) {
            effective_radix = 2;
        }
    }

    if (effective_radix == 0) effective_radix = 2;

    ctx->cfg.radix = effective_radix;
    if (!ctx->radix_pattern.empty()) ctx->cfg.radix_pattern = ctx->radix_pattern;

    const bool requires_pow2 = (effective_radix >= 2);

    size_t plan_n = static_cast<size_t>(ctx->N);
    if (requires_pow2) {
        const bool is_pow2 = is_power_of_two(plan_n);
        if (!is_pow2) {
            if (pad_mode == 1 /*always*/) {
                plan_n = next_power_of_two(plan_n);
            } else {
                size_t attempted = next_power_of_two(plan_n);
                fprintf(stderr, "fft_init: refusing to auto-pad N=%d -> %zu because pad_mode!=ALWAYS; set pad_mode=1 to permit padding or supply power-of-two N\n", ctx->N, attempted);
#if defined(_WIN32)
                OutputDebugStringA("fft_init: refused to auto-pad; pad_mode not ALWAYS\n");
#endif
                delete ctx;
                return nullptr;
            }
        }
    } else if (pad_mode == 1 /*always*/) {
        plan_n = next_power_of_two(plan_n);
    }
    ctx->N = static_cast<int>(plan_n);
    {
        try {
            auto tok = ctx->cache.get_plan(ctx->N, ctx->inverse, ctx->cfg);
            (void)tok;
        } catch (const std::exception& ex) {
            fprintf(stderr, "fft_init: exception while creating plan for N=%d: %s\n", ctx->N, ex.what());
#if defined(_WIN32)
            OutputDebugStringA("fft_init: exception while creating plan (see stderr)\n");
#endif
            delete ctx;
            return nullptr;
        }
    }
    return static_cast<void*>(ctx);
    } catch (...) {
        fprintf(stderr, "fft_init: unknown exception caught, returning NULL\n");
#if defined(_WIN32)
        OutputDebugStringA("fft_init: unknown exception caught\n");
#endif
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

size_t fft_execute_batched(void* handle,
                           const float* pcm,
                           size_t pcm_len,
                           float* out_real,
                           float* out_imag,
                           float* out_mag,
                           int pad_mode,
                           size_t max_frames) {
    if (!handle || !pcm || pcm_len == 0 || !out_real || !out_imag || !out_mag) return 0;
    FftContext* ctx = static_cast<FftContext*>(handle);
    const int N = ctx->N;
    if (N <= 0) return 0;
    // Determine analysis window and hop
    int W = ctx->window ? ctx->window : N;
    int H = ctx->hop ? ctx->hop : W;
    if (W <= 0 || H <= 0) return 0;
    if (W > N) {
        // Cannot analyze windows larger than plan size.
        return 0;
    }

    // Determine effective pad policy: function param 0 => defer to ctx
    int effective_pad = (pad_mode == 0) ? ctx->pad_mode : pad_mode;

    // Compute number of frames
    size_t frames = 0;
    if (pcm_len < static_cast<size_t>(W)) {
        if (effective_pad == 1) frames = 1; else return 0;
    } else {
        // at least one full frame
        frames = 1 + (pcm_len - static_cast<size_t>(W)) / static_cast<size_t>(H);
        // check for a trailing partial frame
        size_t last_start = (frames - 1) * static_cast<size_t>(H);
        size_t remaining = (pcm_len > last_start) ? (pcm_len - last_start) : 0;
        if (remaining < static_cast<size_t>(W)) {
            if (effective_pad == 1) {
                // keep partial and pad
            } else if (effective_pad == 2) {
                // drop partial
                if (frames > 0) frames -= 1;
            } else {
                // pad_mode==0 treated as refuse
                return 0;
            }
        }
    }

    if (frames == 0) return 0;
    if (max_frames != 0 && frames > max_frames) frames = max_frames;

    const size_t plan_n = static_cast<size_t>(N);

    // Chunking: avoid huge temporary allocations. Aim for <= 64MB buffer.
    const size_t max_bytes = 64ULL * 1024ULL * 1024ULL;
    const size_t bytes_per_frame = plan_n * sizeof(std::complex<float>);
    size_t batch_frames = std::max((size_t)1, static_cast<size_t>(max_bytes / (bytes_per_frame + 1)));
    if (batch_frames > frames) batch_frames = frames;

    size_t produced = 0;
    try {
        for (size_t bstart = 0; bstart < frames; bstart += batch_frames) {
            size_t bcount = std::min(batch_frames, frames - bstart);
            std::vector<std::complex<float>> buffer(plan_n * bcount);
            // Fill columns (column-major: contiguous columns of length N)
            for (size_t f = 0; f < bcount; ++f) {
                size_t frame_idx = bstart + f;
                size_t start = frame_idx * static_cast<size_t>(H);
                size_t copy_count = 0;
                if (start < pcm_len) copy_count = std::min(static_cast<size_t>(W), pcm_len - start);
                // Copy samples
                size_t col_base = f * plan_n;
                for (size_t i = 0; i < copy_count; ++i) {
                    buffer[col_base + i] = std::complex<float>(pcm[start + i], 0.0f);
                }
                // Zero-pad remainder up to W
                for (size_t i = copy_count; i < static_cast<size_t>(W); ++i) {
                    buffer[col_base + i] = std::complex<float>(0.0f, 0.0f);
                }
                // Zero-pad up to plan_n
                for (size_t i = static_cast<size_t>(W); i < plan_n; ++i) {
                    buffer[col_base + i] = std::complex<float>(0.0f, 0.0f);
                }
            }

            // Build a plan and set its dispatcher to the outer pool (if allowed)
            eigfft::PlanRuntimeConfig local_cfg = compute_effective_runtime(*ctx);
            auto token = ctx->cache.get_plan(ctx->N, ctx->inverse, local_cfg);
            auto& plan = token.plan();
            if (ctx->kernel == 1) plan.use_kernel(eigfft::KernelKind::CooleyTukey);
            else if (ctx->kernel == 2) plan.use_kernel(eigfft::KernelKind::Stockham);

            PoolDispatcher dispatcher;
            dispatcher.pool = (ctx->allow_outer_parallel && ctx->pool) ? ctx->pool.get() : nullptr;
            plan.set_dispatcher(&dispatcher);

            Eigen::Map<Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> sub(buffer.data(), static_cast<Eigen::Index>(plan_n), static_cast<Eigen::Index>(bcount));
            eigfft::fft_inplace_batched<float>(sub, plan);

            // Copy outputs (full chunk)
            for (size_t f = 0; f < bcount; ++f) {
                size_t frame_idx = bstart + f;
                size_t out_base = frame_idx * plan_n;
                size_t col_base = f * plan_n;
                for (size_t i = 0; i < plan_n; ++i) {
                    const auto& v = buffer[col_base + i];
                    out_real[out_base + i] = v.real();
                    out_imag[out_base + i] = v.imag();
                    if (ctx->store_polar) {
                        out_mag[out_base + i] = v.real();
                    } else {
                        out_mag[out_base + i] = std::abs(v);
                    }
                }
            }
            plan.set_dispatcher(nullptr);
            produced += bcount;
        }
        return produced;
    } catch (...) {
        return produced;
    }
}

size_t fft_ctx_size(void* handle) {
    if (!handle) return 0;
    FftContext* ctx = static_cast<FftContext*>(handle);
    return static_cast<size_t>(ctx->N);
}

size_t fft_ctx_worker_threads(void* handle) {
    if (!handle) return 0;
    FftContext* ctx = static_cast<FftContext*>(handle);
    if (ctx->pool) {
        const int workers = ctx->pool->size();
        return static_cast<size_t>(workers > 0 ? workers : 1);
    }
    const int requested = ctx->cfg.threads;
    return static_cast<size_t>(requested > 0 ? requested : 1);
}

size_t fft_ctx_effective_threads(void* handle) {
    if (!handle) return 0;
    auto* ctx = static_cast<FftContext*>(handle);
    try {
        eigfft::PlanRuntimeConfig runtime = ctx->cfg;
        runtime.inverse = ctx->inverse;
        auto token = ctx->cache.get_plan(ctx->N, ctx->inverse, runtime);
        const int effective = token.plan().effective_threads(ctx->N);
        return static_cast<size_t>(effective > 0 ? effective : 1);
    } catch (...) {
        return 0;
    }
}

void fft_free(void* handle) {
    if (!handle) return;
    FftContext* ctx = static_cast<FftContext*>(handle);
    delete ctx;
}

void* fft_init_full(size_t n,
                    int threads,
                    int lanes,
                    int inverse,
                    int kernel,
                    int radix,
                    const int* radix_pattern,
                    size_t radix_pattern_len,
                    int pad_mode,
                    int window,
                    int hop,
                    int stft_mode,
                    int transform,
                    int reduce_magnitude,
                    int store_polar,
                    int half_spectrum,
                    int allow_outer_parallel,
                    int allow_inner_parallel,
                    int inner_threads) {
    try {
        // Ensure crash handler is active for CFFI users (Python demo, etc.)
        fftfree::install_crash_handler();
        auto* ctx = new FftContext();
        ctx->N = static_cast<int>(n);
        ctx->inverse = (inverse != 0);
        ctx->kernel = kernel;
        ctx->radix = radix;
        ctx->pad_mode = pad_mode;
        ctx->window = window;
        ctx->hop = hop;
        ctx->stft_mode = stft_mode;
        ctx->transform = transform;
        ctx->reduce_magnitude = reduce_magnitude;
        ctx->store_polar = store_polar;
        ctx->half_spectrum = half_spectrum;
        ctx->allow_outer_parallel = allow_outer_parallel;
        ctx->allow_inner_parallel = allow_inner_parallel;
        ctx->inner_threads = inner_threads;
        if (radix_pattern && radix_pattern_len > 0) {
            ctx->radix_pattern.assign(radix_pattern, radix_pattern + radix_pattern_len);
        }

        ctx->cfg.threads = (threads > 0) ? threads : 1; // outer pool threads
        ctx->cfg.lanes = (lanes > 0) ? lanes : 1;
        ctx->cfg.inverse = ctx->inverse;
        ctx->cfg.radix = ctx->radix;
        ctx->cfg.radix_pattern = ctx->radix_pattern;
        ctx->cfg.transform = ctx->transform;
        ctx->cfg.reduce_magnitude = (ctx->reduce_magnitude != 0);
        ctx->cfg.store_polar = (ctx->store_polar != 0);
        ctx->cfg.half_spectrum = (ctx->half_spectrum != 0);
        ctx->cfg.allow_outer_parallel = (ctx->allow_outer_parallel != 0);
        ctx->cfg.allow_inner_parallel = (ctx->allow_inner_parallel != 0);
        ctx->cfg.inner_threads = ctx->inner_threads;

        // Warm the cache with a plan instance
        {
            eigfft::PlanRuntimeConfig warm_cfg = compute_effective_runtime(*ctx);
            auto token = ctx->cache.get_plan(ctx->N, ctx->inverse, warm_cfg);
            auto& plan = token.plan();
            if (ctx->kernel == 1) plan.use_kernel(eigfft::KernelKind::CooleyTukey);
            else if (ctx->kernel == 2) plan.use_kernel(eigfft::KernelKind::Stockham);
        }
        // Create worker pool for outer parallelism if requested
        if ((ctx->allow_outer_parallel != 0) && ctx->cfg.threads > 1) {
            ctx->pool = std::make_unique<eigfft::WorkerPool>(ctx->cfg.threads);
        }
        return ctx;
    } catch (...) {
        return nullptr;
    }
}
}
