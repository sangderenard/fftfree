// fft_cffi.hpp
// C-compatible FFT function for cffi
#pragma once
#include <cstddef>
#include <complex>
// Expose core enums from the library headers rather than inventing new
// duplicated values here. Callers should include this header to get the
// authoritative kernel/radix definitions.
#include "eigen_fft.hpp"

#if defined(_WIN32) || defined(__CYGWIN__)
#  ifdef FFT_CFFI_EXPORTS
#    define FFT_CFFI_API __declspec(dllexport)
#  else
#    define FFT_CFFI_API __declspec(dllimport)
#  endif
#else
#  define FFT_CFFI_API
#endif

// Preferred alias moving forward: FFTFREE_API
#ifndef FFTFREE_API
#define FFTFREE_API FFT_CFFI_API
#endif

extern "C" {
    // in_pcm: pointer to float PCM data (mono)
    // out_real, out_imag, out_mag: pointers to float arrays for output
    // n: number of samples
    // threads: desired number of threads (>=1). If <=0, implementation clamps to a safe default.
    FFT_CFFI_API void fft_pcm_to_channels(const float* in_pcm, float* out_real, float* out_imag, float* out_mag, size_t n, int threads);

    // Simple deployment-oriented API
    // Create a reusable FFT context with prebuilt plans.
    //   n: FFT size (must be power of two)
    //   threads: number of thread-local plan instances to precreate (>=1)
    //   lanes: lane capacity hint (<=0 => auto)
    //   inverse: non-zero for inverse transform
    // Returns an opaque handle or NULL on failure.
    // kernel: 0=auto, 1=cooleytukey, 2=stockham
    // radix: 0=unspecified (caller omitted). If radix==0 and kernel is specified
    //         the Plan defaults for that algorithm are used for any unspecified
    //         parameters. If radix==0 and kernel==0 (auto) the radix defaults
    //         to 2. Otherwise provide the desired radix (e.g., 2 or 4).
    // pad_mode: 0=auto (pad if required by kernel), 1=always pad up to next power-of-two, 2=never pad (error if incompatible)
    //
    // Phase-2 additions:
    // - exported integer constants for kernels/pad/radix are provided below
    // - radix_pattern: optional pointer to an array of ints describing a mixed-radix
    //   pattern (e.g., {2,2,4}). The pattern is copied by the callee; caller may
    //   pass nullptr with len=0 to indicate no pattern.

    // Do NOT invent or duplicate core enums here. Use the values defined in
    // `eigfft::KernelKind` and `eigfft::ButterflyRadix` from the library
    // headers. For convenience bind C++ constants here so C++ callers can use
    // them without repeating the enum definitions.
}

// Provide C-compatible integer constants that are derived from the authoritative
// enums in the library headers. These are computed at compile-time so they
// always reflect the upstream enum ordering/values.
enum {
    FFT_KERNEL_COOLEYTUKEY = static_cast<int>(eigfft::KernelKind::CooleyTukey),
    FFT_KERNEL_STOCKHAM = static_cast<int>(eigfft::KernelKind::Stockham),
    FFT_KERNEL_EXTERNAL = static_cast<int>(eigfft::KernelKind::External)
};

enum {
    FFT_BUTTERFLY_RADIX_2 = static_cast<int>(eigfft::ButterflyRadix::Radix2),
    FFT_BUTTERFLY_RADIX_4 = static_cast<int>(eigfft::ButterflyRadix::Radix4),
    FFT_BUTTERFLY_RADIX_8 = static_cast<int>(eigfft::ButterflyRadix::Radix8),
    FFT_BUTTERFLY_RADIX_16 = static_cast<int>(eigfft::ButterflyRadix::Radix16)
};

// Transform mode constants for C callers
enum {
    FFT_TRANSFORM_C2C = 0,
    FFT_TRANSFORM_R2C = 1,
    FFT_TRANSFORM_C2R = 2,
    FFT_TRANSFORM_R2R = 3
};

extern "C" {

    // New Phase-2 ABI: accept radix_pattern pointer and length. Caller may pass
    // nullptr/0 to indicate no pattern. The pattern values are plain ints (e.g.
    // 2,4). radix == 0 still means unspecified and probing behavior remains.
    // Backwards-compatible init: existing behavior preserved. Use fft_init_ex
    // to pass STFT/window parameters.
    FFT_CFFI_API void* fft_init(size_t n, int threads, int lanes, int inverse, int kernel, int radix, const int* radix_pattern, size_t radix_pattern_len, int pad_mode);

    // Extended initializer (STFT-aware). Parameters:
    //  - window: analysis window size (W). If 0, defaults to plan N (no extra subwindowing).
    //  - hop: hop/stride in samples between windows. If 0, defaults to window.
    //  - stft_mode: 0=disabled (legacy), 1=batched STFT helper enabled, 2=streaming mode (reserved)
    FFT_CFFI_API void* fft_init_ex(size_t n, int threads, int lanes, int inverse, int kernel, int radix, const int* radix_pattern, size_t radix_pattern_len, int pad_mode, int window, int hop, int stft_mode);

    // Full initializer exposing all runtime configuration knobs used by the Plan.
    // transform: 0=C2C, 1=R2C, 2=C2R, 3=R2R
    // reduce_magnitude: non-zero to write |X| (or (|X|,phase) if store_polar)
    // store_polar: non-zero to pack (mag, phase) into complex slots
    // half_spectrum: non-zero to use 0..N/2 packing for real FFTs (R2C/C2R)
    FFT_CFFI_API void* fft_init_full(size_t n,
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
                                     int inner_threads);

    // Execute FFT using a pre-initialized context. Returns 1 on success, 0 on error.
    FFT_CFFI_API int fft_execute(void* handle,
                                 const float* in_pcm,
                                 float* out_real,
                                 float* out_imag,
                                 float* out_mag,
                                 size_t n);
    // Batched STFT helper: scatter overlapping windows from `pcm` (length pcm_len)
    // into independent N-sized columns, run a batched FFT across all frames and
    // write outputs into out_real/out_imag/out_mag arrays. Caller must provide
    // buffers with capacity at least max_frames * ctx_N. The function returns
    // the number of frames produced (0 on error). Use pad_mode to control final
    // partial-frame padding (0=auto/refuse per init rules, 1=pad last, 2=never).
    // Layout: output is flattened frame-major: frame0_bin0..binN-1, frame1_bin0..binN-1, ...
    FFT_CFFI_API size_t fft_execute_batched(void* handle,
                                           const float* pcm,
                                           size_t pcm_len,
                                           float* out_real,
                                           float* out_imag,
                                           float* out_mag,
                                           int pad_mode,
                                           size_t max_frames);
    // Query effective FFT size (plan size) for a context
    FFT_CFFI_API size_t fft_ctx_size(void* handle);
    // Query the number of worker threads available to the context (outer parallelism).
    FFT_CFFI_API size_t fft_ctx_worker_threads(void* handle);
    // Query the effective plan threads (after runtime/hardware limits).
    FFT_CFFI_API size_t fft_ctx_effective_threads(void* handle);

    // Destroy a context created by fft_init (safe to pass NULL).
    FFT_CFFI_API void fft_free(void* handle);
}
