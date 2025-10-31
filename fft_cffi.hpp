// fft_cffi.hpp
// C-compatible FFT function for cffi
#pragma once
#include <cstddef>
#include <complex>

#if defined(_WIN32) || defined(__CYGWIN__)
#  ifdef FFT_CFFI_EXPORTS
#    define FFT_CFFI_API __declspec(dllexport)
#  else
#    define FFT_CFFI_API __declspec(dllimport)
#  endif
#else
#  define FFT_CFFI_API
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
    // pad_mode: 0=auto (pad if required by kernel), 1=always pad up to next power-of-two, 2=never pad (error if incompatible)
    FFT_CFFI_API void* fft_init(size_t n, int threads, int lanes, int inverse, int kernel, int pad_mode);

    // Execute FFT using a pre-initialized context. Returns 1 on success, 0 on error.
    FFT_CFFI_API int fft_execute(void* handle,
                                 const float* in_pcm,
                                 float* out_real,
                                 float* out_imag,
                                 float* out_mag,
                                 size_t n);
    // Query effective FFT size (plan size) for a context
    FFT_CFFI_API size_t fft_ctx_size(void* handle);

    // Destroy a context created by fft_init (safe to pass NULL).
    FFT_CFFI_API void fft_free(void* handle);
}

