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
    FFT_CFFI_API void fft_pcm_to_channels(const float* in_pcm, float* out_real, float* out_imag, float* out_mag, size_t n);
}

