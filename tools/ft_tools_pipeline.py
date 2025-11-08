#!/usr/bin/env python3
"""
End-to-end FT tools pipeline demo/test.

- Loads a WAV file
- Computes STFT via fft_cffi shared library (R2C half-spectrum)
- Exports pre-processing magnitude PNG
- Applies the ft_tools pipeline (resample + quantizer + contrast units)
- Exports post-processing magnitude PNG
- Runs ISTFT (C2R) and overlap-adds to reconstruct PCM
- Writes reconstructed WAV
"""
from __future__ import annotations

import argparse
from pathlib import Path
import os
import sys
import numpy as np
from cffi import FFI
from PIL import Image
from scipy.io import wavfile


def _candidate_library_names():
    if os.name == "nt":
        yield "fft_cffi.dll"
    elif sys.platform == "darwin":
        yield "libfft_cffi.dylib"
    else:
        yield "libfft_cffi.so"


def _discover_library(explicit: str | None):
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Library {p} not found")
        return str(p)
    project_root = Path(__file__).resolve().parents[1]
    search_dirs = [project_root, project_root / "build", project_root / "build" / "Release"]
    for d in search_dirs:
        for name in _candidate_library_names():
            candidate = d / name
            if candidate.exists():
                return str(candidate)
            if d.exists():
                found = next(d.rglob(name), None)
                if found:
                    return str(found)
    env = os.environ.get("FFTFREE_CFFI_LIB")
    if env:
        p = Path(env).expanduser().resolve()
        if p.exists():
            return str(p)
    raise FileNotFoundError("Could not locate fft_cffi shared library. Use --lib or set FFTFREE_CFFI_LIB")


def _build_ffi():
    ffi = FFI()
    ffi.cdef(
        """
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
                    int inner_threads,
                    int save_crash_logs,
                    int silent_crash_reports);

size_t fft_execute_batched(void* handle,
                           const float* pcm,
                           size_t pcm_len,
                           float* out_real,
                           float* out_imag,
                           float* out_mag,
                           int pad_mode,
                           int enable_backup,
                           size_t max_frames);

size_t fft_execute_complex_batched(void* handle,
                                   const float* in_real,
                                   const float* in_imag,
                                   size_t frames,
                                   float* out_pcm,
                                   int pad_mode,
                                   int enable_backup,
                                   size_t max_frames);

void fft_free(void* handle);

struct ft_grid_spec {
  double time_origin;
  double freq_origin;
  double time_step;
  double freq_step;
};

struct ft_resample_plan_c {
  int upsample_time;
  int upsample_freq;
  int downsample_time;
  int downsample_freq;
  int preserve_energy;
};

struct ft_quantizer_config {
  int enabled;
  float base_frequency;
  float magnitude_step;
  float mix;
  float distance_decay;
};

struct ft_contrast_config {
  int enabled;
  int kernel_time;
  int kernel_freq;
  float brightness;
  float contrast;
  float gamma;
  int clamp_zero;
};

int ft_apply_complex_pipeline(const struct ft_grid_spec* spec,
                              const struct ft_resample_plan_c* plan,
                              const struct ft_quantizer_config* quant,
                              const struct ft_contrast_config* contrast,
                              float* real,
                              float* imag,
                              int time_bins,
                              int freq_bins);
"""
    )
    return ffi


def parse_args():
    p = argparse.ArgumentParser(description="FT tools STFT pipeline test")
    p.add_argument("input", help="Input WAV file (mono or multi-channel; first channel is used)")
    p.add_argument("--lib", help="Path to fft_cffi shared library")
    p.add_argument("--N", type=int, default=1024, help="STFT size / FFT length (power of two recommended)")
    p.add_argument("--hop", type=int, default=512, help="Hop size between STFT frames")
    p.add_argument("--threads", type=int, default=4, help="Worker threads for FFT contexts")
    p.add_argument("--upsample-time", type=int, default=1, help="Upsample factor in time before units run")
    p.add_argument("--upsample-freq", type=int, default=1, help="Upsample factor in frequency before units run")
    p.add_argument("--downsample-time", type=int, default=1, help="Downsample factor in time after units run")
    p.add_argument("--downsample-freq", type=int, default=1, help="Downsample factor in frequency after units run")
    p.add_argument("--no-preserve-energy", action="store_true", help="Disable averaging when downsampling")

    # Quantizer settings
    p.add_argument("--disable-quant", action="store_true", help="Disable exponential harmonic quantizer")
    p.add_argument("--quant-base", type=float, default=55.0, help="Base frequency for harmonic attraction")
    p.add_argument("--quant-step", type=float, default=0.05, help="Magnitude quantization step")
    p.add_argument("--quant-mix", type=float, default=0.5, help="Blend between original and quantized magnitude [0,1]")
    p.add_argument("--quant-decay", type=float, default=8.0, help="Exponential decay controlling attraction vs harmonic distance")

    # Contrast settings
    p.add_argument("--disable-contrast", action="store_true", help="Disable spectral local contrast boost")
    p.add_argument("--contrast-kernel-time", type=int, default=3, help="Kernel width in time bins (<=0 => global)")
    p.add_argument("--contrast-kernel-freq", type=int, default=3, help="Kernel height in frequency bins (<=0 => global)")
    p.add_argument("--contrast-brightness", type=float, default=0.0, help="Additive brightness applied after local contrast")
    p.add_argument("--contrast", type=float, default=0.5, help="Contrast gain relative to local mean")
    p.add_argument("--contrast-gamma", type=float, default=1.0, help="Gamma correction on magnitudes (>0)")
    p.add_argument("--no-contrast-clamp-zero", action="store_true", help="Allow negative magnitudes in contrast stage")

    # Outputs / visualization
    p.add_argument("--output-dir", help="Directory for outputs (defaults alongside input)")
    p.add_argument("--before-png", help="Override path for pre-processing magnitude PNG")
    p.add_argument("--after-png", help="Override path for post-processing magnitude PNG")
    p.add_argument("--output-wav", help="Override path for reconstructed WAV")
    p.add_argument("--png-scale", choices=["db", "linear", "log1p"], default="db", help="Magnitude -> PNG mapping")
    p.add_argument("--db-floor", type=float, default=-80.0, help="dB floor when --png-scale=db")
    p.add_argument("--db-ref", choices=["global", "frame"], default="global", help="Reference for dB scaling")
    p.add_argument("--no-db", dest="use_db", action="store_false", help="Shortcut for --png-scale=linear")
    p.set_defaults(use_db=True)

    p.add_argument("--sr", type=int, help="Override output sample rate (defaults to WAV sample rate)")
    return p.parse_args()


def _prepare_output_paths(args):
    inp = Path(args.input).resolve()
    out_dir = Path(args.output_dir).resolve() if args.output_dir else inp.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    before_png = Path(args.before_png).resolve() if args.before_png else out_dir / (inp.stem + ".ft-before.png")
    after_png = Path(args.after_png).resolve() if args.after_png else out_dir / (inp.stem + ".ft-after.png")
    out_wav = Path(args.output_wav).resolve() if args.output_wav else out_dir / (inp.stem + ".ft.wav")
    return before_png, after_png, out_wav


def _normalize_png(mag: np.ndarray, args) -> np.ndarray:
    m = np.array(mag, dtype=np.float32, copy=True)
    if m.size == 0:
        return np.zeros_like(m, dtype=np.uint8)
    scale = args.png_scale
    if not getattr(args, "use_db", True) and scale == "db":
        scale = "linear"
    if scale == "linear":
        mn = float(np.min(m))
        mx = float(np.max(m))
        if mx <= mn:
            return np.zeros_like(m, dtype=np.uint8)
        norm = (m - mn) / (mx - mn)
        return np.uint8(np.clip(norm * 255.0, 0, 255))
    if scale == "log1p":
        ref = float(np.max(m))
        if ref <= 0:
            return np.zeros_like(m, dtype=np.uint8)
        med = float(np.median(m)) if m.size else 0.0
        k = 1.0 / med if med > 0 else 1.0 / ref
        num = np.log1p(k * m)
        den = np.log1p(k * ref) + 1e-12
        norm = np.clip(num / den, 0.0, 1.0)
        return np.uint8(np.clip(norm * 255.0, 0, 255))
    # dB scale
    eps = 1e-12
    if args.db_ref == "frame":
        ref = np.max(m, axis=0, keepdims=True) + eps
        ref[ref <= eps] = 1.0
        m_norm = m / ref
    else:
        ref = float(np.max(m)) + eps
        if ref <= eps:
            return np.zeros_like(m, dtype=np.uint8)
        m_norm = m / ref
    m_norm = np.clip(m_norm, eps, 1.0)
    db = 20.0 * np.log10(m_norm)
    floor = float(args.db_floor)
    db = np.clip(db, floor, 0.0)
    norm = (db - floor) / (0.0 - floor + 1e-12)
    return np.uint8(np.clip(norm * 255.0, 0, 255))


def _save_png(mag_matrix: np.ndarray, path: Path, args):
    img = _normalize_png(mag_matrix, args)
    Image.fromarray(img, "L").save(str(path))
    print(f"[info] wrote {path} shape={img.shape} scale={args.png_scale}")


def _compute_frames(pcm_len: int, N: int, hop: int) -> int:
    if pcm_len <= 0:
        return 0
    if pcm_len <= N:
        return 1
    frames = 1 + (pcm_len - N) // hop
    last_start = (frames - 1) * hop
    remaining = pcm_len - last_start
    if remaining > 0 and remaining < N:
        frames += 1  # final padded frame
    return frames


def _overlap_add(frames_pcm: np.ndarray, frames: int, N: int, hop: int) -> np.ndarray:
    out_len = (frames - 1) * hop + N
    recon = np.zeros(out_len, dtype=np.float32)
    counts = np.zeros(out_len, dtype=np.float32)
    for f in range(frames):
        start = f * hop
        frame = frames_pcm[f * N:(f + 1) * N]
        recon[start:start + N] += frame
        counts[start:start + N] += 1.0
    mask = counts > 0
    recon[mask] /= counts[mask]
    return recon


def main():
    args = parse_args()
    if args.png_scale == "db" and not getattr(args, "use_db", True):
        args.png_scale = "linear"
    lib_path = _discover_library(args.lib)
    ffi = _build_ffi()
    lib = ffi.dlopen(lib_path)

    sr, data = wavfile.read(args.input)
    if data.ndim > 1:
        data = data[:, 0]
    data = np.asarray(data, dtype=np.float32)
    if args.sr:
        sr = int(args.sr)

    N = int(args.N)
    hop = int(args.hop)
    threads = max(1, int(args.threads))
    pad_mode = 1  # pad tail frame if needed

    bins = N // 2 + 1
    pcm = np.ascontiguousarray(data, dtype=np.float32)
    pcm_len = int(pcm.shape[0])
    frames_est = max(1, _compute_frames(pcm_len, N, hop))

    # Forward context (R2C half-spectrum)
    fwd = lib.fft_init_full(
        int(N), threads, 1,
        0,
        0, 0, ffi.NULL, 0,
        pad_mode,
        int(N), int(hop), 1,
        1,
        0, 0,
        1,
        1, 0, 0,
        0,
        1,
    )
    if fwd == ffi.NULL:
        raise RuntimeError("fft_init_full (forward) failed")

    out_real = np.zeros(frames_est * bins, dtype=np.float32)
    out_imag = np.zeros(frames_est * bins, dtype=np.float32)
    out_mag = np.zeros(frames_est * bins, dtype=np.float32)

    produced = lib.fft_execute_batched(
        fwd,
        ffi.cast("float *", pcm.ctypes.data),
        pcm_len,
        ffi.cast("float *", out_real.ctypes.data),
        ffi.cast("float *", out_imag.ctypes.data),
        ffi.cast("float *", out_mag.ctypes.data),
        pad_mode,
        0,
        0,
    )
    if produced == 0:
        lib.fft_free(fwd)
        raise RuntimeError("fft_execute_batched returned 0 frames")
    frames = int(produced)

    # Trim to produced*
    out_real = out_real[: frames * bins]
    out_imag = out_imag[: frames * bins]
    out_mag = out_mag[: frames * bins]

    mag_before = out_mag.reshape(frames, bins).T
    pcm_time_step = hop / float(sr)
    freq_step = sr / float(N)

    before_png, after_png, out_wav = _prepare_output_paths(args)
    _save_png(mag_before, before_png, args)

    # Prepare pipeline inputs
    real_proc = np.copy(out_real)
    imag_proc = np.copy(out_imag)

    spec = ffi.new("struct ft_grid_spec*")
    spec.time_origin = 0.0
    spec.freq_origin = 0.0
    spec.time_step = pcm_time_step if pcm_time_step > 0 else 1.0
    spec.freq_step = freq_step if freq_step > 0 else 1.0

    plan = ffi.new("struct ft_resample_plan_c*")
    plan.upsample_time = int(max(1, args.upsample_time))
    plan.upsample_freq = int(max(1, args.upsample_freq))
    plan.downsample_time = int(max(1, args.downsample_time))
    plan.downsample_freq = int(max(1, args.downsample_freq))
    plan.preserve_energy = 0 if args.no_preserve_energy else 1

    quant = ffi.new("struct ft_quantizer_config*")
    quant.enabled = 0 if args.disable_quant else 1
    quant.base_frequency = float(args.quant_base)
    quant.magnitude_step = float(args.quant_step)
    quant.mix = float(args.quant_mix)
    quant.distance_decay = float(args.quant_decay)

    contrast = ffi.new("struct ft_contrast_config*")
    contrast.enabled = 0 if args.disable_contrast else 1
    contrast.kernel_time = int(args.contrast_kernel_time)
    contrast.kernel_freq = int(args.contrast_kernel_freq)
    contrast.brightness = float(args.contrast_brightness)
    contrast.contrast = float(args.contrast)
    contrast.gamma = max(1e-3, float(args.contrast_gamma))
    contrast.clamp_zero = 0 if args.no_contrast_clamp_zero else 1

    ok = lib.ft_apply_complex_pipeline(
        spec,
        plan,
        quant,
        contrast,
        ffi.cast("float *", real_proc.ctypes.data),
        ffi.cast("float *", imag_proc.ctypes.data),
        frames,
        bins,
    )
    if ok == 0:
        lib.fft_free(fwd)
        raise RuntimeError("ft_apply_complex_pipeline failed (shape mismatch or invalid parameters)")

    mag_after = np.sqrt(real_proc.reshape(frames, bins).T ** 2 + imag_proc.reshape(frames, bins).T ** 2)
    _save_png(mag_after, after_png, args)

    # Inverse context (C2R half-spectrum)
    inv = lib.fft_init_full(
        int(N), threads, 1,
        1,
        0, 0, ffi.NULL, 0,
        pad_mode,
        int(N), int(hop), 1,
        2,
        0, 0,
        1,
        1, 0, 0,
        0,
        1,
    )
    if inv == ffi.NULL:
        lib.fft_free(fwd)
        raise RuntimeError("fft_init_full (inverse) failed")

    frames_pcm = np.zeros(frames * N, dtype=np.float32)
    recovered = lib.fft_execute_complex_batched(
        inv,
        ffi.cast("float *", real_proc.ctypes.data),
        ffi.cast("float *", imag_proc.ctypes.data),
        frames,
        ffi.cast("float *", frames_pcm.ctypes.data),
        pad_mode,
        0,
        frames,
    )
    if recovered == 0:
        lib.fft_free(fwd)
        lib.fft_free(inv)
        raise RuntimeError("fft_execute_complex_batched failed")

    recon = _overlap_add(frames_pcm, frames, N, hop)

    # Match length to original (pad or trim)
    if recon.shape[0] < pcm_len:
        padded = np.zeros(pcm_len, dtype=np.float32)
        padded[: recon.shape[0]] = recon
        recon = padded
    else:
        recon = recon[:pcm_len]

    # Scale to int16 for output
    max_abs = float(np.max(np.abs(recon)))
    if max_abs > 0:
        scaled = recon / max_abs * 0.9
    else:
        scaled = recon
    out_int16 = np.int16(np.clip(scaled * 32767.0, -32768, 32767))
    wavfile.write(str(out_wav), int(sr), out_int16)
    print(f"[info] wrote {out_wav} sr={sr} frames={frames} bins={bins}")

    lib.fft_free(fwd)
    lib.fft_free(inv)


if __name__ == "__main__":
    main()
