#!/usr/bin/env python3
"""
Convert an FFT PNG (frequency × frames image) back into a WAV by
inventing phases for magnitude-only inputs.

This is a playful helper: it reads a PNG produced by `fft_to_png.py` (RGB
channels usually: real, imag, mag) and uses the grayscale/magnitude channel
to synthesize complex spectra by adding random phases per-bin/frame. Then it
runs the inverse FFT via the library's `fft_execute_complex_batched` C API and
overlap-adds frames into a PCM stream which is written to disk as a WAV.

Limitations:
- The script invents phases randomly; results are synthetic and not a true
  reconstruction of the original audio.
- If the source image encoded half‑spectrum packing, enable `--half` so the
  inverse uses (N/2+1) bins per frame.

Usage examples:
  png_to_wav.py input.png out.wav --sr 44100 --amp 1.0 --hop 512 --half

"""
from pathlib import Path
import argparse
import os
import sys
import numpy as np
from PIL import Image
from cffi import FFI
from scipy.io import wavfile


def _candidate_library_names():
    if os.name == "nt":
        yield "fft_cffi.dll"
    elif sys.platform == "darwin":
        yield "libfft_cffi.dylib"
    else:
        yield "libfft_cffi.so"


def _discover_library(explicit=None):
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Library {p} not found")
        return str(p)
    project_root = Path(__file__).resolve().parent
    build_root = project_root / "build"
    search_dirs = [project_root]
    if build_root.exists():
        search_dirs.append(build_root)
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
        return env
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

void* fft_init_full_v2(size_t n,
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
                    int silent_crash_reports,
                    int apply_windows,
                    int apply_ola,
                    int analysis_window_kind,
                    float analysis_param1,
                    float analysis_param2,
                    int synthesis_window_kind,
                    float synthesis_param1,
                    float synthesis_param2,
                    int window_norm_policy,
                    int cola_mode);

size_t fft_execute_complex_batched(void* handle,
                                   const float* in_real,
                                   const float* in_imag,
                                   size_t frames,
                                   float* out_pcm,
                                   int pad_mode,
                                   int enable_backup,
                                   size_t max_frames);

void fft_free(void* handle);
size_t fft_ctx_size(void* handle);
"""
    )
    return ffi


def parse_args():
    p = argparse.ArgumentParser(description="Invert FFT PNG into WAV by inventing phases")
    p.add_argument("input", help="Input PNG file (height=N, width=frames)")
    p.add_argument("output", help="Output WAV file")
    p.add_argument("--lib", help="Path to fft_cffi shared library")
    p.add_argument("--sr", type=int, default=44100, help="Output sample rate")
    p.add_argument("--amp", type=float, default=1.0, help="Magnitude scale factor (for visual->linear) default=1.0")
    p.add_argument("--phase", choices=["zero","random","linear","infer","from_png"], default="infer", help="Phase strategy: infer (minimum phase, default), zero, random, linear, or from_png if RGB real/imag present")
    p.add_argument("--seed", type=int, default=None, help="Random seed for phase=random")
    p.add_argument("--hop", type=int, default=0, help="Hop size between frames. Default: equal to N (no overlap-add).")
    p.add_argument("--use-meta", action="store_true", help="Use PNG metadata to invert scaling/orientation when possible")
    p.add_argument("--flip-ud", action="store_true", help="Force vertical flip of input image (low freq at bottom)")
    p.add_argument("--half", action="store_true", help="Treat image rows as half-spectrum (N/2+1 bins)")
    p.add_argument("--threads", type=int, default=4, help="Worker threads for FFT context")
    return p.parse_args()


def main():
    args = parse_args()
    img_path = Path(args.input)
    out_wav = Path(args.output)
    lib_path = _discover_library(args.lib)

    ffi = _build_ffi()
    lib = ffi.dlopen(lib_path)

    # Load image and metadata
    im = Image.open(str(img_path))
    meta = {}
    try:
        if hasattr(im, "text"):
            meta = dict(im.text)
        elif hasattr(im, "info"):
            meta = dict(im.info)
    except Exception:
        meta = {}
    arr = np.array(im)
    # Optional orientation flip
    orient = meta.get("FFTFREE_ORIENT", "") if args.use_meta else ""
    if args.flip-ud or (args.use_meta and orient.lower() == "bins_bottom"):
        arr = np.flipud(arr)

    # Determine dims
    N = arr.shape[0]
    frames = arr.shape[1]
    print(f"Image size: N={N} bins, frames={frames}")

    # Extract magnitude (and optional phase channels)
    use_png_complex = False
    if arr.ndim == 2:
        mag = arr.astype(np.float32)
        if args.use_meta and meta:
            scale = meta.get("FFTFREE_SCALE", "")
            invertible = meta.get("FFTFREE_INVERTIBLE", "0") == "1"
            if invertible and scale == "linear":
                try:
                    mmin = float(meta.get("FFTFREE_MAG_MIN", "0"))
                    mmax = float(meta.get("FFTFREE_MAG_MAX", "0"))
                    if mmax > mmin:
                        mag = mmin + (mag / 255.0) * (mmax - mmin)
                except Exception:
                    pass
            elif invertible and scale == "db" and meta.get("FFTFREE_DB_REF", "") == "global":
                try:
                    ref = float(meta.get("FFTFREE_DB_GLOBAL_REF", "0"))
                    floor = float(meta.get("FFTFREE_DB_FLOOR", "-80"))
                    norm = np.clip(mag / 255.0, 0.0, 1.0)
                    db = floor + norm * (0.0 - floor)
                    m_norm = np.power(10.0, db / 20.0)
                    mag = m_norm * ref
                except Exception:
                    pass
        # Normalize 0..255 -> 0..1 then scale
        if mag.max() > 1.0:
            mag = mag / 255.0
        mag = mag * float(args.amp)
    else:
        # RGB: expect real, imag, magnitude
        real_u8 = arr[:, :, 0].astype(np.float32)
        imag_u8 = arr[:, :, 1].astype(np.float32)
        mag = arr[:, :, 2].astype(np.float32)
        real = real_u8; imag = imag_u8
        if args.use_meta and meta:
            try:
                rmin = float(meta.get("FFTFREE_REAL_MIN", "0"))
                rmax = float(meta.get("FFTFREE_REAL_MAX", "0"))
                imin = float(meta.get("FFTFREE_IMAG_MIN", "0"))
                imax = float(meta.get("FFTFREE_IMAG_MAX", "0"))
                if rmax > rmin:
                    real = rmin + (real_u8 / 255.0) * (rmax - rmin)
                if imax > imin:
                    imag = imin + (imag_u8 / 255.0) * (imax - imin)
                if args.phase == "from_png":
                    use_png_complex = True
            except Exception:
                use_png_complex = False
        # magnitude channel: treat as 0..255
        mag = (mag / 255.0) * float(args.amp)

    # Decide bins per frame for complex input
    if args.half:
        bins = N // 2 + 1
        print(f"Using half_spectrum packing: bins={bins}")
    else:
        bins = N
        print(f"Using full spectrum: bins={bins}")

    # If half-spectrum but image contains full N rows, take first bins rows
    if bins < N:
        mag = mag[:bins, :]

    # Determine plan size: auto-pad N to next power-of-two for plan stability
    def _next_pow2(v: int) -> int:
        if v <= 1:
            return 1
        vv = v - 1
        vv |= vv >> 1; vv |= vv >> 2; vv |= vv >> 4; vv |= vv >> 8; vv |= vv >> 16
        return vv + 1

    plan_N = _next_pow2(N)
    if plan_N != N:
        print(f"Auto-padding plan N: {N} -> {plan_N}")
    plan_bins = plan_N // 2 + 1 if args.half else plan_N

    # Build complex arrays sized for the plan (frame-major)
    in_real = np.zeros(frames * plan_bins, dtype=np.float32)
    in_imag = np.zeros(frames * plan_bins, dtype=np.float32)
    if use_png_complex and args.phase == "from_png":
        # Use reconstructed complex from PNG meta (if present)
        # Arrange as (bins, frames)
        if real.ndim == 2 and imag.ndim == 2:
            for f in range(frames):
                base = f * plan_bins
                in_real[base:base + bins] = real[:bins, f]
                in_imag[base:base + bins] = imag[:bins, f]
        else:
            # Fallback to infer if shapes aren’t 2D
            use_png_complex = False
    if not use_png_complex and args.phase == "infer":
        # Phase inference via C ABI
        ffi2 = FFI()
        ffi2.cdef("""
        void* phase_init(int N, int hop, int half_spectrum, int mode, int iterations);
        size_t phase_infer_execute(void* handle,
                                  const float* in_mag,
                                  size_t frames,
                                  float* out_real,
                                  float* out_imag);
        void phase_free(void* handle);
        """)
        lib2 = ffi2.dlopen(lib_path)
        hop = args.hop if args.hop > 0 else N
        half_flag = 1 if args.half else 0
        # Mode 1 => minimum-phase reconstruction (via cepstrum)
        ph = lib2.phase_init(int(plan_N), int(hop), int(half_flag), 1, 0)
        if ph == ffi2.NULL:
            raise RuntimeError("phase_init failed")
        # Frame-major magnitudes padded to plan_bins
        mag_plan = np.zeros((plan_bins, frames), dtype=np.float32)
        mag_plan[:bins, :] = mag
        mag_flat = mag_plan.T.reshape(-1).astype(np.float32)
        produced = lib2.phase_infer_execute(ph,
                                            ffi2.cast("float *", mag_flat.ctypes.data),
                                            int(frames),
                                            ffi2.cast("float *", in_real.ctypes.data),
                                            ffi2.cast("float *", in_imag.ctypes.data))
        lib2.phase_free(ph)
        if int(produced) != int(frames):
            raise RuntimeError("phase_infer_execute failed")
    elif args.phase == "random":
        rng = np.random.RandomState(args.seed)
        phases = rng.uniform(-np.pi, np.pi, size=(bins, frames)).astype(np.float32)
        for f in range(frames):
            # copy image bins into lower portion; pad high end with zeros
            base = f * plan_bins
            for b in range(bins):
                m = float(mag[b, f])
                ph = float(phases[b, f])
                in_real[base + b] = m * np.cos(ph)
                in_imag[base + b] = m * np.sin(ph)
    elif args.phase == "linear":
        # Deterministic linear phase propagation per bin
        hop = args.hop if args.hop > 0 else N
        two_pi = 2.0 * np.pi
        dphi = two_pi * np.arange(plan_bins, dtype=np.float64) * float(hop) / float(plan_N)
        phi = np.zeros(plan_bins, dtype=np.float64)
        for f in range(frames):
            base = f * plan_bins
            c = np.cos(phi).astype(np.float32)
            s = np.sin(phi).astype(np.float32)
            in_real[base:base + bins] = mag[:, f].astype(np.float32) * c[:bins]
            in_imag[base:base + bins] = mag[:, f].astype(np.float32) * s[:bins]
            phi += dphi
    else:
        # Zero phase: complex = magnitude + i*0
        for f in range(frames):
            base = f * plan_bins
            in_real[base:base + bins] = mag[:, f]
            # high-end already zeroed in allocation

    # Create inverse FFT context
    STFT_N = plan_N
    STFT_W = N
    STFT_H = args.hop if args.hop > 0 else N
    pad_mode = 1  # pad last frame if needed
    transform = 2  # C2R
    half_flag = 1 if args.half else 0

    init_full_v2 = getattr(lib, 'fft_init_full_v2', None)
    init_full = getattr(lib, 'fft_init_full', None)
    if init_full_v2 is not None:
        ctx = init_full_v2(
            STFT_N,
            int(args.threads),
            1,
            1,  # inverse
            0,
            0,
            ffi.NULL,
            0,
            pad_mode,
            STFT_W,
            STFT_H,
            1,
            transform,
            0,
            0,
            half_flag,
            1,
            0,
            0,
            0,
            0,
            0,  # apply_windows
            0,  # apply_ola
            0,  # analysis_window_kind
            0.0,
            0.0,
            0,  # synthesis_window_kind
            0.0,
            0.0,
            0,  # window_norm_policy
            0,  # cola_mode
        )
    else:
        ctx = init_full(
        STFT_N,
        int(args.threads),
        1,
        1,  # inverse
        0,
        0,
        ffi.NULL,
        0,
        pad_mode,
        STFT_W,
        STFT_H,
        1,
        transform,
        0,
        0,
        half_flag,
        1,
        0,
        0,
        0,
        0,
    )
    if ctx == ffi.NULL:
        raise RuntimeError("fft_init_full failed")

    # Prepare output buffer for per-frame PCM: frames * N
    out_frames_buf = np.zeros(frames * STFT_N, dtype=np.float32)

    produced = lib.fft_execute_complex_batched(ctx,
                                              ffi.cast("float *", in_real.ctypes.data),
                                              ffi.cast("float *", in_imag.ctypes.data),
                                              frames,
                                              ffi.cast("float *", out_frames_buf.ctypes.data),
                                              pad_mode,
                                              1,  # enable_backup
                                              0)
    if produced == 0:
        lib.fft_free(ctx)
        raise RuntimeError("fft_execute_complex_batched failed")

    # Overlap-add into final PCM
    hop = STFT_H
    out_len = (frames - 1) * hop + STFT_N
    final = np.zeros(out_len, dtype=np.float32)
    counts = np.zeros(out_len, dtype=np.float32)
    for f in range(frames):
        start = f * hop
        frame_pcm = out_frames_buf[f * STFT_N:(f + 1) * STFT_N]
        final[start:start + STFT_N] += frame_pcm
        counts[start:start + STFT_N] += 1.0

    # Avoid divide-by-zero
    mask = counts > 0
    final[mask] /= counts[mask]

    # Normalize to int16
    max_abs = np.max(np.abs(final))
    if max_abs == 0:
        scaled = final
    else:
        scaled = final / max_abs * 0.9
    int16 = np.int16(np.clip(scaled * 32767.0, -32768, 32767))

    wavfile.write(str(out_wav), int(args.sr), int16)
    lib.fft_free(ctx)
    print(f"Wrote WAV to {out_wav} (sr={args.sr}, len={len(int16)} samples)")


if __name__ == "__main__":
    main()
