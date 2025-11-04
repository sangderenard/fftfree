#!/usr/bin/env python3
"""
Round-trip magnitude-only phase test:

- Loads a WAV, computes STFT (R2C, half-spectrum), discards phase (keeps magnitude)
- Reconstructs complex spectra using a chosen phase strategy (infer, linear, random, zero)
- Runs inverse STFT (C2R), overlap-adds, and compares against original
- Repeats for N trials and reports SNR/REL statistics (no files written by default)

Usage:
  python tools/roundtrip_phase_test.py input.wav \
      --trials 5 --N 1024 --hop 512 --phase infer --threads 4

"""
from pathlib import Path
import argparse
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


def _discover_library(explicit=None):
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
size_t fft_ctx_size(void* handle);

void* phase_init(int N, int hop, int half_spectrum, int mode, int iterations);
size_t phase_infer_execute(void* handle,
                          const float* in_mag,
                          size_t frames,
                          float* out_real,
                          float* out_imag);
void phase_free(void* handle);
"""
    )
    return ffi


def parse_args():
    p = argparse.ArgumentParser(description="Round-trip magnitude-only phase test")
    p.add_argument("input", help="Input WAV file")
    p.add_argument("--lib", help="Path to fft_cffi shared library")
    p.add_argument("--trials", type=int, default=5, help="Number of round-trip trials")
    p.add_argument("--N", type=int, default=1024, help="STFT size (power of two)")
    p.add_argument("--hop", type=int, default=512, help="Hop size between frames")
    p.add_argument("--threads", type=int, default=4, help="Worker threads for contexts")
    p.add_argument("--phase", choices=["infer", "linear", "random", "zero"], default="infer", help="Phase strategy")
    p.add_argument("--seed", type=int, default=1337, help="Seed for random phase")
    p.add_argument("--save", action="store_true", help="Save final reconstructed WAV (input_basename.rt.wav) and a PNG of STFT magnitude")
    # Visualization controls for PNG export
    # dB mapping (on by default). Use --no-db to disable.
    p.add_argument("--db-floor", type=float, default=-80.0, help="Floor (dB) for PNG mapping when dB enabled (default: -80 dB)")
    p.add_argument("--no-db", dest="use_db", action="store_false", help="Disable dB mapping for PNG; use linear min-max")
    p.set_defaults(use_db=True)
    # Optional explicit scale (default chosen from --no-db if not provided)
    p.add_argument("--scale", choices=["db", "linear", "log1p"], help="Magnitude->8-bit mapping for PNG (default: db unless --no-db)")
    p.add_argument("--db-ref", choices=["global", "frame"], default="global", help="dB reference: global (whole image max) or frame (per-column max)")
    p.add_argument("--sr", type=int, default=None, help="Override output sample rate")
    return p.parse_args()


def snr_db(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    num = np.sum(x * x) + 1e-30
    den = np.sum((x - y) ** 2) + 1e-30
    return 10.0 * np.log10(num / den)


def rel_error(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    max_abs = np.max(np.abs(x)) + 1e-30
    return np.max(np.abs(x - y)) / max_abs


def main():
    args = parse_args()
    # Back-compat default: if --scale not set, choose based on --no-db/--use_db
    if getattr(args, "scale", None) is None:
        args.scale = "db" if getattr(args, "use_db", True) else "linear"
    lib_path = _discover_library(args.lib)
    ffi = _build_ffi()
    lib = ffi.dlopen(lib_path)

    # Read audio
    sr, data = wavfile.read(args.input)
    if args.sr:
        sr = int(args.sr)
    if data.ndim > 1:
        data = data[:, 0]
    data = np.asarray(data, dtype=np.float32)

    N = int(args.N)
    H = int(args.hop)
    threads = int(args.threads)
    pad_mode = 1  # pad last frame if needed
    bins = N // 2 + 1

    # Forward context (R2C half-spectrum)
    fwd = lib.fft_init_full(
        int(N), threads, 1,
        0,  # inverse
        0, 0, ffi.NULL, 0,
        pad_mode,
        int(N), int(H), 1,
        1,  # transform=R2C
        0, 0,
        1,  # half_spectrum
        1, 0, 0,
        0,
        1,  # silent_crash_reports
    )
    if fwd == ffi.NULL:
        raise RuntimeError("fft_init_full (forward) failed")

    pcm = np.ascontiguousarray(data, dtype=np.float32)
    pcm_len = int(pcm.shape[0])
    # Estimate frames like the C path
    if pcm_len < N:
        frames_est = 1
    else:
        frames_est = 1 + (pcm_len - N) // H
        last_start = (frames_est - 1) * H
        remaining = pcm_len - last_start if pcm_len > last_start else 0
        if remaining < N:
            pass  # pad last frame

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
    # Trim to actual frames*bins
    out_mag = out_mag[: frames * bins]

    # Phase strategy
    in_real = np.zeros(frames * bins, dtype=np.float32)
    in_imag = np.zeros(frames * bins, dtype=np.float32)
    if args.phase == "infer":
        ph = lib.phase_init(int(N), int(H), 1, 0, 0)
        if ph == ffi.NULL:
            raise RuntimeError("phase_init failed")
        ok = lib.phase_infer_execute(
            ph,
            ffi.cast("float *", out_mag.ctypes.data),
            int(frames),
            ffi.cast("float *", in_real.ctypes.data),
            ffi.cast("float *", in_imag.ctypes.data),
        )
        lib.phase_free(ph)
        if int(ok) != int(frames):
            raise RuntimeError("phase_infer_execute failed")
    elif args.phase == "linear":
        two_pi = 2.0 * np.pi
        dphi = two_pi * np.arange(bins, dtype=np.float64) * float(H) / float(N)
        phi = np.zeros(bins, dtype=np.float64)
        mag_mat = out_mag.reshape(frames, bins)
        for f in range(frames):
            base = f * bins
            c = np.cos(phi).astype(np.float32)
            s = np.sin(phi).astype(np.float32)
            in_real[base:base + bins] = mag_mat[f].astype(np.float32) * c
            in_imag[base:base + bins] = mag_mat[f].astype(np.float32) * s
            phi += dphi
    elif args.phase == "random":
        rng = np.random.RandomState(args.seed)
        phases = rng.uniform(-np.pi, np.pi, size=(frames, bins)).astype(np.float32)
        mag_mat = out_mag.reshape(frames, bins)
        for f in range(frames):
            base = f * bins
            in_real[base:base + bins] = mag_mat[f] * np.cos(phases[f])
            in_imag[base:base + bins] = mag_mat[f] * np.sin(phases[f])
    else:  # zero
        mag_mat = out_mag.reshape(frames, bins)
        for f in range(frames):
            base = f * bins
            in_real[base:base + bins] = mag_mat[f]
            # imag remains zero

    # Inverse context (C2R)
    inv = lib.fft_init_full(
        int(N), threads, 1,
        1,  # inverse
        0, 0, ffi.NULL, 0,
        pad_mode,
        int(N), int(H), 1,
        2,  # transform=C2R
        0, 0,
        1,  # half_spectrum
        1, 0, 0,
        0,
        1,
    )
    if inv == ffi.NULL:
        lib.fft_free(fwd)
        raise RuntimeError("fft_init_full (inverse) failed")

    out_frames_buf = np.zeros(frames * N, dtype=np.float32)
    ok = lib.fft_execute_complex_batched(
        inv,
        ffi.cast("float *", in_real.ctypes.data),
        ffi.cast("float *", in_imag.ctypes.data),
        int(frames),
        ffi.cast("float *", out_frames_buf.ctypes.data),
        pad_mode,
        1,
        0,
    )
    if ok == 0:
        lib.fft_free(fwd)
        lib.fft_free(inv)
        raise RuntimeError("fft_execute_complex_batched failed")

    # Overlap-add
    out_len = (frames - 1) * H + N
    recon = np.zeros(out_len, dtype=np.float32)
    counts = np.zeros(out_len, dtype=np.float32)
    for f in range(frames):
        start = f * H
        frame_pcm = out_frames_buf[f * N:(f + 1) * N]
        recon[start:start + N] += frame_pcm
        counts[start:start + N] += 1.0
    mask = counts > 0
    recon[mask] /= counts[mask]

    # Trim original to out_len for fair compare (or pad with zeros)
    if pcm.shape[0] < out_len:
        x = np.zeros(out_len, dtype=np.float32)
        x[: pcm.shape[0]] = pcm
    else:
        x = pcm[:out_len]

    # Trials
    snrs = []
    rels = []
    trials = max(1, int(args.trials))
    for _ in range(trials):
        snrs.append(snr_db(x, recon))
        rels.append(rel_error(x, recon))

    # Stats
    snrs = np.array(snrs)
    rels = np.array(rels)
    print(f"Trials={trials} N={N} H={H} phase={args.phase}")
    print(f"SNR: mean={snrs.mean():.3f} dB std={snrs.std(ddof=0):.3f} min={snrs.min():.3f} max={snrs.max():.3f}")
    print(f"REL: mean={rels.mean():.6g} std={rels.std(ddof=0):.6g} min={rels.min():.6g} max={rels.max():.6g}")

    if args.save:
        from scipy.io.wavfile import write
        out_path = Path(args.input).with_suffix(".rt.wav")
        max_abs = float(np.max(np.abs(recon)))
        if max_abs > 0:
            scaled = recon / max_abs * 0.9
        else:
            scaled = recon
        int16 = np.int16(np.clip(scaled * 32767.0, -32768, 32767))
        write(str(out_path), int(sr), int16)
        print(f"Saved reconstructed WAV to {out_path}")

        # Save magnitude PNG from the forward STFT as grayscale for reference
        # Also print raw magnitude stats to verify data range
        mag_mat = out_mag.reshape(frames, bins).T.astype(np.float32)  # (bins, frames)
        try:
            vec = mag_mat.ravel()
            finite = vec[np.isfinite(vec)] if vec.size else vec
            if finite.size > 0:
                mn = float(finite.min()); mx = float(finite.max())
                mean = float(finite.mean()); std = float(finite.std(ddof=0))
                zeros = int(np.sum(finite == 0.0)); nz = int(finite.size - zeros)
                p50 = float(np.percentile(finite, 50))
                p90 = float(np.percentile(finite, 90))
                p99 = float(np.percentile(finite, 99))
                print(f"RAW_MAG_STATS: frames={frames} bins/frame={bins} size={finite.size}")
                print(f"  min={mn:.6g} max={mx:.6g} mean={mean:.6g} std={std:.6g} zeros={zeros} nonzeros={nz}")
                print(f"  p50={p50:.6g} p90={p90:.6g} p99={p99:.6g}")
            else:
                print("RAW_MAG_STATS: empty magnitude vector")
        except Exception as _ex:
            print(f"[warn] exception computing RAW_MAG_STATS: {_ex}")
        # Visualization scale helper
        def _normalize_linear_0_255(arr: np.ndarray) -> np.ndarray:
            a = np.array(arr, dtype=np.float32, copy=True)
            if a.size == 0:
                return a
            min_val = float(np.min(a))
            a -= min_val
            max_val = float(np.max(a))
            if max_val == 0.0:
                return np.zeros_like(a, dtype=np.float32)
            a /= (max_val + 1e-8)
            a *= 255.0
            return a

        def _mag_viz_u8(mag_mat_: np.ndarray) -> np.ndarray:
            scale = args.scale
            if scale == "linear":
                return _normalize_linear_0_255(mag_mat_).astype(np.uint8)
            if scale == "db":
                eps = 1e-12
                m = np.array(mag_mat_, dtype=np.float32, copy=True)
                if args.db_ref == "frame":
                    ref = np.max(m, axis=0, keepdims=True) + eps
                    m_norm = m / ref
                else:
                    ref = float(np.max(m)) + eps
                    m_norm = m / ref
                m_norm = np.clip(m_norm, eps, 1.0)
                db = 20.0 * np.log10(m_norm)
                floor = float(args.db_floor)
                db = np.clip(db, floor, 0.0)
                norm = (db - floor) / (0.0 - floor + 1e-12)
                return np.uint8(np.clip(norm * 255.0, 0, 255))
            # log1p scale
            m = np.array(mag_mat_, dtype=np.float32, copy=True)
            ref = np.max(m)
            if ref <= 0:
                return np.zeros_like(m, dtype=np.uint8)
            med = float(np.median(m)) if m.size else 0.0
            k = 1.0/med if med > 0 else 1.0/ref
            num = np.log1p(k * m)
            den = np.log1p(k * ref) + 1e-12
            norm = np.clip(num / den, 0.0, 1.0)
            return np.uint8(np.clip(norm * 255.0, 0, 255))

        mag_img = _mag_viz_u8(mag_mat)
        png_path = Path(args.input).with_suffix(".rt.png")
        Image.fromarray(mag_img, "L").save(str(png_path))
        print(f"Saved magnitude PNG to {png_path} (shape={mag_img.shape}, mag-scale={args.scale}, ref={args.db_ref}, floor={args.db_floor} dB)")

    lib.fft_free(fwd)
    lib.fft_free(inv)


if __name__ == "__main__":
    main()
