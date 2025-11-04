import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image
from cffi import FFI
from scipy.io import wavfile


ffi = FFI()
ffi.cdef(
    """
void fft_pcm_to_channels(const float* in_pcm,
                         float* out_real,
                         float* out_imag,
                         float* out_mag,
                         size_t n,
                         int threads);
void* fft_init(size_t n,
               int threads,
               int lanes,
               int inverse,
               int kernel,
               int radix,
               const int* radix_pattern,
               size_t radix_pattern_len,
               int pad_mode);
void* fft_init_ex(size_t n,
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
                  int stft_mode);
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
int fft_execute(void* handle,
                const float* in_pcm,
                float* out_real,
                float* out_imag,
                float* out_mag,
                size_t n);
size_t fft_ctx_size(void* handle);
size_t fft_ctx_worker_threads(void* handle);
size_t fft_ctx_effective_threads(void* handle);
void fft_free(void* handle);
size_t fft_execute_batched(void* handle,
                           const float* pcm,
                           size_t pcm_len,
                           float* out_real,
                           float* out_imag,
                           float* out_mag,
                           int pad_mode,
                           int enable_backup,
                           size_t max_frames);
"""
)


def _candidate_library_names() -> Iterable[str]:
    if os.name == "nt":
        yield "fft_cffi.dll"
    elif sys.platform == "darwin":
        yield "libfft_cffi.dylib"
    else:
        yield "libfft_cffi.so"


def _discover_library(explicit: Optional[str]) -> Path:
    """Return the path to the fft_cffi shared library.

    The lookup order is:
    1. User provided ``--lib`` argument.
    2. ``FFTFREE_CFFI_LIB`` environment variable.
    3. A heuristic search inside the local ``build`` directory.
    """

    if explicit:
        lib_path = Path(explicit).expanduser().resolve()
        if not lib_path.exists():
            raise FileNotFoundError(f"Specified library '{lib_path}' does not exist")
        return lib_path

    env_override = os.environ.get("FFTFREE_CFFI_LIB")
    if env_override:
        lib_path = Path(env_override).expanduser().resolve()
        if not lib_path.exists():
            raise FileNotFoundError(
                f"FFTFREE_CFFI_LIB points to '{lib_path}', which does not exist"
            )
        return lib_path


    project_root = Path(__file__).resolve().parent
    build_root = project_root / "build"
    release_root = build_root / "Release"
    search_dirs = [project_root]
    if build_root.exists():
        search_dirs.append(build_root)
    if release_root.exists():
        search_dirs.append(release_root)

    for directory in search_dirs:
        for name in _candidate_library_names():
            candidate = directory / name
            if candidate.exists():
                return candidate
            if directory.exists():
                match = next(directory.rglob(name), None)
                if match:
                    return match

    raise FileNotFoundError(
        "Could not locate fft_cffi shared library. "
        "Specify it explicitly with --lib or set FFTFREE_CFFI_LIB."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an FFT RGB image from WAV audio")
    parser.add_argument("input", help="Path to a mono or stereo WAV file")
    parser.add_argument("output", help="Destination PNG file (ignored when --no-image is used)")
    parser.add_argument(
        "--lib",
        help="Explicit path to the fft_cffi shared library (overrides FFTFREE_CFFI_LIB)",
    )
    parser.add_argument("--threads", type=int, default=8, help="Worker threads (default: 8)")
    parser.add_argument("--lanes", type=int, default=0, help="Lane capacity hint (0=auto)")
    parser.add_argument(
        "--kernel",
        choices=["auto", "ct", "stockham"],
        default="auto",
        help="FFT kernel selection: auto, ct (Cooley–Tukey), stockham",
    )
    parser.add_argument(
        "--transform",
        choices=["r2c", "c2c"],
        default="r2c",
        help="Transform mode for forward STFT: r2c (default, real->complex, halves bands) or c2c (full complex)",
    )
    parser.add_argument(
        "--output",
        choices=["mag", "rgb"],
        default="mag",
        help="Image output: mag (grayscale magnitude) or rgb (real/imag/mag stack)",
    )
    parser.add_argument(
        "--full-spectrum",
        action="store_true",
        help="Force full spectrum rows in output (disables half-spectrum packing)",
    )
    parser.add_argument(
        "--pad",
        choices=["auto", "always", "never"],
        default="auto",
        help="Padding policy: auto (required by kernel), always, or never",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print descriptive statistics for the complex FFT output",
    )
    parser.add_argument(
        "--no-image",
        action="store_true",
        help="Skip writing the FFT visualization PNG",
    )
    # Magnitude display mapping: default to a non-destructive nonlinear (log1p) mapping
    parser.add_argument(
        "--scale",
        choices=["linear", "log1p", "db"],
        default="log1p",
        help="Magnitude visualization scale: linear, log1p (default), or db",
    )
    # dB controls (used only when --scale=db)
    parser.add_argument(
        "--no-db",
        dest="use_db",
        action="store_false",
        help="Disable dB mapping for magnitude image; use linear min-max instead",
    )
    parser.add_argument(
        "--db-floor",
        type=float,
        default=-80.0,
        help="Magnitude floor (dB) for PNG mapping when dB is enabled (default: -80 dB)",
    )
    parser.add_argument(
        "--db-ref",
        choices=["global", "frame"],
        default="global",
        help="Reference for dB mapping: global = normalize by global max; frame = normalize each frame by its per-frame max",
    )
    parser.set_defaults(use_db=True)
    return parser.parse_args()


def main() -> None:
    import traceback
    try:
        args = _parse_args()

        lib_path = _discover_library(args.lib)
        lib = ffi.dlopen(str(lib_path))
        try:
            fft_fn = lib.fft_pcm_to_channels
        except AttributeError as exc:  # pragma: no cover - defensive path for misbuilt libs
            print("ERROR: Shared library is missing 'fft_pcm_to_channels'. Ensure fft_cffi was built and exported correctly.")
            traceback.print_exc()
            return

        # Read audio file
        _sample_rate, data = wavfile.read(args.input)
        if data.ndim > 1:
            data = data[:, 0]  # Use first channel if stereo
        data = np.ascontiguousarray(data, dtype=np.float32)
        n = int(data.shape[0])
        orig_n = n
        if n <= 0:
            raise ValueError("Input WAV contains no samples")

        # Optional padding here: only if user requested always-pad. Otherwise let the library decide.
        if args.pad == "always":
            next_pow2 = 1 << (n - 1).bit_length()
            if next_pow2 != n:
                pad = next_pow2 - n
                data = np.pad(data, (0, pad), mode="constant")
                n = next_pow2
                print(f"Padding: extended {orig_n} -> {n} samples (+{pad}) to meet power-of-two length.")

        # Map kernel/pad to ABI codes
        kernel_code = {"auto": 0, "ct": 1, "stockham": 2}[args.kernel]
        pad_code = {"auto": 0, "always": 1, "never": 2}[args.pad]

        # Use STFT batched helper: create a plan with N=1024 bins and window/hop
        STFT_N = 1024
        STFT_W = 1024
        STFT_H = 512
        # Prefer the full initializer (creates worker pool when requested) while
        # preserving the STFT/window/hop behavior from fft_init_ex. Fall back to
        # fft_init_ex when the shared library doesn't export the full API.
        init_full_v2 = getattr(lib, "fft_init_full_v2", None)
        init_full = getattr(lib, "fft_init_full", None)
        if init_full_v2 is not None:
            # Parameters: n, threads, lanes, inverse, kernel, radix, radix_pattern, radix_pattern_len,
            # pad_mode, window, hop, stft_mode,
            # transform, reduce_magnitude, store_polar, half_spectrum,
            # allow_outer_parallel, allow_inner_parallel, inner_threads
            # Map transform and half-spectrum
            transform_code = 1 if args.transform == "r2c" else 0
            half_flag = 0 if args.full_spectrum else (1 if args.transform == "r2c" else 0)
            ctx = init_full_v2(
                STFT_N,
                args.threads,
                args.lanes,
                0,              # inverse
                kernel_code,
                0,              # radix (unspecified)
                ffi.NULL,
                0,              # radix_pattern_len
                pad_code,
                STFT_W,
                STFT_H,
                1,              # stft_mode
                transform_code, # transform
                0,              # reduce_magnitude
                0,              # store_polar
                half_flag,      # half_spectrum
                1,              # allow_outer_parallel -> create WorkerPool
                0,              # allow_inner_parallel
                0,              # inner_threads
                0,              # save_crash_logs
                0,              # silent_crash_reports
                0,              # apply_windows (off)
                0,              # apply_ola (off)
                0,              # analysis_window_kind (RECT)
                0.0,            # analysis_param1
                0.0,            # analysis_param2
                0,              # synthesis_window_kind (RECT)
                0.0,            # synthesis_param1
                0.0,            # synthesis_param2
                0,              # window_norm_policy (NONE)
                0               # cola_mode (OFF)
            )
        elif init_full is not None:
            ctx = init_full(
                STFT_N,
                args.threads,
                args.lanes,
                0,
                kernel_code,
                0,
                ffi.NULL,
                0,
                pad_code,
                STFT_W,
                STFT_H,
                1,
                transform_code,
                0,
                0,
                half_flag,
                1,
                0,
                0,
                0,
                0
            )
        else:
            # Older/shared libs may not export fft_init_full; keep previous behavior.
            ctx = lib.fft_init_ex(STFT_N, args.threads, args.lanes, 0, kernel_code, 0, ffi.NULL, 0, pad_code, STFT_W, STFT_H, 1)
        if ctx == ffi.NULL:
            print("ERROR: fft_init_ex failed (ensure N is power-of-two and library is built correctly)")
            return
        worker_threads = int(lib.fft_ctx_worker_threads(ctx))
        effective_threads = int(lib.fft_ctx_effective_threads(ctx))
        if worker_threads <= 0:
            worker_threads = 1
        if effective_threads <= 0:
            effective_threads = 1
        if worker_threads <= 1 or effective_threads <= 1:
            print(
                f"WARNING: FFT context running single-threaded "
                f"(worker_threads={worker_threads}, effective_threads={effective_threads})."
            )
        else:
            print(
                f"FFT context using {worker_threads} worker threads "
                f"(effective plan threads={effective_threads})."
            )
        # Estimate expected frames so we can allocate outputs. Use same logic as C helper.
        pcm_len = int(data.shape[0])
        W = STFT_W
        H = STFT_H
        if pcm_len < W:
            if pad_code == 1:
                frames = 1
            else:
                print("ERROR: input shorter than window and pad policy forbids padding")
                lib.fft_free(ctx)
                return
        else:
            frames = 1 + (pcm_len - W) // H
            last_start = (frames - 1) * H
            remaining = pcm_len - last_start if pcm_len > last_start else 0
            if remaining < W:
                if pad_code == 1:
                    pass
                elif pad_code == 2:
                    if frames > 0:
                        frames -= 1
                else:
                    print("ERROR: trailing partial frame and pad_mode=auto refuses padding")
                    lib.fft_free(ctx)
                    return

        if frames <= 0:
            print("ERROR: No frames to process")
            lib.fft_free(ctx)
            return

        plan_n = lib.fft_ctx_size(ctx)
        if plan_n == 0:
            print("ERROR: fft_ctx_size returned 0")
            lib.fft_free(ctx)
            return
        if plan_n != STFT_N and args.pad != "always":
            print(f"Padding (library): plan N adjusted {STFT_N} -> {plan_n}")

        N_eff = int(plan_n)
        # Determine bins per frame based on half_spectrum flag actually used
        bins_per_frame = (N_eff // 2 + 1) if (not args.full_spectrum and args.transform == "r2c") else N_eff
        # allocate flattened output buffers: frames * bins
        out_count = frames * bins_per_frame
        out_real = np.zeros(out_count, dtype=np.float32)
        out_imag = np.zeros(out_count, dtype=np.float32)
        out_mag = np.zeros(out_count, dtype=np.float32)

        try:
            produced = lib.fft_execute_batched(
                ctx,
                ffi.cast("float *", data.ctypes.data),
                pcm_len,
                ffi.cast("float *", out_real.ctypes.data),
                ffi.cast("float *", out_imag.ctypes.data),
                ffi.cast("float *", out_mag.ctypes.data),
                pad_code,
                0,
                0,
            )
            if produced == 0:
                print("ERROR: fft_execute_batched failed or produced 0 frames")
                return
            frames = int(produced)
        finally:
            lib.fft_free(ctx)

        # Normalization helpers used by both the PNG writer and stats reporting.
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

        def _mag_viz_u8(mag_mat: np.ndarray) -> np.ndarray:
            # mag_mat: (bins, frames), float32 magnitude
            scale = args.scale
            if scale == "linear":
                return _normalize_linear_0_255(mag_mat).astype(np.uint8)
            if scale == "db":
                eps = 1e-12
                m = np.array(mag_mat, dtype=np.float32, copy=True)
                if args.db_ref == "frame":
                    ref = np.max(m, axis=0, keepdims=True) + eps
                    m_norm = m / ref
                else:
                    ref = float(np.max(m)) + eps
                    m_norm = m / ref
                m_norm = np.clip(m_norm, eps, 1.0)
                db = 20.0 * np.log10(m_norm)
                floor_db = float(args.db_floor)
                db = np.clip(db, floor_db, 0.0)
                norm = (db - floor_db) / (0.0 - floor_db + 1e-12)
                return np.uint8(np.clip(norm * 255.0, 0, 255))
            # log1p mapping: y = log1p(k * m) / log1p(k * m_ref)
            # Choose k adaptively from median to place median around mid-gray.
            m = np.array(mag_mat, dtype=np.float32, copy=True)
            ref = np.max(m)
            if ref <= 0:
                return np.zeros_like(m, dtype=np.uint8)
            med = float(np.median(m)) if m.size else 0.0
            # Aim: log1p(k*med)/log1p(k*ref) ~= 0.5 -> solve for k numerically simple
            # Use heuristic k: k = 1.0/med if med>0 else 1.0/ref
            if med > 0:
                k = 1.0 / med
            else:
                k = 1.0 / ref
            num = np.log1p(k * m)
            den = np.log1p(k * ref) + 1e-12
            norm = np.clip(num / den, 0.0, 1.0)
            return np.uint8(np.clip(norm * 255.0, 0, 255))

        # Trim outputs to produced frames before any stats/reshapes
        used = frames * bins_per_frame
        out_real = out_real[:used]
        out_imag = out_imag[:used]
        out_mag  = out_mag[:used]

        if args.stats:
            def describe(name: str, arr: np.ndarray) -> None:
                finite = arr[np.isfinite(arr)]
                if finite.size == 0:
                    print(f"{name}: no finite values to summarize")
                    return
                print(
                    f"{name}: min={finite.min():.6g} max={finite.max():.6g} "
                    f"mean={finite.mean():.6g} std={finite.std(ddof=0):.6g}"
                )

            print("FFT output statistics (real/imaginary/magnitude):")
            describe("real", out_real)
            describe("imag", out_imag)
            describe("mag", out_mag)

            print("Normalized FFT channel statistics (0-255 visualization scale):")

            # Helper for stats-only normalization (linear 0..255). Visualization
            # proper uses _mag_viz_u8 below which applies --scale (linear/log1p/db).
            def _normalize_for_image(arr: np.ndarray) -> np.ndarray:
                return _normalize_linear_0_255(arr)

            def describe_normalized(name: str, arr: np.ndarray) -> None:
                normalized = _normalize_for_image(arr)
                describe(name, normalized)

            describe_normalized("real_norm", out_real)
            describe_normalized("imag_norm", out_imag)
            describe_normalized("mag_norm", out_mag)

        if not args.no_image:
            # Raw magnitude stats just before visualization to validate data range
            try:
                mag_vec = np.asarray(out_mag, dtype=np.float32)
                used_frames = int(frames)
                used_bins = int(bins_per_frame)
                if mag_vec.size != used_frames * used_bins:
                    print(f"[warn] magnitude size mismatch: vec={mag_vec.size} expected={used_frames*used_bins}")
                finite = mag_vec[np.isfinite(mag_vec)] if mag_vec.size else mag_vec
                if finite.size > 0:
                    mn = float(finite.min()); mx = float(finite.max())
                    mean = float(finite.mean()); std = float(finite.std(ddof=0))
                    zeros = int(np.sum(finite == 0.0))
                    nz = int(finite.size - zeros)
                    p50 = float(np.percentile(finite, 50))
                    p90 = float(np.percentile(finite, 90))
                    p99 = float(np.percentile(finite, 99))
                    print(f"RAW_MAG_STATS: frames={used_frames} bins/frame={used_bins} size={finite.size}")
                    print(f"  min={mn:.6g} max={mx:.6g} mean={mean:.6g} std={std:.6g} zeros={zeros} nonzeros={nz}")
                    print(f"  p50={p50:.6g} p90={p90:.6g} p99={p99:.6g}")
                else:
                    print("RAW_MAG_STATS: empty magnitude vector")
            except Exception as _ex_magstats:
                print(f"[warn] exception computing raw magnitude stats: {_ex_magstats}")

            # Prepare channel images
            # Build matrices shaped for image construction first
            try:
                real_mat_full = _normalize_linear_0_255(out_real).astype(np.uint8).reshape((frames, bins_per_frame)).T
                imag_mat_full = _normalize_linear_0_255(out_imag).astype(np.uint8).reshape((frames, bins_per_frame)).T
                mag_mat_lin = out_mag.reshape((frames, bins_per_frame)).T  # raw magnitudes (bins, frames)
            except Exception as ex:
                print(f"ERROR: unable to reshape channels to (frames={frames}, bins={bins_per_frame}): {ex}")
                lib.fft_free(ctx)
                return

            from PIL.PngImagePlugin import PngInfo
            orient = "bins_top"  # rows=frequency bins from 0..Nyquist at image top
            if args.output == "rgb":
                # Real/imag linear normalization; magnitude optionally dB
                # Capture channel min/max so RGB can be inverted to linear complex for baselines.
                real_min = float(np.min(out_real)) if out_real.size else 0.0
                real_max = float(np.max(out_real)) if out_real.size else 0.0
                imag_min = float(np.min(out_imag)) if out_imag.size else 0.0
                imag_max = float(np.max(out_imag)) if out_imag.size else 0.0
                mag_mat_db_u8 = _mag_viz_u8(mag_mat_lin)
                img2d = np.stack([real_mat_full, imag_mat_full, mag_mat_db_u8], axis=2)
                meta = PngInfo()
                meta.add_text("FFTFREE_FMT", "rgb")
                meta.add_text("FFTFREE_CHANNELS", "R=real,G=imag,B=mag")
                meta.add_text("FFTFREE_ORIENT", orient)
                meta.add_text("FFTFREE_SCALE", str(args.scale))
                meta.add_text("FFTFREE_DB_REF", str(args.db_ref))
                meta.add_text("FFTFREE_DB_FLOOR", str(float(args.db_floor)))
                meta.add_text("FFTFREE_REAL_MIN", str(real_min))
                meta.add_text("FFTFREE_REAL_MAX", str(real_max))
                meta.add_text("FFTFREE_IMAG_MIN", str(imag_min))
                meta.add_text("FFTFREE_IMAG_MAX", str(imag_max))
                # If using global dB ref, store the reference magnitude used for mapping for potential inversion.
                if args.scale == "db" and args.db_ref == "global":
                    try:
                        meta.add_text("FFTFREE_DB_GLOBAL_REF", str(float(np.max(mag_mat_lin))))
                        meta.add_text("FFTFREE_INVERTIBLE", "1")
                    except Exception:
                        meta.add_text("FFTFREE_INVERTIBLE", "0")
                else:
                    # Per-frame dB or log1p are not strictly invertible from image alone
                    meta.add_text("FFTFREE_INVERTIBLE", "0")
                Image.fromarray(img2d, "RGB").save(args.output, pnginfo=meta)
                print(f"Saved FFT image to {args.output} using {lib_path} with shape {img2d.shape} (mag-scale={args.scale}, ref={args.db_ref}, floor={args.db_floor} dB)")
            else:
                # Magnitude-only grayscale. Prefer dB mapping by default.
                mag_u8 = _mag_viz_u8(mag_mat_lin)
                meta = PngInfo()
                meta.add_text("FFTFREE_FMT", "mag")
                meta.add_text("FFTFREE_CHANNELS", "L=mag")
                meta.add_text("FFTFREE_ORIENT", orient)
                meta.add_text("FFTFREE_SCALE", str(args.scale))
                meta.add_text("FFTFREE_DB_REF", str(args.db_ref))
                meta.add_text("FFTFREE_DB_FLOOR", str(float(args.db_floor)))
                if args.scale == "linear":
                    # Store linear normalization min/max for invertibility
                    mag_min = float(np.min(out_mag)) if out_mag.size else 0.0
                    mag_max = float(np.max(out_mag)) if out_mag.size else 0.0
                    meta.add_text("FFTFREE_MAG_MIN", str(mag_min))
                    meta.add_text("FFTFREE_MAG_MAX", str(mag_max))
                    meta.add_text("FFTFREE_INVERTIBLE", "1")
                elif args.scale == "db" and args.db_ref == "global":
                    try:
                        meta.add_text("FFTFREE_DB_GLOBAL_REF", str(float(np.max(mag_mat_lin))))
                        meta.add_text("FFTFREE_INVERTIBLE", "1")
                    except Exception:
                        meta.add_text("FFTFREE_INVERTIBLE", "0")
                else:
                    meta.add_text("FFTFREE_INVERTIBLE", "0")
                Image.fromarray(mag_u8, "L").save(args.output, pnginfo=meta)
                print(f"Saved magnitude image to {args.output} using {lib_path} with shape {mag_u8.shape} (mag-scale={args.scale}, ref={args.db_ref}, floor={args.db_floor} dB)")
        else:
            print("Image generation disabled (--no-image)")
    except Exception as e:
        print("ERROR:", e)
        traceback.print_exc()

if __name__ == "__main__":
    main()
