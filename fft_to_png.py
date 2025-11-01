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
                    int inner_threads);
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
        init_full = getattr(lib, "fft_init_full", None)
        if init_full is not None:
            # Parameters: n, threads, lanes, inverse, kernel, radix, radix_pattern, radix_pattern_len,
            # pad_mode, window, hop, stft_mode,
            # transform, reduce_magnitude, store_polar, half_spectrum,
            # allow_outer_parallel, allow_inner_parallel, inner_threads
            ctx = init_full(
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
                0,              # transform (C2C)
                0,              # reduce_magnitude
                0,              # store_polar
                0,              # half_spectrum
                1,              # allow_outer_parallel -> create WorkerPool
                0,              # allow_inner_parallel
                0               # inner_threads
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
        # allocate flattened output buffers: frames * N
        out_count = frames * N_eff
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
            )
            if produced == 0:
                print("ERROR: fft_execute_batched failed or produced 0 frames")
                return
            frames = int(produced)
        finally:
            lib.fft_free(ctx)

        # Normalize helpers used by both the PNG writer and stats reporting.
        def _normalize_for_image(arr: np.ndarray) -> np.ndarray:
            """Return a float32 array scaled to the 0-255 visualization range."""

            scaled = np.array(arr, dtype=np.float32, copy=True)
            if scaled.size == 0:
                return scaled

            min_val = float(np.min(scaled))
            scaled -= min_val
            max_val = float(np.max(scaled))
            if max_val == 0.0:
                return np.zeros_like(scaled, dtype=np.float32)

            scaled /= (max_val + 1e-8)
            scaled *= 255.0
            return scaled

        def norm(arr: np.ndarray) -> np.ndarray:
            return _normalize_for_image(arr).astype(np.uint8)

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

            def describe_normalized(name: str, arr: np.ndarray) -> None:
                normalized = _normalize_for_image(arr)
                describe(name, normalized)

            describe_normalized("real_norm", out_real)
            describe_normalized("imag_norm", out_imag)
            describe_normalized("mag_norm", out_mag)

        if not args.no_image:
            real_img = norm(out_real)
            imag_img = norm(out_imag)
            mag_img = norm(out_mag)

            # Try to reshape into (frames, N) then transpose to (N, frames)
            # so the vertical axis represents frequency bins (N=1024).
            try:
                real_mat = real_img.reshape((frames, N_eff)).T
                imag_mat = imag_img.reshape((frames, N_eff)).T
                mag_mat = mag_img.reshape((frames, N_eff)).T
            except Exception as ex:
                # No fallbacks: fail loudly so caller can fix parameters/output sizing.
                print(f"ERROR: unable to reshape STFT outputs to (frames={frames}, N={N_eff}): {ex}")
                lib.fft_free(ctx)
                return

            # Stack into RGB image with shape (N, frames, 3)
            img2d = np.stack([real_mat, imag_mat, mag_mat], axis=2)
            Image.fromarray(img2d, "RGB").save(args.output)
            print(f"Saved FFT image to {args.output} using {lib_path} with shape {img2d.shape}")
        else:
            print("Image generation disabled (--no-image)")
    except Exception as e:
        print("ERROR:", e)
        traceback.print_exc()

if __name__ == "__main__":
    main()
