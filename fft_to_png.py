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
               int pad_mode);
int fft_execute(void* handle,
                const float* in_pcm,
                float* out_real,
                float* out_imag,
                float* out_mag,
                size_t n);
size_t fft_ctx_size(void* handle);
void fft_free(void* handle);
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

        # Use the simplified deployment API: init -> execute -> free
        ctx = lib.fft_init(n, args.threads, args.lanes, 0, kernel_code, pad_code)
        if ctx == ffi.NULL:
            print("ERROR: fft_init failed (ensure N is power-of-two and library is built correctly)")
            return
        plan_n = lib.fft_ctx_size(ctx)
        if plan_n == 0:
            print("ERROR: fft_ctx_size returned 0")
            lib.fft_free(ctx)
            return
        if plan_n != n and args.pad != "always":
            print(f"Padding (library): extended {n} -> {plan_n} samples to satisfy kernel constraints.")
        n_eff = int(plan_n)

        # Allocate output buffers sized to plan
        out_real = np.zeros(n_eff, dtype=np.float32)
        out_imag = np.zeros(n_eff, dtype=np.float32)
        out_mag = np.zeros(n_eff, dtype=np.float32)
        try:
            ok = lib.fft_execute(
                ctx,
                ffi.cast("float *", data.ctypes.data),
                ffi.cast("float *", out_real.ctypes.data),
                ffi.cast("float *", out_imag.ctypes.data),
                ffi.cast("float *", out_mag.ctypes.data),
                orig_n if args.pad != "always" else n,
            )
            if ok == 0:
                print("ERROR: fft_execute failed")
                return
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

            # Stack into RGB image
            length = len(real_img)
            img = np.stack([real_img, imag_img, mag_img], axis=1)
            side = int(np.ceil(np.sqrt(length)))
            padded = np.zeros((side * side, 3), dtype=np.uint8)
            padded[:length] = img
            img2d = padded.reshape((side, side, 3))

            Image.fromarray(img2d, "RGB").save(args.output)
            print(f"Saved FFT image to {args.output} using {lib_path}")
        else:
            print("Image generation disabled (--no-image)")
    except Exception as e:
        print("ERROR:", e)
        traceback.print_exc()

if __name__ == "__main__":
    main()
