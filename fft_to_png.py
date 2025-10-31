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
void fft_pcm_to_channels(const float* in_pcm, float* out_real, float* out_imag, float* out_mag, size_t n);
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
    search_dirs = [project_root]
    if build_root.exists():
        search_dirs.append(build_root)

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
    parser.add_argument("output", help="Destination PNG file")
    parser.add_argument(
        "--lib",
        help="Explicit path to the fft_cffi shared library (overrides FFTFREE_CFFI_LIB)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    lib_path = _discover_library(args.lib)
    lib = ffi.dlopen(str(lib_path))

    # Read audio file
    _sample_rate, data = wavfile.read(args.input)
    if data.ndim > 1:
        data = data[:, 0]  # Use first channel if stereo
    data = np.ascontiguousarray(data, dtype=np.float32)
    n = data.shape[0]

    # Allocate output buffers
    out_real = np.zeros(n, dtype=np.float32)
    out_imag = np.zeros(n, dtype=np.float32)
    out_mag = np.zeros(n, dtype=np.float32)

    # Call C++ FFT via cffi
    lib.fft_pcm_to_channels(
        ffi.cast("float *", data.ctypes.data),
        ffi.cast("float *", out_real.ctypes.data),
        ffi.cast("float *", out_imag.ctypes.data),
        ffi.cast("float *", out_mag.ctypes.data),
        n
    )

    # Normalize to 0-255
    def norm(arr):
        arr = arr.astype(np.float32)
        arr -= arr.min()
        max_val = arr.max()
        if max_val == 0:
            return np.zeros_like(arr, dtype=np.uint8)
        arr /= (max_val + 1e-8)
        arr *= 255
        return arr.astype(np.uint8)

    real_img = norm(out_real)
    imag_img = norm(out_imag)
    mag_img = norm(out_mag)

    # Stack into RGB image
    length = len(real_img)
    img = np.stack([real_img, imag_img, mag_img], axis=1)
    side = int(np.ceil(np.sqrt(length)))
    padded = np.zeros((side*side, 3), dtype=np.uint8)
    padded[:length] = img
    img2d = padded.reshape((side, side, 3))

    Image.fromarray(img2d, "RGB").save(args.output)
    print(f"Saved FFT image to {args.output} using {lib_path}")

if __name__ == "__main__":
    main()
