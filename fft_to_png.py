import sys
import numpy as np
from scipy.io import wavfile
from PIL import Image
from cffi import FFI
import os

ffi = FFI()
ffi.cdef("""
void fft_pcm_to_channels(const float* in_pcm, float* out_real, float* out_imag, float* out_mag, size_t n);
""")

# Load the shared library (update name as needed)
if os.name == "nt":
    libname = os.path.join(os.path.dirname(__file__), "build", "Release", "fft_cffi.dll")
else:
    libname = os.path.join(os.path.dirname(__file__), "build", "Release", "libfft_cffi.so")
lib = ffi.dlopen(libname)

def main():
    if len(sys.argv) != 3:
        print("Usage: python fft_to_png.py input.wav output.png")
        sys.exit(1)
    audio_path = sys.argv[1]
    output_path = sys.argv[2]

    # Read audio file
    sample_rate, data = wavfile.read(audio_path)
    if data.ndim > 1:
        data = data[:, 0]  # Use first channel if stereo
    data = data.astype(np.float32)
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
        arr /= (arr.max() + 1e-8)
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

    Image.fromarray(img2d, 'RGB').save(output_path)
    print(f"Saved FFT image to {output_path}")

if __name__ == "__main__":
    main()
