# FFTFree

An Eigen-based multithreaded batched FFT kernel with tunable parameters.

## Features

- Header-only implementation using Eigen for vectorization
- Batched FFT processing (multiple signals at once)
- Multithreaded using OpenMP
- Power-of-two sizes (radix-2 Cooley-Tukey)
- In-place computation
- Tunable threading (enable/disable)

## Building

Requires CMake 3.14+, C++17 compiler with OpenMP support.

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

Eigen is automatically downloaded via FetchContent.

## Usage

```cpp
#include "eigen_fft.hpp"
#include <Eigen/Core>

int main() {
  const int N = 1024; // FFT size (power of 2)
  const int B = 64;   // Batch size (number of signals)

  Eigen::MatrixXcd X(N, B); // Complex matrix, rows=time/freq, cols=batch
  // Fill X with time-domain data...

  eigfft::Plan<double> plan(N, false, true); // N, inverse=false, threads=true
  eigfft::fft_inplace_batched<double>(X, plan); // Now frequency-domain

  // For inverse
  eigfft::Plan<double> iplan(N, true, true);
  eigfft::fft_inplace_batched<double>(X, iplan); // Back to time-domain
}
```

## Tuning

- `plan.tuning.parallel_dim`: `Auto` (heuristic), `Columns` (good for large B), `KBlocks` (good for small B)
- `plan.tuning.packet_step`: SIMD coarsening (0=auto, or multiple of packet size for wider batches)
- `plan.tuning.schedule`: `Auto`, `Static`, `Dynamic`, `Guided` (scheduling policy)
- `plan.tuning.min_work_per_thread`: Heuristic threshold for dynamic scheduling
- CMake options: `-DFFTFREE_NATIVE=ON`, `-DFFTFREE_OPENMP=ON`, `-DFFTFREE_FAST_MATH=ON`

## Performance Notes

- Use `-O3 -march=native -ffast-math` for compilation
- Enable Eigen vectorization
- For Windows, set FTZ/DAZ to avoid denormal slowdowns
- Batch size B should be chosen to fill SIMD lanes