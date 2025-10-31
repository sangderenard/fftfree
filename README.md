# FFTFree

An Eigen-based multithreaded batched FFT kernel with tunable parameters.

## Features

- Header-only implementation using Eigen for vectorization
- Batched FFT processing (multiple signals at once)
- Multithreaded using OpenMP
- Power-of-two sizes (radix-2 Cooley-Tukey)
- In-place computation
- Parallel-first design with optional debug-only sequential fallback
- Optional real-time probe to gauge sustainable streaming throughput

## Building

Requires CMake 3.14+, C++17 compiler with OpenMP support.

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

Eigen is automatically downloaded via FetchContent.

To run the self-tests:

```bash
ctest --output-on-failure
```

## Usage

```cpp
#include "eigen_fft.hpp"
#include "plan_support.hpp"
#include <Eigen/Core>

int main() {
  const int N = 1024; // FFT size (power of 2)
  const int B = 64;   // Batch size (number of signals)

  Eigen::MatrixXcd X(N, B); // Complex matrix, rows=time/freq, cols=batch
  // Fill X with time-domain data...

  eigfft::PlanRuntimeConfig cfg;   // defaults: 4 threads, 2 Stockham lanes
  eigfft::PlanEnvironment<double> forward_env;
  forward_env.initialize(N, /*inverse=*/false, cfg);
  eigfft::fft_inplace_batched<double>(X, forward_env.plan()); // Now frequency-domain

  eigfft::PlanEnvironment<double> inverse_env;
  inverse_env.initialize(N, /*inverse=*/true, cfg);
  eigfft::fft_inplace_batched<double>(X, inverse_env.plan()); // Back to time-domain
}
```

## Tuning

- `plan.tuning.parallel_dim`: `Auto` (heuristic), `Columns` (good for large B), `KBlocks` (good for small B)
- `plan.tuning.packet_step`: SIMD coarsening (0=auto, or multiple of packet size for wider batches)
- `plan.tuning.schedule`: `Auto`, `Static`, `Dynamic`, `Guided` (scheduling policy)
- `plan.tuning.min_work_per_thread`: Heuristic threshold for dynamic scheduling
- CMake options: `-DFFTFREE_NATIVE=ON`, `-DFFTFREE_OPENMP=ON`, `-DFFTFREE_FAST_MATH=ON`
- On MSVC you can override the SIMD level explicitly with
  `-DFFTFREE_NATIVE_ARCH=SSE2|AVX|AVX2` if auto-detection does not match your CPU

> ⚠️ Sequential execution is disabled by default. Define `EIGFFT_ALLOW_SEQUENTIAL` at
> configure time if you need to opt into the legacy single-thread fallback for
> debugging or comparison runs.

### Real-time probe

The demo binary can simulate a streaming workload after the micro-benchmarks to
measure sustained frame/sample rates. Enable it with `--realtime` and adjust the
parameters as needed:

```bash
fft_example --rt \
  --rt-sample-rate=48000 \
  --rt-window=2048 \
  --rt-stride=512 \
  --rt-duration=10 \
  --rt-delay-ms=5 \
  --rt-safety=0.85
```

The probe reports deadline overruns, worst-frame timings, and the maximum
sample/frame rates achievable while maintaining the configured safety margin.

## Performance Notes

- Use `-O3 -march=native -ffast-math` for compilation
- Enable Eigen vectorization
- For Windows, set FTZ/DAZ to avoid denormal slowdowns
- Batch size B should be chosen to fill SIMD lanes
