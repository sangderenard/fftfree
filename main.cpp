#include "eigen_fft.hpp"

#include <Eigen/Core>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

int main() {
  std::cout << "fftfree micro-benchmark" << std::endl;
  try {
    // Disable Eigen's own parallelism so the plan controls threading.
    Eigen::setNbThreads(1);

    struct BenchmarkCase {
      int N;
      int B;
      int repeats;
    };

    const std::vector<BenchmarkCase> cases = {
        {256, 128, 12},
        {1024, 64, 8},
        {2048, 96, 6},
        {4096, 32, 4},
    };

    std::mt19937 rng(1337);
    std::normal_distribution<double> dist(0.0, 1.0);

    auto fill_random = [&](Eigen::MatrixXcd& mat) {
      for (int i = 0; i < mat.rows(); ++i) {
        for (int j = 0; j < mat.cols(); ++j) {
          mat(i, j) = {dist(rng), dist(rng)};
        }
      }
    };

    auto run_plan = [](const Eigen::MatrixXcd& seed, eigfft::Plan<double>& plan, int repeats) {
      double accum = 0.0;
      Eigen::MatrixXcd work(seed.rows(), seed.cols());
      for (int r = 0; r < repeats; ++r) {
        work = seed;
        const auto start = std::chrono::high_resolution_clock::now();
        eigfft::fft_inplace_batched<double>(work, plan);
        const auto end = std::chrono::high_resolution_clock::now();
        accum += std::chrono::duration<double>(end - start).count();
      }
      return accum / static_cast<double>(repeats);
    };

    std::cout << std::fixed << std::setprecision(6);
    for (const auto& task : cases) {
      std::cout << "\n--- Benchmark N=" << task.N << " B=" << task.B
                << " repeats=" << task.repeats << " ---" << std::endl;

      Eigen::MatrixXcd seed(task.N, task.B);
      fill_random(seed);

      eigfft::Plan<double> scalar_plan(task.N, /*inverse=*/false, /*threads=*/false);
      scalar_plan.tuning.packet_step = 1;  // disable SIMD batching for baseline
      scalar_plan.tuning.force_ftz_daz = false;

      eigfft::Plan<double> simd_plan(task.N, /*inverse=*/false, /*threads=*/true);
      simd_plan.tuning.packet_step = 0;  // auto => packet_cols
      simd_plan.tuning.parallel_dim = eigfft::Plan<double>::ParallelDim::Columns;
      simd_plan.tuning.min_work_per_thread = 32;
      simd_plan.tuning.force_ftz_daz = false;

      const double scalar_time = run_plan(seed, scalar_plan, task.repeats);
      const double simd_time = run_plan(seed, simd_plan, task.repeats);

      std::cout << "  scalar (threads=1, lanes=1)  : " << scalar_time << " s" << std::endl;
      std::cout << "  simd+threads (lanes=" << simd_plan.packet_cols
                << ", max threads=" << simd_plan.effective_threads(task.B)
                << ") : " << simd_time << " s" << std::endl;
      if (simd_time < scalar_time) {
        std::cout << "    speedup: " << (scalar_time / simd_time) << "x" << std::endl;
      } else {
        std::cout << "    slowdown: " << (simd_time / scalar_time) << "x" << std::endl;
      }
    }

    std::cout << "\nHand-rolled Cooley-Tukey kernel uses Eigen packets (packet_cols="
              << eigfft::Plan<double>(256).packet_cols
              << ") and OpenMP for batched columns." << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return 1;
  }
}

