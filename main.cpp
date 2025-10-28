#define _USE_MATH_DEFINES
#include <cmath>
#include "eigen_fft.hpp"
#include <Eigen/Core>
#include <iostream>
#include <chrono>

int main() {
  try {
    Eigen::setNbThreads(1);  // Disable Eigen's internal threading
    const int N = 1024, B = 64;
    Eigen::MatrixXcd X(N, B); // fill with time-domain complex data
    // For demo, fill with some data
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < B; ++j) {
        X(i, j) = std::complex<double>(std::sin(2 * M_PI * i / N), std::cos(2 * M_PI * i / N));
      }
    }

    eigfft::Plan<double> plan(N, /*inverse=*/false, /*threads=*/false);
    plan.tuning.parallel_dim = eigfft::Plan<double>::ParallelDim::Auto; // or Columns/KBlocks
    plan.tuning.packet_step  = 0;    // 0 => auto (=packet size). Try 2× for huge B.
    plan.tuning.schedule     = eigfft::Plan<double>::Schedule::Auto;
    plan.tuning.min_work_per_thread = 64;
    std::cout << "Starting FFT" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    eigfft::fft_inplace_batched<double>(X, plan); // now frequency-domain
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "FFT completed in " << elapsed.count() << " seconds." << std::endl;

    // For inverse
    // eigfft::Plan<double> iplan(N, /*inverse=*/true, /*threads=*/true);
    // eigfft::fft_inplace_batched<double>(X, iplan); // back to time-domain

    // std::cout << "Inverse FFT completed." << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return 1;
  }
}