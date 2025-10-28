# CMakeLists.txt
cmake_minimum_required(VERSION 3.14)  # FetchContent requires 3.11+, but 3.14 for better support
project(fftfree)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(FFTFREE_NATIVE "Enable -march=native (or /arch:AVX2 on MSVC)" ON)
option(FFTFREE_OPENMP "Enable OpenMP" ON)
option(FFTFREE_FAST_MATH "Enable fast-math" ON)

# Fetch Eigen
include(FetchContent)
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(eigen)

if (FFTFREE_OPENMP)
  find_package(OpenMP REQUIRED)
endif()

# Add executable
add_executable(fft_example main.cpp)

# Link libraries
target_link_libraries(fft_example Eigen3::Eigen)
if (FFTFREE_OPENMP)
  target_link_libraries(fft_example OpenMP::OpenMP_CXX)
  target_compile_definitions(fft_example PRIVATE EIGFFT_USE_OMP=1)
endif()

if (MSVC)
  if (FFTFREE_NATIVE)
    target_compile_options(fft_example PRIVATE /arch:AVX2)
  endif()
  if (FFTFREE_FAST_MATH)
    target_compile_options(fft_example PRIVATE /fp:fast)
  endif()
else()
  if (FFTFREE_NATIVE)
    target_compile_options(fft_example PRIVATE -march=native)
  endif()
  if (FFTFREE_FAST_MATH)
    target_compile_options(fft_example PRIVATE -ffast-math)
  endif()
  target_compile_options(fft_example PRIVATE -fno-math-errno)
endif()

# Link libraries
target_link_libraries(fft_example Eigen3::Eigen OpenMP::OpenMP_CXX)