# CMakeLists.txt
cmake_minimum_required(VERSION 3.14)  # FetchContent requires 3.11+, but 3.14 for better support
project(fftfree)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(FFTFREE_NATIVE "Enable explicit SIMD tuning (use FFTFREE_NATIVE_ARCH to pick level)" ON)
set(FFTFREE_NATIVE_ARCH "AVX2" CACHE STRING "SIMD level when FFTFREE_NATIVE=ON (choices: SSE2, AVX, AVX2)")
set_property(CACHE FFTFREE_NATIVE_ARCH PROPERTY STRINGS "SSE2" "AVX" "AVX2")
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

include(CTest)
enable_testing()

if (FFTFREE_OPENMP)
  find_package(OpenMP REQUIRED)
endif()

# Add executable
add_executable(fft_example main.cpp)

# Link libraries
if (FFTFREE_OPENMP)
  target_link_libraries(fft_example OpenMP::OpenMP_CXX)
  target_compile_definitions(fft_example PRIVATE EIGFFT_USE_OMP=1)
endif()

set(_fftf_native_flag "")

if (MSVC)
  if (FFTFREE_NATIVE)
    if (FFTFREE_NATIVE_ARCH STREQUAL "AVX2")
      set(_fftf_native_flag "/arch:AVX2")
    elseif (FFTFREE_NATIVE_ARCH STREQUAL "AVX")
      set(_fftf_native_flag "/arch:AVX")
    elseif (FFTFREE_NATIVE_ARCH STREQUAL "SSE2")
      set(_fftf_native_flag "")
    else()
      message(FATAL_ERROR "Unsupported FFTFREE_NATIVE_ARCH='${FFTFREE_NATIVE_ARCH}'. Choose SSE2, AVX or AVX2.")
    endif()
    if (_fftf_native_flag)
      target_compile_options(fft_example PRIVATE ${_fftf_native_flag})
    endif()
  endif()
  if (FFTFREE_FAST_MATH)
    target_compile_options(fft_example PRIVATE /fp:fast)
  endif()
else()
  if (FFTFREE_NATIVE)
    if (FFTFREE_NATIVE_ARCH STREQUAL "AVX2")
      target_compile_options(fft_example PRIVATE -mavx2)
    elseif (FFTFREE_NATIVE_ARCH STREQUAL "AVX")
      target_compile_options(fft_example PRIVATE -mavx)
    elseif (FFTFREE_NATIVE_ARCH STREQUAL "SSE2")
      target_compile_options(fft_example PRIVATE -msse2)
    else()
      message(FATAL_ERROR "Unsupported FFTFREE_NATIVE_ARCH='${FFTFREE_NATIVE_ARCH}'. Choose SSE2, AVX or AVX2.")
    endif()
  endif()
  if (FFTFREE_FAST_MATH)
    target_compile_options(fft_example PRIVATE -ffast-math)
  endif()
  target_compile_options(fft_example PRIVATE -fno-math-errno)
endif()

target_link_libraries(fft_example Eigen3::Eigen)

add_executable(fft_tests tests/fftfree_tests.cpp)
target_link_libraries(fft_tests PRIVATE Eigen3::Eigen)

if (FFTFREE_OPENMP)
  target_link_libraries(fft_tests PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(fft_tests PRIVATE EIGFFT_USE_OMP=1)
endif()

if (MSVC)
  if (FFTFREE_NATIVE)
    if (_fftf_native_flag)
      target_compile_options(fft_tests PRIVATE ${_fftf_native_flag})
    endif()
  endif()
  if (FFTFREE_FAST_MATH)
    target_compile_options(fft_tests PRIVATE /fp:fast)
  endif()
else()
  if (FFTFREE_NATIVE)
    if (FFTFREE_NATIVE_ARCH STREQUAL "AVX2")
      target_compile_options(fft_tests PRIVATE -mavx2)
    elseif (FFTFREE_NATIVE_ARCH STREQUAL "AVX")
      target_compile_options(fft_tests PRIVATE -mavx)
    elseif (FFTFREE_NATIVE_ARCH STREQUAL "SSE2")
      target_compile_options(fft_tests PRIVATE -msse2)
    endif()
  endif()
  if (FFTFREE_FAST_MATH)
    target_compile_options(fft_tests PRIVATE -ffast-math)
  endif()
  target_compile_options(fft_tests PRIVATE -fno-math-errno)
endif()

add_test(NAME fftfree.tests COMMAND fft_tests)
