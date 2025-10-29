# CMakeLists.txt
cmake_minimum_required(VERSION 3.14)  # FetchContent requires 3.11+, but 3.14 for better support
project(fftfree)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(FFTFREE_NATIVE "Enable explicit SIMD tuning (use FFTFREE_NATIVE_ARCH to pick level)" ON)
option(FFTFREE_NATIVE_ARCH_PIN "Keep a manually selected FFTFREE_NATIVE_ARCH value, even if it matches the legacy AVX2 default" OFF)
set(FFTFREE_NATIVE_ARCH "AUTO" CACHE STRING "SIMD level when FFTFREE_NATIVE=ON (choices: AUTO, SSE2, AVX, AVX2)")
set_property(CACHE FFTFREE_NATIVE_ARCH PROPERTY STRINGS "AUTO" "SSE2" "AVX" "AVX2")

# Prior revisions defaulted to AVX2 on MSVC, which breaks deployments on CPUs lacking AVX2.
# Upgrades would keep that cached default, so reset it to AUTO unless the user explicitly pins it.
if (NOT FFTFREE_NATIVE_ARCH_PIN)
  string(TOUPPER "${FFTFREE_NATIVE_ARCH}" _fftfree_native_arch_upper)
  if (_fftfree_native_arch_upper STREQUAL "AVX2")
    message(STATUS "Resetting FFTFREE_NATIVE_ARCH from legacy default AVX2 to AUTO. Set FFTFREE_NATIVE_ARCH_PIN=ON to keep AVX2.")
    set(FFTFREE_NATIVE_ARCH "AUTO" CACHE STRING "SIMD level when FFTFREE_NATIVE=ON (choices: AUTO, SSE2, AVX, AVX2)" FORCE)
  endif()
  unset(_fftfree_native_arch_upper)
endif()
option(FFTFREE_OPENMP "Enable OpenMP" ON)
option(FFTFREE_FAST_MATH "Enable fast-math" ON)

# Detect the highest SIMD level supported by MSVC when AUTO is requested.
set(FFTFREE_MSVC_AUTO_ARCH_FLAG "")
if (MSVC AND FFTFREE_NATIVE AND FFTFREE_NATIVE_ARCH STREQUAL "AUTO")
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("/arch:AVX2" FFTFREE_MSVC_HAS_AVX2)
  if (FFTFREE_MSVC_HAS_AVX2)
    set(FFTFREE_MSVC_AUTO_ARCH_FLAG "/arch:AVX2")
  else()
    check_cxx_compiler_flag("/arch:AVX" FFTFREE_MSVC_HAS_AVX)
    if (FFTFREE_MSVC_HAS_AVX)
      set(FFTFREE_MSVC_AUTO_ARCH_FLAG "/arch:AVX")
    endif()
  endif()
endif()

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
function(fftfree_apply_target_options target)
  target_link_libraries(${target} PRIVATE Eigen3::Eigen)

  if (FFTFREE_OPENMP)
    target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(${target} PRIVATE EIGFFT_USE_OMP=1)
  endif()

  if (MSVC)
    set(_fftf_native_flag "")
    if (FFTFREE_NATIVE)
      if (FFTFREE_NATIVE_ARCH STREQUAL "AUTO")
        if (FFTFREE_MSVC_AUTO_ARCH_FLAG)
          set(_fftf_native_flag ${FFTFREE_MSVC_AUTO_ARCH_FLAG})
        else()
          # MSVC defaults to the highest ISA supported by the target platform
          set(_fftf_native_flag "")
        endif()
      elseif (FFTFREE_NATIVE_ARCH STREQUAL "AVX2")
        set(_fftf_native_flag "/arch:AVX2")
      elseif (FFTFREE_NATIVE_ARCH STREQUAL "AVX")
        set(_fftf_native_flag "/arch:AVX")
      elseif (FFTFREE_NATIVE_ARCH STREQUAL "SSE2")
        set(_fftf_native_flag "")
      else()
        message(FATAL_ERROR "Unsupported FFTFREE_NATIVE_ARCH='${FFTFREE_NATIVE_ARCH}'. Choose AUTO, SSE2, AVX or AVX2.")
      endif()
    endif()
    if (_fftf_native_flag)
      target_compile_options(${target} PRIVATE ${_fftf_native_flag})
    endif()
    if (FFTFREE_FAST_MATH)
      target_compile_options(${target} PRIVATE /fp:fast)
    endif()
  else()
    if (FFTFREE_NATIVE)
      if (FFTFREE_NATIVE_ARCH STREQUAL "AUTO")
        target_compile_options(${target} PRIVATE -march=native)
      elseif (FFTFREE_NATIVE_ARCH STREQUAL "AVX2")
        target_compile_options(${target} PRIVATE -mavx2)
      elseif (FFTFREE_NATIVE_ARCH STREQUAL "AVX")
        target_compile_options(${target} PRIVATE -mavx)
      elseif (FFTFREE_NATIVE_ARCH STREQUAL "SSE2")
        target_compile_options(${target} PRIVATE -msse2)
      else()
        message(FATAL_ERROR "Unsupported FFTFREE_NATIVE_ARCH='${FFTFREE_NATIVE_ARCH}'. Choose AUTO, SSE2, AVX or AVX2.")
      endif()
    endif()
    if (FFTFREE_FAST_MATH)
      target_compile_options(${target} PRIVATE -ffast-math)
    endif()
    target_compile_options(${target} PRIVATE -fno-math-errno)
  endif()
endfunction()

add_executable(fft_example main.cpp)
fftfree_apply_target_options(fft_example)

add_executable(fft_tests tests/fftfree_tests.cpp)
fftfree_apply_target_options(fft_tests)

add_test(NAME fftfree.tests COMMAND fft_tests)
