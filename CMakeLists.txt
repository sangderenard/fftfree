# Maximum optimization flags for Release configuration
if (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  if (MSVC)
    set(_fftfree_native_flag "")
    if (FFTFREE_NATIVE)
      if (FFTFREE_NATIVE_ARCH STREQUAL "AVX2")
        set(_fftfree_native_flag "/arch:AVX2")
      elseif (FFTFREE_NATIVE_ARCH STREQUAL "AVX")
        set(_fftfree_native_flag "/arch:AVX")
      elseif (FFTFREE_NATIVE_ARCH STREQUAL "SSE2")
        set(_fftfree_native_flag "/arch:SSE2")
      endif()
    endif()
    add_compile_options(/O2 /fp:fast ${_fftfree_native_flag})
  else()
    add_compile_options(-O3 -ffast-math -march=native)
  endif()
endif()
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
option(EIGFFT_TIMING "Enable timing instrumentation (compile-time)" OFF)
option(EIGFFT_TRACE_THREADS "Enable detailed thread/worker pool tracing" OFF)

# Detect the highest SIMD level actually supported by the host CPU when AUTO is
# requested on MSVC.  Using compiler-flag probes alone is insufficient because
# MSVC accepts "/arch:AVX2" even on CPUs that do not support AVX2, which leads
# to illegal-instruction crashes at runtime.
set(FFTFREE_MSVC_AUTO_ARCH_FLAG "")
if (MSVC AND FFTFREE_NATIVE AND FFTFREE_NATIVE_ARCH STREQUAL "AUTO")
  set(_fftfree_detect_src "${CMAKE_CURRENT_LIST_DIR}/cmake/detect_msvc_arch.cpp")
  if (EXISTS "${_fftfree_detect_src}")
    if (CMAKE_CROSSCOMPILING)
      message(WARNING "Host processor feature detection is unavailable while cross-compiling; falling back to default arch selection.")
    else()
      try_run(
        _fftfree_arch_run_result
        _fftfree_arch_compile_result
        "${CMAKE_BINARY_DIR}/CMakeFiles"
        "${_fftfree_detect_src}"
        CMAKE_FLAGS
          "-DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}"
          "-DCMAKE_CXX_STANDARD_REQUIRED=${CMAKE_CXX_STANDARD_REQUIRED}"
        RUN_OUTPUT_VARIABLE _fftfree_arch_output
        COMPILE_OUTPUT_VARIABLE _fftfree_arch_compile_output
      )

      if (_fftfree_arch_compile_result EQUAL 0 AND "${_fftfree_arch_run_result}" STREQUAL "0")
        string(STRIP "${_fftfree_arch_output}" _fftfree_arch_output)
        if (_fftfree_arch_output STREQUAL "AVX2")
          set(FFTFREE_MSVC_AUTO_ARCH_FLAG "/arch:AVX2")
        elseif (_fftfree_arch_output STREQUAL "AVX")
          set(FFTFREE_MSVC_AUTO_ARCH_FLAG "/arch:AVX")
        endif()
      else()
        set(_fftfree_arch_run_output "${_fftfree_arch_output}")
        string(STRIP "${_fftfree_arch_run_output}" _fftfree_arch_run_output)
        message(WARNING "Host processor feature detection failed; falling back to default arch selection.\nCompile output: ${_fftfree_arch_compile_output}\nRun output: ${_fftfree_arch_run_output}")
      endif()

      unset(_fftfree_arch_output)
      unset(_fftfree_arch_run_output)
      unset(_fftfree_arch_compile_output)
      unset(_fftfree_arch_compile_result)
      unset(_fftfree_arch_run_result)
    endif()
  else()
    message(WARNING "Missing MSVC SIMD detector source '${_fftfree_detect_src}'.")
  endif()
  unset(_fftfree_detect_src)
endif()


# Fetch Eigen
include(FetchContent)
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(eigen)

# Enable LTO for Release builds
if (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  include(CheckIPOSupported)
  check_ipo_supported(RESULT lto_supported OUTPUT lto_output)
  if(lto_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    message(STATUS "LTO enabled for Release build.")
  else()
    message(WARNING "LTO not supported: ${lto_output}")
  endif()
endif()

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

  if (EIGFFT_TIMING)
    # Propagate compile-time timing flag into targets so the header-guarded
    # EIGFFT_TIMING macro can be enabled from CMake with -DEIGFFT_TIMING=ON
    target_compile_definitions(${target} PRIVATE EIGFFT_TIMING=1)
  endif()

  if (EIGFFT_TRACE_THREADS)
    target_compile_definitions(${target} PRIVATE EIGFFT_TRACE_THREADS=1)
  endif()

  if (MSVC)
    set(_fftf_native_flag "")
    set(_fftf_native_flag_message "default (compiler-defined)")
    if (FFTFREE_NATIVE)
      if (CMAKE_SIZEOF_VOID_P EQUAL 4)
        set(_fftf_native_is_win32 TRUE)
      else()
        set(_fftf_native_is_win32 FALSE)
      endif()

      if (FFTFREE_NATIVE_ARCH STREQUAL "AUTO")
        if (FFTFREE_MSVC_AUTO_ARCH_FLAG)
          set(_fftf_native_flag ${FFTFREE_MSVC_AUTO_ARCH_FLAG})
          set(_fftf_native_flag_message "${FFTFREE_MSVC_AUTO_ARCH_FLAG}")
        else()
          if (_fftf_native_is_win32)
            set(_fftf_native_flag "/arch:SSE2")
            set(_fftf_native_flag_message "/arch:SSE2 (Win32 fallback)")
          else()
            set(_fftf_native_flag "")
            set(_fftf_native_flag_message "default (x64 baseline SSE2)")
          endif()
        endif()
      elseif (FFTFREE_NATIVE_ARCH STREQUAL "AVX2")
        set(_fftf_native_flag "/arch:AVX2")
        set(_fftf_native_flag_message "/arch:AVX2")
      elseif (FFTFREE_NATIVE_ARCH STREQUAL "AVX")
        set(_fftf_native_flag "/arch:AVX")
        set(_fftf_native_flag_message "/arch:AVX")
      elseif (FFTFREE_NATIVE_ARCH STREQUAL "SSE2")
        if (_fftf_native_is_win32)
          set(_fftf_native_flag "/arch:SSE2")
          set(_fftf_native_flag_message "/arch:SSE2 (Win32 explicit)")
        else()
          set(_fftf_native_flag "")
          set(_fftf_native_flag_message "default (x64 baseline SSE2)")
        endif()
      else()
        message(FATAL_ERROR "Unsupported FFTFREE_NATIVE_ARCH='${FFTFREE_NATIVE_ARCH}'. Choose AUTO, SSE2, AVX or AVX2.")
      endif()

      message(STATUS "SIMD arch for ${target}: ${_fftf_native_flag_message} [flag='${_fftf_native_flag}'] (FFTFREE_NATIVE_ARCH=${FFTFREE_NATIVE_ARCH})")
    else()
      message(STATUS "SIMD arch for ${target}: FFTFREE_NATIVE=OFF (compiler default)")
    endif()

    if (_fftf_native_flag)
      target_compile_options(${target} PRIVATE ${_fftf_native_flag})
    endif()
    if (FFTFREE_FAST_MATH)
      target_compile_options(${target} PRIVATE /fp:fast)
    endif()
  # Workarounds for MSVC Debug build crashes observed on some toolsets:
  # - enable /FS to reduce PDB write races
  # - prefer /Z7 (embedded debug info) in Debug to avoid PDB generation bugs
  #   that can trigger internal compiler failures when writing complex debug
  #   information.
  target_compile_options(${target} PRIVATE $<$<CONFIG:Debug>:/FS>)
  target_compile_options(${target} PRIVATE $<$<CONFIG:Debug>:/Z7>)
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


add_library(fft_cffi SHARED fft_cffi.cpp eigen_fft.hpp plan_support.hpp butterfly/butterfly_base.hpp butterfly/butterfly_factory.hpp butterfly/butterfly_radix2_dit.hpp butterfly/butterfly_radix2_dif.hpp butterfly/butterfly_radix4_dit.hpp)
fftfree_apply_target_options(fft_cffi)
target_compile_definitions(fft_cffi PRIVATE FFT_CFFI_EXPORTS)

add_executable(fft_example main.cpp)
fftfree_apply_target_options(fft_example)

add_executable(fft_tests tests/fftfree_tests.cpp)
fftfree_apply_target_options(fft_tests)

add_test(NAME fftfree.tests COMMAND fft_tests)

# Thread tracing test application exercising worker pool dispatch.
add_executable(fft_threads_trace tests/threads_trace_test.cpp)
fftfree_apply_target_options(fft_threads_trace)
add_test(NAME fftfree.threads_trace COMMAND fft_threads_trace)

# Minimal restore/retry unit test
add_executable(fft_restore_test tests/restore_test.cpp)
fftfree_apply_target_options(fft_restore_test)
add_test(NAME fftfree.restore_test COMMAND fft_restore_test)
