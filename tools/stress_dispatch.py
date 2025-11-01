"""
Stress-test harness for fftfree dispatching.

Modes:
  - Controller (default): spawn subprocesses for each configuration so crashes/segfaults are isolated.
  - Child: run a single configuration repeatedly and return non-zero on error.

This script uses the same C ABI as fft_to_png.py and prefers fft_init_full when available.
"""
from __future__ import annotations
import argparse
import os
import sys
import subprocess
import json
import random
import tempfile
from pathlib import Path
from typing import Optional, List
import time

from cffi import FFI
import numpy as np

ffi = FFI()
ffi.cdef(
    """
void* fft_init_ex(size_t n,
                  int threads,
                  int lanes,
                  int inverse,
                  int kernel,
                  int radix,
                  const int* radix_pattern,
                  size_t radix_pattern_len,
                  int pad_mode,
                  int window,
                  int hop,
                  int stft_mode);
void* fft_init_full(size_t n,
                    int threads,
                    int lanes,
                    int inverse,
                    int kernel,
                    int radix,
                    const int* radix_pattern,
                    size_t radix_pattern_len,
                    int pad_mode,
                    int window,
                    int hop,
                    int stft_mode,
                    int transform,
                    int reduce_magnitude,
                    int store_polar,
                    int half_spectrum,
                    int allow_outer_parallel,
                    int allow_inner_parallel,
                    int inner_threads);
size_t fft_execute_batched(void* handle,
                           const float* pcm,
                           size_t pcm_len,
                           float* out_real,
                           float* out_imag,
                           float* out_mag,
                           int pad_mode,
                           size_t max_frames);
size_t fft_ctx_worker_threads(void* handle);
size_t fft_ctx_effective_threads(void* handle);
void fft_free(void* handle);
size_t fft_ctx_size(void* handle);
"""
)


def candidate_library_names() -> List[str]:
    if os.name == "nt":
        return ["fft_cffi.dll"]
    elif sys.platform == "darwin":
        return ["libfft_cffi.dylib"]
    else:
        return ["libfft_cffi.so"]


def find_library(explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(str(p))
        return p
    project_root = Path(__file__).resolve().parent.parent
    build_root = project_root / "build"
    search_dirs = [project_root]
    if build_root.exists():
        search_dirs.append(build_root)
        release = build_root / "Release"
        if release.exists():
            search_dirs.append(release)
    for d in search_dirs:
        for name in candidate_library_names():
            cand = d / name
            if cand.exists():
                return cand
            if d.exists():
                match = next(d.rglob(name), None)
                if match:
                    return match
    raise FileNotFoundError("Could not find fft shared library; set --lib or FFTFREE_CFFI_LIB")


def child_run(lib_path: str, cfg: dict, iters: int, seed: int) -> int:
    lib = ffi.dlopen(lib_path)
    # map codes
    kernel_code = 0  # auto
    pad_code = 0
    N = cfg.get("N", 1024)
    W = cfg.get("W", N)
    H = cfg.get("H", W)
    threads = cfg.get("threads", 1)
    lanes = cfg.get("lanes", 0)
    allow_outer = 1 if cfg.get("allow_outer", True) else 0
    allow_inner = 1 if cfg.get("allow_inner", False) else 0
    inner_threads = cfg.get("inner_threads", 0)
    use_full = hasattr(lib, "fft_init_full")
    rng = np.random.RandomState(seed)
    # small PCM that produces many frames: produce pcm_len = W + H*frames
    frames_per_iter = cfg.get("frames", 16)
    pcm_len = W + H * (frames_per_iter - 1)

    for i in range(iters):
        # create random pcm
        pcm = (rng.randn(pcm_len).astype(np.float32) * 0.1).copy()
        if use_full:
            ctx = lib.fft_init_full(N, threads, lanes, 0, kernel_code, 0, ffi.NULL, 0, pad_code, W, H, 1, 0, 0, 0, 0, allow_outer, allow_inner, inner_threads)
        else:
            ctx = lib.fft_init_ex(N, threads, lanes, 0, kernel_code, 0, ffi.NULL, 0, pad_code, W, H, 1)
        if ctx == ffi.NULL:
            print("CHILD: init failed for cfg=", cfg, file=sys.stderr)
            return 2
        plan_n = int(lib.fft_ctx_size(ctx))
        out_count = frames_per_iter * plan_n
        out_real = np.zeros(out_count, dtype=np.float32)
        out_imag = np.zeros(out_count, dtype=np.float32)
        out_mag = np.zeros(out_count, dtype=np.float32)
        try:
            produced = lib.fft_execute_batched(ctx, ffi.cast("float *", pcm.ctypes.data), pcm_len, ffi.cast("float *", out_real.ctypes.data), ffi.cast("float *", out_imag.ctypes.data), ffi.cast("float *", out_mag.ctypes.data), pad_code, frames_per_iter)
            if produced == 0:
                print("CHILD: execute produced 0 frames", file=sys.stderr)
                lib.fft_free(ctx)
                return 3
        except Exception as e:
            print("CHILD: exception during execute:", e, file=sys.stderr)
            lib.fft_free(ctx)
            return 4
        lib.fft_free(ctx)
    return 0


def controller(lib_path: str, threads_list: List[int], inner_list: List[int], iters: int):
    configs = []
    for t in threads_list:
        # try outer parallel
        configs.append({"threads": t, "allow_outer": True, "allow_inner": False, "inner_threads": 0, "N": 1024, "W": 1024, "H": 512, "frames": 32})
        # try inner parallel only
        configs.append({"threads": 1, "allow_outer": False, "allow_inner": True, "inner_threads": max(1, t), "N": 1024, "W": 1024, "H": 512, "frames": 32})
    failures = []
    for idx, cfg in enumerate(configs):
        print(f"Controller: running config #{idx+1}/{len(configs)}: {cfg}")
        cmd = [sys.executable, str(Path(__file__).resolve()), "--child", "--lib", str(lib_path), "--cfg", json.dumps(cfg), "--iters", str(iters), "--seed", str(random.randrange(1<<30))]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = proc.communicate()
        print(out, end='')
        if err:
            print(err, file=sys.stderr)
        if proc.returncode != 0:
            print(f"Controller: child for config {idx} failed with exit {proc.returncode}")
            failures.append((cfg, proc.returncode, out, err))
        else:
            print(f"Controller: config #{idx} OK")
    if failures:
        print("Controller: completed with failures:")
        for f in failures:
            print(f)
        return 1
    print("Controller: all configs completed OK")
    return 0


def parse_args():
    p = argparse.ArgumentParser(description="Stress test fftfree dispatching")
    p.add_argument("--lib", help="explicit path to shared library", default=None)
    p.add_argument("--child", action="store_true", help="run as child for single config")
    p.add_argument("--cfg", help="json-encoded cfg dict for child mode", default=None)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        lib_path = find_library(args.lib)
    except Exception as e:
        print("Could not find library:", e, file=sys.stderr)
        sys.exit(2)
    if args.child:
        if not args.cfg:
            print("Child mode requires --cfg JSON", file=sys.stderr)
            sys.exit(2)
        cfg = json.loads(args.cfg)
        rc = child_run(str(lib_path), cfg, args.iters, args.seed)
        sys.exit(rc)
    else:
        # controller mode: run a few thread counts
        threads_list = [1, 2, 4, 8]
        inner_list = [1, 2, 4]
        rc = controller(str(lib_path), threads_list, inner_list, args.iters)
        sys.exit(rc)
