#!/usr/bin/env python3
"""
Train mipmap blend weights over a bank of FFT/STFT configurations using PyTorch
while executing all analysis/synthesis (forward/inverse) via the C library
(`fft_cffi`) through cffi. The training objective combines:

- Reconstruction SNR on the waveform after inverse
- Perceptual spectrogram terms on a canonical grid:
  - Soft-histogram match on log-magnitude
  - Local contrast (std over neighborhoods)
- Optional smoothness regularization on the weights (TV over time/freq)

Notes
-----
- Forward and inverse are executed by the C library. We implement a custom
  autograd.Function whose forward calls C inverse (C2R) and whose backward
  calls C forward (R2C) as the adjoint, keeping the entire path differentiable
  w.r.t. the weighting network parameters.
- We treat the per-config complex STFTs X_u as data (no gradient w.r.t. audio
  samples). Gradients flow through the blend weights into the blended complex
  spectrogram, then back through the adjoint STFT to the weights.
- Complex tensors are represented as two real channels: (real, imag).
- All tensors are kept on CPU to interop with cffi; CUDA can be added with
  pinned transfers but is out-of-scope for this minimal scaffold.

Usage (example)
---------------
  python train_mipmap_weights.py \
    --input MegaMix.wav \
    --epochs 2 --steps-per-epoch 50 \
    --ns 512 1024 2048 4096 \
    --hop-frac 4 \
    --threads 8 \
    --lr 1e-3 --tv 1e-3 --lambda-hist 0.1 --lambda-contrast 0.1

This will:
- Build forward (R2C) contexts for each N in --ns and one inverse (C2R)
  context for a reference N (default: the middle entry of --ns)
- Stream random chunks from the input WAV and optimize the weight network
  to minimize the composite loss.
"""
from __future__ import annotations

import argparse
import math
import os
import queue
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cffi import FFI
from scipy import signal as sp_signal
from scipy.io import wavfile


def _candidate_library_names() -> Iterable[str]:
    if os.name == "nt":
        yield "fft_cffi.dll"
    elif sys.platform == "darwin":
        yield "libfft_cffi.dylib"
    else:
        yield "libfft_cffi.so"


def _discover_library(explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Library {p} not found")
        return p
    env = os.environ.get("FFTFREE_CFFI_LIB")
    if env:
        p = Path(env).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"FFTFREE_CFFI_LIB -> {p} not found")
        return p
    root = Path(__file__).resolve().parent
    build = root / "build"
    rel = build / "Release"
    for d in (root, build, rel):
        for name in _candidate_library_names():
            cand = d / name
            if cand.exists():
                return cand
            if d.exists():
                m = next(d.rglob(name), None)
                if m:
                    return m
    raise FileNotFoundError("Could not locate fft_cffi shared library. Use --lib or set FFTFREE_CFFI_LIB")


def _build_ffi() -> FFI:
    ffi = FFI()
    ffi.cdef(
        """
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
                    int inner_threads,
                    int save_crash_logs,
                    int silent_crash_reports);
void* fft_init_full_v2(size_t n,
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
                    int inner_threads,
                    int save_crash_logs,
                    int silent_crash_reports,
                    int apply_windows,
                    int apply_ola,
                    int analysis_window_kind,
                    float analysis_param1,
                    float analysis_param2,
                    int synthesis_window_kind,
                    float synthesis_param1,
                    float synthesis_param2,
                    int window_norm_policy,
                    int cola_mode);

size_t fft_execute_batched(void* handle,
                           const float* pcm,
                           size_t pcm_len,
                           float* out_real,
                           float* out_imag,
                           float* out_mag,
                           int pad_mode,
                           int enable_backup,
                           size_t max_frames);

size_t fft_execute_complex_batched(void* handle,
                                   const float* in_real,
                                   const float* in_imag,
                                   size_t frames,
                                   float* out_pcm,
                                   int pad_mode,
                                   int enable_backup,
                                   size_t max_frames);

size_t fft_ctx_size(void* handle);
void   fft_free(void* handle);
"""
    )
    return ffi


@dataclass
class FftPlan:
    n: int
    hop: int
    half: int
    inverse: int  # 0=fwd R2C, 1=inv C2R
    handle: object
    apply_windows: int
    apply_ola: int


class CffiFft:
    def __init__(self, lib_path: Path, threads: int):
        self.ffi = _build_ffi()
        self.lib = self.ffi.dlopen(str(lib_path))
        self.threads = int(max(1, threads))
        self._plans: List[FftPlan] = []

    def make_plan(self,
                  n: int,
                  hop: int,
                  inverse: bool,
                  half_spectrum: bool,
                  *,
                  apply_windows: bool,
                  apply_ola: bool,
                  analysis_win_kind: int,
                  analysis_p1: float,
                  analysis_p2: float,
                  synthesis_win_kind: int,
                  synthesis_p1: float,
                  synthesis_p2: float,
                  win_norm_policy: int,
                  cola_mode: int) -> FftPlan:
        n = int(n)
        hop = int(hop)
        inv = 1 if inverse else 0
        half = 1 if half_spectrum else 0
        init_full_v2 = getattr(self.lib, 'fft_init_full_v2', None)
        init_full = getattr(self.lib, 'fft_init_full', None)
        if init_full_v2 is not None:
            h = init_full_v2(
                int(n),
                int(self.threads),
                1,
                int(inv),
                0,  # kernel auto
                0,  # radix unspecified
                self.ffi.NULL,
                0,
                1,          # pad last frame
                int(n),     # window = N
                int(hop),   # hop
                1,          # stft mode
                1 if not inverse else 2,  # transform: R2C for forward, C2R for inverse
                0,          # reduce_magnitude
                0,          # store_polar
                int(half),
                1,          # allow outer pool
                0,
                0,
                0,
                0,
                1 if apply_windows else 0,
                1 if (apply_ola and inverse) else 0,
                int(analysis_win_kind),
                float(analysis_p1),
                float(analysis_p2),
                int(synthesis_win_kind),
                float(synthesis_p1),
                float(synthesis_p2),
                int(win_norm_policy),
                int(cola_mode),
            )
        else:
            h = init_full(
                int(n),
                int(self.threads),
                1,
                int(inv),
                0,  # kernel auto
                0,  # radix unspecified
                self.ffi.NULL,
                0,
                1,          # pad last frame
                int(n),     # window = N
                int(hop),   # hop
                1,          # stft mode
                1 if not inverse else 2,  # transform: R2C for forward, C2R for inverse
                0,          # reduce_magnitude
                0,          # store_polar
                int(half),
                1,          # allow outer pool
                0,
                0,
                0,
                0,
            )
        if h == self.ffi.NULL:
            raise RuntimeError("fft_init_full failed")
        plan = FftPlan(n=n, hop=hop, half=half, inverse=inv, handle=h,
                       apply_windows=1 if apply_windows else 0,
                       apply_ola=1 if (apply_ola and inverse) else 0)
        self._plans.append(plan)
        return plan

    def free_all(self):
        for p in self._plans:
            try:
                self.lib.fft_free(p.handle)
            except Exception:
                pass
        self._plans.clear()

    # Forward STFT (R2C): returns real, imag, mag with shape (bins, frames)
    def stft(self, plan: FftPlan, pcm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        assert plan.inverse == 0, "Plan must be forward (R2C)"
        pcm = np.ascontiguousarray(pcm.astype(np.float32))
        bins = plan.n // 2 + 1 if plan.half else plan.n
        # Worst-case frames: len//hop + 2
        max_frames = int(pcm.shape[0] // max(1, plan.hop) + 3)
        out_real = np.zeros(max_frames * bins, dtype=np.float32)
        out_imag = np.zeros(max_frames * bins, dtype=np.float32)
        out_mag = np.zeros(max_frames * bins, dtype=np.float32)
        produced = int(self.lib.fft_execute_batched(
            plan.handle,
            self.ffi.cast("float *", pcm.ctypes.data),
            int(pcm.size),
            self.ffi.cast("float *", out_real.ctypes.data),
            self.ffi.cast("float *", out_imag.ctypes.data),
            self.ffi.cast("float *", out_mag.ctypes.data),
            1,  # pad last
            1,  # enable_backup
            0,
        ))
        if produced <= 0:
            raise RuntimeError("fft_execute_batched failed")
        used = produced * bins
        real = out_real[:used].reshape(produced, bins).T  # (bins, frames)
        imag = out_imag[:used].reshape(produced, bins).T
        mag  = out_mag[:used].reshape(produced, bins).T
        return real, imag, mag, bins, produced

    # Inverse ISTFT (C2R): real/imag (bins, frames) -> waveform 1-D
    def istft(self, plan: FftPlan, real: np.ndarray, imag: np.ndarray) -> np.ndarray:
        assert plan.inverse == 1, "Plan must be inverse (C2R)"
        assert real.shape == imag.shape and real.ndim == 2
        bins, frames = map(int, real.shape)
        plan_bins = plan.n // 2 + 1 if plan.half else plan.n
        if bins > plan_bins:
            real = real[:plan_bins, :]
            imag = imag[:plan_bins, :]
            bins = plan_bins
        in_real = np.zeros(frames * plan_bins, dtype=np.float32)
        in_imag = np.zeros(frames * plan_bins, dtype=np.float32)
        # Flatten frame-major
        for f in range(frames):
            base = f * plan_bins
            in_real[base:base + bins] = real[:, f]
            in_imag[base:base + bins] = imag[:, f]
        out_frames = np.zeros(frames * plan.n, dtype=np.float32)
        produced = int(self.lib.fft_execute_complex_batched(
            plan.handle,
            self.ffi.cast("float *", in_real.ctypes.data),
            self.ffi.cast("float *", in_imag.ctypes.data),
            int(frames),
            self.ffi.cast("float *", out_frames.ctypes.data),
            1,
            1,
            0,
        ))
        if produced <= 0:
            raise RuntimeError("fft_execute_complex_batched failed")
        # If C did OLA internally, extract overlapped length and return
        hop = plan.hop
        out_len = (frames - 1) * hop + plan.n
        if plan.apply_ola:
            return out_frames[:out_len]
        # Otherwise, perform Python-side OLA
        wav = np.zeros(out_len, dtype=np.float32)
        counts = np.zeros(out_len, dtype=np.float32)
        for f in range(frames):
            start = f * hop
            seg = out_frames[f * plan.n:(f + 1) * plan.n]
            wav[start:start + plan.n] += seg
            counts[start:start + plan.n] += 1.0
        m = counts > 0
        wav[m] /= counts[m]
        return wav


class ISTFT_CFFI_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, real: torch.Tensor, imag: torch.Tensor, plans: Tuple[CffiFft, FftPlan, FftPlan]):
        # real/imag: [B, F, T]
        assert real.shape == imag.shape and real.ndim == 3
        cffi, inv_plan, fwd_plan = plans
        B, Freq, Frames = real.shape
        outs = []
        input_device = real.device
        for b in range(B):
            r_np = real[b].detach().cpu().to(torch.float32).numpy()
            i_np = imag[b].detach().cpu().to(torch.float32).numpy()
            wav = cffi.istft(inv_plan, r_np, i_np)
            outs.append(torch.from_numpy(wav).to(input_device))
        wav_pad = torch.nn.utils.rnn.pad_sequence(outs, batch_first=True)
        ctx.save_for_backward(torch.tensor([wav.shape[0] for wav in outs], dtype=torch.long))
        ctx.plans = plans
        ctx.spec_shape = (B, Freq, Frames)
        ctx.input_device = input_device
        return wav_pad  # [B, T_out] (same T_out per-batch if frames constant)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        # Map grad_out (waveform) -> grad w.r.t. input complex spectrogram via adjoint STFT
        lengths, = ctx.saved_tensors
        cffi, inv_plan, fwd_plan = ctx.plans
        B, Freq, Frames = ctx.spec_shape
        device = ctx.input_device
        grad_real = torch.zeros((B, Freq, Frames), dtype=torch.float32, device=device)
        grad_imag = torch.zeros((B, Freq, Frames), dtype=torch.float32, device=device)
        for b in range(B):
            g = grad_out[b, : lengths[b].item()].detach().cpu().to(torch.float32).numpy()
            r, i, _m, bins, frames = cffi.stft(fwd_plan, g)
            # Align to expected (Freq, Frames)
            r_t = torch.from_numpy(r).to(device)
            i_t = torch.from_numpy(i).to(device)
            if r_t.shape != (Freq, Frames):
                # Bilinear resize to match saved spec shape
                r_t = _resize_2d(r_t.unsqueeze(0).unsqueeze(0), (Freq, Frames)).squeeze(0).squeeze(0)
                i_t = _resize_2d(i_t.unsqueeze(0).unsqueeze(0), (Freq, Frames)).squeeze(0).squeeze(0)
            grad_real[b] = r_t
            grad_imag[b] = i_t
        return grad_real, grad_imag, None


def _resize_2d(x: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
    # x: [N, C, H, W], out_hw=(H_out, W_out)
    return F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)


class TinyWeightNet(nn.Module):
    """Shallow per-tile gating network over stacked magnitudes.

    Input:  [B, U, F, T] (magnitudes on canonical grid)
    Output: [B, U, F, T] (softmax over U per (f,t))
    """
    def __init__(self, U: int):
        super().__init__()
        # Depthwise conv to extract per-node features + 1x1 to mix
        self.dw = nn.Conv2d(U, U, kernel_size=3, padding=1, groups=U)
        self.pw = nn.Conv2d(U, U, kernel_size=1)

    def forward(self, mags: torch.Tensor) -> torch.Tensor:
        z = F.leaky_relu(self.dw(mags), 0.1)
        z = self.pw(z)
        # Softmax across node dimension U
        w = torch.softmax(z, dim=1)
        return w


def log_mag(x: torch.Tensor, eps: float = 1e-8, k: float = 1.0) -> torch.Tensor:
    return torch.log1p(k * torch.clamp(x, min=0.0) + eps)


def soft_histogram(x: torch.Tensor, bins: int = 64, minv: float = 0.0, maxv: float = 1.0, sigma: float = 0.02) -> torch.Tensor:
    """Differentiable histogram over last two dims (F,T).

    Returns [B, U?, bins] if input is [B, U?, F, T] or [B, F, T].
    """
    orig_shape = x.shape
    if x.ndim == 4:
        B, U, F, T = x.shape
        x_flat = x.reshape(B, U, -1)
        reduce_dim = 2
    elif x.ndim == 3:
        B, F, T = x.shape
        x_flat = x.reshape(B, -1)
        reduce_dim = 1
    else:
        raise ValueError("soft_histogram expects 3D or 4D input")
    centers = torch.linspace(minv, maxv, bins, device=x.device, dtype=x.dtype)
    # Gaussian kernel density per bin center
    def _kern(v):
        return torch.exp(-0.5 * ((v[..., None] - centers[None, None, :]) / sigma) ** 2)
    k = _kern(x_flat)
    h = k.sum(dim=reduce_dim)
    # Normalize per-sample
    h = h / (h.sum(dim=-1, keepdim=True) + 1e-8)
    return h


def _nan_guard(name: str, tensor: torch.Tensor, clamp: Optional[float] = None) -> torch.Tensor:
    if torch.isfinite(tensor).all():
        return tensor
    nan_count = torch.isnan(tensor).sum().item()
    posinf_count = torch.isposinf(tensor).sum().item()
    neginf_count = torch.isneginf(tensor).sum().item()
    print(f"[nan] {name} non-finite detected (nan={nan_count}, +inf={posinf_count}, -inf={neginf_count})")
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    if clamp is not None:
        tensor = torch.clamp(tensor, min=-clamp, max=clamp)
    return tensor


def local_contrast_map(x: torch.Tensor, ksize: int = 7) -> torch.Tensor:
    # x: [B, 1, F, T] or [B, F, T]
    if x.ndim == 3:
        x = x.unsqueeze(1)
    pad = ksize // 2
    w = torch.ones((1, 1, ksize, ksize), dtype=x.dtype, device=x.device)
    w = w / (ksize * ksize)
    mean = F.conv2d(x, w, padding=pad)
    mean2 = F.conv2d(x * x, w, padding=pad)
    var = torch.clamp(mean2 - mean * mean, min=0.0)
    std = torch.sqrt(var + 1e-8)
    return std.squeeze(1)


def snr_loss(x_hat: torch.Tensor, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # Return ratio MSE/energy (minimize)
    e = torch.mean((x_hat - x) ** 2, dim=-1)
    s = torch.mean(x ** 2, dim=-1) + eps
    return (e / s).mean()


def resample_to_rate(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample a 1-D float32 signal to match ``dst_rate`` using polyphase FIR."""

    if samples.size == 0 or src_rate == dst_rate:
        return samples
    if src_rate <= 0 or dst_rate <= 0:
        raise ValueError("Sample rates must be positive for resampling")

    g = math.gcd(int(src_rate), int(dst_rate))
    up = int(dst_rate // g)
    down = int(src_rate // g)
    resampled = sp_signal.resample_poly(samples.astype(np.float32, copy=False), up, down)
    return resampled.astype(np.float32, copy=False)


def chunk_stream(data: np.ndarray,
                 chunk: int,
                 overlap_frac: float,
                 loop: bool) -> Iterator[np.ndarray]:
    """Yield sequential chunks from ``data`` with optional overlap and looping.

    Parameters
    ----------
    data
        1-D numpy array of audio samples (float32 recommended).
    chunk
        Number of samples per yielded segment.
    overlap_frac
        Fraction in [0, 0.95] describing how much of each chunk should be
        re-used in the subsequent segment (0 -> disjoint, 0.5 -> 50% overlap).
    loop
        If True, wrap around to the beginning when the end is reached.
    """

    if chunk <= 0:
        raise ValueError("chunk_stream requires chunk > 0")
    if not 0.0 <= overlap_frac < 1.0:
        raise ValueError("overlap_frac must be in [0.0, 1.0)")

    total = int(data.shape[0])
    if total == 0:
        raise ValueError("chunk_stream received empty input")

    step = int(round(chunk * (1.0 - overlap_frac)))
    if step <= 0:
        step = 1

    cursor = 0
    while True:
        end = cursor + chunk
        if end <= total:
            seg = data[cursor:end]
        else:
            tail = data[cursor:]
            if not loop:
                if tail.size == 0:
                    return
                pad = chunk - tail.size
                seg = np.pad(tail, (0, pad), mode="constant")
                yield np.ascontiguousarray(seg)
                return
            needed = chunk - tail.size
            pieces = [tail]
            while needed > 0:
                take = min(needed, total)
                pieces.append(data[:take])
                needed -= take
            seg = np.concatenate(pieces, axis=0)
        yield np.ascontiguousarray(seg)

        if loop:
            cursor = (cursor + step) % total
        else:
            cursor += step
            if cursor >= total:
                return


class SpectrogramScroller:
    """Maintain scrolling pygame visualization synced with streamed audio."""

    def __init__(self,
                 width: int,
                 height: int,
                 mode: str,
                 history_frames: int,
                 node_specs: Sequence[Tuple[int, int]],
                 top_initial: Optional[str] = None,
                 bottom_initial: Optional[str] = None):
        import pygame

        self.width = int(width)
        self.height = int(height)
        self.mode = mode
        self.history_frames = history_frames if history_frames > 0 else self.width
        self.node_specs = list(node_specs)
        self.font = pygame.font.SysFont(None, 16)

        self.series: Dict[str, Optional[np.ndarray]] = {}
        self.labels: Dict[str, str] = {}
        self.mode_sequence: List[str] = []
        self.top_mode: str = "ref"
        self.bottom_mode: str = "weights"
        self.requested_top = (top_initial or "ref").lower()
        self.requested_bottom = (bottom_initial or "weights").lower()

    @staticmethod
    def _to_numpy(x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().numpy().astype(np.float32, copy=False)

    def _append(self, buffer: Optional[np.ndarray], new: np.ndarray) -> np.ndarray:
        if buffer is None:
            if new.shape[1] <= self.history_frames:
                return new.copy()
            return new[:, -self.history_frames:].copy()
        total_frames = buffer.shape[1] + new.shape[1]
        if total_frames <= self.history_frames:
            return np.concatenate([buffer, new], axis=1)
        keep = self.history_frames - new.shape[1]
        if keep <= 0:
            return new[:, -self.history_frames:]
        return np.concatenate([buffer[:, -keep:], new], axis=1)

    def _store_series(self, key: str, value: np.ndarray) -> None:
        existing = self.series.get(key)
        self.series[key] = self._append(existing, value)

    def _ensure_modes(self, node_count: int) -> None:
        if self.mode_sequence:
            return
        self.mode_sequence = ["ref", "blend"] + [f"node{idx}" for idx in range(node_count)]
        self.labels["ref"] = "Reference"
        self.labels["blend"] = "Blend (weighted mix)"
        for idx, spec in enumerate(self.node_specs):
            n, hop = spec
            self.labels[f"node{idx}"] = f"Node {idx} (N={n}, H={hop})"
        self.labels["weights"] = "Weights (nodes x freq)"
        self.mode_sequence.append("weights")
        self.top_mode = self._resolve_mode(self.requested_top, "blend", allow_weights=False)
        self.bottom_mode = self._resolve_mode(self.requested_bottom, "weights", allow_weights=True)

    def _resolve_mode(self, requested: Optional[str], fallback: str, *, allow_weights: bool) -> str:
        if not self.mode_sequence:
            return fallback
        if requested is None:
            return fallback
        key = requested.lower()
        if key == "weights":
            return "weights" if allow_weights else fallback
        if key in self.mode_sequence and (allow_weights or key != "weights"):
            return key
        if key.startswith("node"):
            return key if key in self.mode_sequence else fallback
        return fallback

    def update(self,
               ref_log: torch.Tensor,
               blend_log: torch.Tensor,
               node_logs: Sequence[torch.Tensor],
               weights: torch.Tensor) -> None:
        ref_np = self._to_numpy(ref_log)
        blend_np = self._to_numpy(blend_log)
        weights_np = self._to_numpy(weights)
        node_np_list = []
        for idx, node_tensor in enumerate(node_logs):
            key = f"node{idx}"
            node_np = self._to_numpy(node_tensor)
            node_np_list.append(node_np)
            self._store_series(key, node_np)

        if weights_np.ndim == 3 and len(node_np_list) == weights_np.shape[0]:
            mags = []
            for node_np in node_np_list:
                mags.append(np.expm1(np.clip(node_np, 0.0, None)))
            mags_stack = np.stack(mags, axis=0)
            combined_mag = np.sum(np.clip(weights_np, 0.0, None) * mags_stack, axis=0)
            combined_mag = np.clip(combined_mag, 0.0, None)
            combined_log = np.log1p(combined_mag)
        else:
            combined_log = blend_np

        if weights_np.ndim == 3:
            weights_heat = weights_np.reshape(weights_np.shape[0] * weights_np.shape[1], weights_np.shape[2])
        else:
            weights_heat = weights_np

        self._store_series("ref", ref_np)
        self._store_series("blend", combined_log)
        self._store_series("weights", weights_heat)
        self._ensure_modes(len(node_logs))

    @staticmethod
    def _norm01(arr: np.ndarray) -> np.ndarray:
        if arr.size == 0:
            return np.zeros_like(arr)
        mn = float(np.min(arr))
        mx = float(np.max(arr))
        if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn <= 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    def _blank_surface(self, width: int, height: int):
        import pygame

        surf = pygame.Surface((width, max(0, height)))
        surf.fill((0, 0, 0))
        return surf

    def _gray_surface(self, data: Optional[np.ndarray], width: int, height: int):
        return self._make_panel_surface(data, width, height, palette="gray")

    def _weights_surface(self, data: Optional[np.ndarray], width: int, height: int):
        return self._make_panel_surface(data, width, height, palette="gray", stretch_rows=True)

    def _make_panel_surface(self,
                            data: Optional[np.ndarray],
                            width: int,
                            height: int,
                            *,
                            palette: str,
                            stretch_rows: bool = False):
        import pygame

        if data is None or data.size == 0:
            return self._blank_surface(width, height)
        vals = self._norm01(data)
        rows, frames = vals.shape
        if rows == 0 or frames == 0:
            return self._blank_surface(width, height)
        # Keep only the most recent columns so we do not rescale horizontally
        cols = min(frames, width)
        vals = vals[:, -cols:]
        if stretch_rows and rows < height:
            repeat = max(1, height // rows)
            vals = np.repeat(vals, repeat, axis=0)
            if vals.shape[0] < height:
                pad = height - vals.shape[0]
                vals = np.concatenate([vals, np.tile(vals[-1:, :], (pad, 1))], axis=0)
            rows = vals.shape[0]
        img = np.flipud(vals)
        if palette == "gray":
            img_u8 = (img * 255.0).astype(np.uint8)
            rgb = np.stack([img_u8.T, img_u8.T, img_u8.T], axis=2)
        else:
            raise ValueError(f"Unsupported palette {palette}")
        surf = pygame.surfarray.make_surface(np.ascontiguousarray(rgb))
        if surf.get_height() != height:
            surf = pygame.transform.smoothscale(surf, (surf.get_width(), height))
        if surf.get_width() == width:
            return surf
        base = self._blank_surface(width, height)
        base.blit(surf, (width - surf.get_width(), 0))
        return base

    @staticmethod
    def _surface_from_array(rgb: np.ndarray, width: int, height: int):
        import pygame

        surf = pygame.surfarray.make_surface(np.ascontiguousarray(rgb))
        if surf.get_size() != (width, height):
            surf = pygame.transform.smoothscale(surf, (width, height))
        return surf

    def cycle_mode(self, panel: str, step: int) -> None:
        if not self.mode_sequence:
            return
        spectral_modes = [m for m in self.mode_sequence if m != "weights"]
        if panel == "top":
            if not spectral_modes:
                return
            idx = spectral_modes.index(self.top_mode) if self.top_mode in spectral_modes else 0
            idx = (idx + step) % len(spectral_modes)
            self.top_mode = spectral_modes[idx]
        elif panel == "bottom":
            pool = self.mode_sequence
            if not pool:
                return
            idx = pool.index(self.bottom_mode) if self.bottom_mode in pool else 0
            idx = (idx + step) % len(pool)
            self.bottom_mode = pool[idx]

    def mode_label(self, mode: str) -> str:
        return self.labels.get(mode, mode)

    def handle_key(self, key: int) -> None:
        import pygame

        if key == pygame.K_1:
            self.cycle_mode("top", -1)
        elif key == pygame.K_2:
            self.cycle_mode("top", 1)
        elif key == pygame.K_q:
            self.cycle_mode("bottom", -1)
        elif key == pygame.K_w:
            self.cycle_mode("bottom", 1)

    def _render_mode(self, key: str, width: int, height: int):
        if key == "weights":
            return self._weights_surface(self.series.get("weights"), width, height)
        data = self.series.get(key)
        return self._gray_surface(data, width, height)

    def render(self, screen) -> None:
        import pygame

        W = self.width
        H = self.height
        top_h = int(round(H * 0.66))
        bot_h = max(1, H - top_h)

        top_surface = self._render_mode(self.top_mode, W, top_h)
        bottom_surface = self._render_mode(self.bottom_mode, W, bot_h)

        screen.blit(top_surface, (0, 0))
        screen.blit(bottom_surface, (0, top_h))

        if self.font is not None:
            top_label = self.font.render(f"Top: {self.mode_label(self.top_mode)} (1/2)", True, (255, 255, 0))
            bottom_label = self.font.render(f"Bottom: {self.mode_label(self.bottom_mode)} (Q/W)", True, (200, 200, 200))
            screen.blit(top_label, (6, 6))
            screen.blit(bottom_label, (6, top_h + 6))


class AudioStreamer:
    """Background audio playback worker that consumes numpy buffers asynchronously."""

    def __init__(self,
                 mixer_channel,
                 input_sr: int,
                 device_sr: int,
                 gain: float,
                 stream_step: Optional[int],
                 max_queue: int = 8):
        import pygame

        self.channel = mixer_channel
        self.input_sr = int(input_sr)
        self.device_sr = int(device_sr) if device_sr > 0 else int(input_sr)
        self.gain = float(gain)
        self.stream_step = int(stream_step) if stream_step else None
        self.queue: "queue.Queue[np.ndarray | None]" = queue.Queue(maxsize=max(1, max_queue))
        self.thread = threading.Thread(target=self._worker, name="AudioStreamer", daemon=True)
        self.running = True
        self.prime_full = True if self.stream_step else False
        init = pygame.mixer.get_init()
        self.channels = int(init[2]) if init else 1
        self.guard: List[object] = []
        self.thread.start()

    def push(self, samples: np.ndarray) -> None:
        if not self.running:
            return
        if samples is None or samples.size == 0:
            return
        try:
            self.queue.put_nowait(samples.astype(np.float32, copy=False))
        except queue.Full:
            try:
                _ = self.queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(samples.astype(np.float32, copy=False))
            except queue.Full:
                pass

    def close(self) -> None:
        if not self.running:
            return
        self.running = False
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            # Force space and retry
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(None)
            except queue.Full:
                pass
        self.thread.join(timeout=1.0)
        self.guard.clear()

    def _worker(self) -> None:
        import pygame

        while True:
            try:
                item = self.queue.get(timeout=0.5)
            except queue.Empty:
                if not self.running:
                    break
                continue
            if item is None:
                break
            try:
                play_buf = self._prepare_buffer(item)
                if play_buf.size == 0:
                    continue
                mx = float(np.max(np.abs(play_buf))) if play_buf.size else 0.0
                if mx > 0.0:
                    play_buf = play_buf / mx
                play_buf = np.clip(play_buf * self.gain, -1.0, 1.0)
                if self.device_sr != self.input_sr:
                    play_buf = resample_to_rate(play_buf, self.input_sr, self.device_sr)
                    play_buf = np.clip(play_buf, -1.0, 1.0)
                y16 = (play_buf * 32767.0).astype(np.int16)
                arr = y16[:, None] if self.channels > 1 else y16
                if self.channels > 1:
                    arr = np.repeat(arr, self.channels, axis=1)
                snd = pygame.sndarray.make_sound(np.ascontiguousarray(arr))
                if self.channel.get_busy():
                    self.channel.queue(snd)
                else:
                    self.channel.play(snd)
                self.guard.append(snd)
                if len(self.guard) > 8:
                    self.guard.pop(0)
            except Exception:
                # Disable on failure
                try:
                    self.channel = None
                except Exception:
                    pass
                break

    def _prepare_buffer(self, buf: np.ndarray) -> np.ndarray:
        data = np.asarray(buf, dtype=np.float32)
        if data.ndim != 1:
            data = data.reshape(-1)
        if self.stream_step is None:
            return data.copy()
        if self.prime_full:
            self.prime_full = False
            return data.copy()
        if self.stream_step <= 0:
            return data.copy()
        tail = data[-self.stream_step:]
        return tail.copy()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train mipmap blend weights using C FFT with PyTorch autograd")
    p.add_argument("--input", required=True, help="Path to mono WAV file to train on")
    p.add_argument("--lib", help="Path to fft_cffi shared library")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--ns", type=int, nargs="+", default=[512, 1024, 2048, 4096], help="FFT sizes for mipmap nodes")
    p.add_argument("--hop-frac", type=int, default=2, help="Hop = N / hop_frac (2 for Hann COLA; 1 for rectangular no-overlap)")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--steps-per-epoch", type=int, default=50)
    p.add_argument("--chunk", type=int, default=44100, help="Training chunk length in samples")
    p.add_argument("--segment-batch-size", type=int, default=1, help="Number of sequential segments to accumulate before a training update")
    p.add_argument("--segment-batch-slide", type=int, default=1, help="Number of segments to drop from the buffer after each update (1 -> slide window)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Torch device for training (auto -> cuda if available)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda-hist", type=float, default=0.1)
    p.add_argument("--lambda-contrast", type=float, default=0.1)
    p.add_argument("--tv", type=float, default=1e-3, help="Total-variation regularization on weights")
    p.add_argument("--stream", action="store_true", help="Process the input sequentially (streaming) instead of random chunks")
    p.add_argument("--stream-overlap", type=float, default=0.5, help="Fraction of each chunk to overlap with the next in streaming mode")
    # Complex blend controls
    p.add_argument("--phase-blend", choices=["unit", "cartesian"], default="unit", help="Blend phase on the unit circle (default) or Cartesian sum")
    p.add_argument("--phase-mag-exp", type=float, default=1.0, help="Exponent beta for magnitude influence in phase averaging: weights multiply by (mag^beta)")
    p.add_argument("--mag-combine", choices=["sum", "powermean"], default="sum", help="How to combine magnitudes across nodes")
    p.add_argument("--mag-p", type=float, default=1.0, help="Power mean exponent p (used when --mag-combine=powermean; p!=0)")
    p.add_argument("--phase-eps", type=float, default=1e-8, help="Stability epsilon for unit-circle phase mixing")
    p.add_argument("--align", choices=["transport", "resize"], default="transport", help="Align node spectrograms to canonical grid via transport (A_ref S_u) or naive resize")
    p.add_argument("--save", default="mipmap_weights.pt")
    # Window/OLA configuration (applied inside C synthesis/analysis)
    p.add_argument("--apply-windows", action="store_true", help="Enable analysis/synthesis windowing inside the C library")
    p.add_argument("--apply-ola", action="store_true", help="Perform overlap-add inside the C inverse executor")
    p.add_argument("--win", choices=["rect","hann","hamming","blackman","tukey","kaiser"], default="hann")
    p.add_argument("--win-alpha", type=float, default=0.5, help="Tukey alpha (when --win=tukey)")
    p.add_argument("--win-beta", type=float, default=0.0, help="Kaiser beta (placeholder)")
    p.add_argument("--win-norm", choices=["none","l2","area"], default="area")
    # Visualization (pygame)
    p.add_argument("--viz", action="store_true", help="Open a pygame window to show blended magnitude spectrogram during training")
    p.add_argument("--viz-width", type=int, default=512)
    p.add_argument("--viz-height", type=int, default=256)
    p.add_argument("--viz-mode", choices=["mag","rgb"], default="mag", help="Visualization mode for ref/blend panels: mag (grayscale) or rgb (R=real,G=imag,B=mag)")
    p.add_argument("--viz-fmin", type=float, default=0.0, help="Min frequency (Hz) to display (0=DC)")
    p.add_argument("--viz-fmax", type=float, default=0.0, help="Max frequency (Hz) to display (0=Nyquist)")
    p.add_argument("--viz-time-frames", type=int, default=0, help="Limit number of time frames shown (0=all)")
    p.add_argument("--viz-history", type=int, default=512, help="Number of canonical frames to retain in the scrolling visualization (0=keep all)")
    p.add_argument("--viz-top-mode", default="blend", help="Initial selection for the top panel (blend, ref, nodeX)")
    p.add_argument("--viz-bottom-mode", default="weights", help="Initial selection for the bottom panel (weights, blend, ref, nodeX)")
    # Audio monitoring
    p.add_argument("--audio", choices=["off", "blend", "residual", "ref"], default="off", help="Monitor reconstructed blend, residual (orig-blend), or reference reconstruction")
    p.add_argument("--audio-gain", type=float, default=0.9, help="Output gain applied before int16 conversion")
    # Debug
    p.add_argument("--debug-check-ref", action="store_true", help="Print per-step SNR of reference round-trip S_ref(A_ref(x)) vs x")
    return p.parse_args()


def choose_reference_index(ns: Sequence[int]) -> int:
    # Middle entry as reference grid
    return max(0, (len(ns) - 1) // 2)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available")
    if device.type == "cuda":
        cuda_index = device.index if device.index is not None else 0
        torch.cuda.set_device(cuda_index)
        torch.backends.cudnn.enabled = False
        device = torch.device("cuda", cuda_index)
    print(f"[device] using {device}")

    # Load audio (mono)
    sr, data = wavfile.read(args.input)
    if data.ndim > 1:
        data = data[:, 0]
    x_all = data.astype(np.float32)
    total_len = int(x_all.shape[0])
    if total_len < args.chunk:
        raise ValueError("Input WAV too short for a training chunk")

    segment_batch_size = max(1, int(args.segment_batch_size))
    segment_batch_slide = max(1, int(args.segment_batch_slide))
    if segment_batch_slide > segment_batch_size:
        segment_batch_slide = segment_batch_size

    stream_iter: Optional[Iterator[np.ndarray]] = None
    stream_step: Optional[int] = None
    if args.stream:
        overlap = float(args.stream_overlap)
        if not 0.0 <= overlap < 1.0:
            raise ValueError("--stream-overlap must be in [0.0, 1.0)")
        stream_iter = chunk_stream(x_all, int(args.chunk), overlap, loop=True)
        step = int(round(int(args.chunk) * (1.0 - overlap)))
        if step <= 0:
            step = 1
        elif step > int(args.chunk):
            step = int(args.chunk)
        stream_step = step

    lib_path = _discover_library(args.lib)
    c = CffiFft(lib_path, threads=args.threads)

    # Build plans
    Ns = list(args.ns)
    hops = [max(1, n // args.hop_frac) for n in Ns]
    ref_idx = choose_reference_index(Ns)
    ref_N, ref_hop = Ns[ref_idx], hops[ref_idx]
    forward_plans: List[FftPlan] = []
    inverse_plans: List[FftPlan] = []

    # Auto defaults based on overlap/window
    has_overlap = any(h < n for h, n in zip(hops, Ns))
    auto_apply_windows = args.apply_windows or has_overlap or (args.win != "rect")
    auto_apply_ola = args.apply_ola or has_overlap

    def _win_kind(name: str) -> int:
        return {"rect":0, "hann":1, "hamming":2, "blackman":3, "tukey":4, "kaiser":5}[name]
    def _win_norm(name: str) -> int:
        return {"none":0, "l2":1, "area":2}[name]

    for n, h in zip(Ns, hops):
        forward_plans.append(c.make_plan(
            n, h,
            inverse=False,
            half_spectrum=True,
            apply_windows=auto_apply_windows,
            apply_ola=False,
            analysis_win_kind=_win_kind(args.win),
            analysis_p1=args.win_alpha,
            analysis_p2=args.win_beta,
            synthesis_win_kind=_win_kind(args.win),
            synthesis_p1=args.win_alpha,
            synthesis_p2=args.win_beta,
            win_norm_policy=_win_norm(args.win_norm),
            cola_mode=1
        ))
        inverse_plans.append(c.make_plan(
            n, h,
            inverse=True,
            half_spectrum=True,
            apply_windows=auto_apply_windows,
            apply_ola=auto_apply_ola,
            analysis_win_kind=_win_kind(args.win),
            analysis_p1=args.win_alpha,
            analysis_p2=args.win_beta,
            synthesis_win_kind=_win_kind(args.win),
            synthesis_p1=args.win_alpha,
            synthesis_p2=args.win_beta,
            win_norm_policy=_win_norm(args.win_norm),
            cola_mode=1
        ))
    inverse_plan = c.make_plan(
        ref_N, ref_hop,
        inverse=True,
        half_spectrum=True,
        apply_windows=auto_apply_windows,
        apply_ola=auto_apply_ola,
        analysis_win_kind=_win_kind(args.win),
        analysis_p1=args.win_alpha,
        analysis_p2=args.win_beta,
        synthesis_win_kind=_win_kind(args.win),
        synthesis_p1=args.win_alpha,
        synthesis_p2=args.win_beta,
        win_norm_policy=_win_norm(args.win_norm),
        cola_mode=1
    )
    forward_ref = forward_plans[ref_idx]

    # Optional pygame viz + audio
    screen = None
    mixer_channel = None
    scroller: Optional[SpectrogramScroller] = None
    audio_streamer: Optional[AudioStreamer] = None
    if args.viz or args.audio != "off":
        try:
            import pygame
            pygame.init()
            if args.viz:
                screen = pygame.display.set_mode((args.viz_width, args.viz_height))
                pygame.display.set_caption("Ref / Blend / Weights (low freq bottom)")
                scroller = SpectrogramScroller(
                    width=args.viz_width,
                    height=args.viz_height,
                    mode=args.viz_mode,
                    history_frames=int(args.viz_history),
                    node_specs=list(zip(Ns, hops)),
                    top_initial=args.viz_top_mode,
                    bottom_initial=args.viz_bottom_mode
                )
            if args.audio != "off":
                try:
                    pygame.mixer.init(frequency=int(sr), size=-16, channels=1, buffer=1024)
                    mixer_channel = pygame.mixer.Channel(0)
                    actual = pygame.mixer.get_init()
                    if actual and actual[0] != int(sr):
                        print(f"[audio] warning: requested {sr} Hz, mixer using {actual[0]} Hz")
                    if mixer_channel is not None:
                        device_sr = actual[0] if actual else sr
                        step_for_stream = stream_step if stream_iter is not None else None
                        audio_streamer = AudioStreamer(mixer_channel, sr, device_sr, args.audio_gain, step_for_stream)
                except Exception as mex:
                    print(f"[audio] mixer init failed: {mex}")
                    mixer_channel = None
        except Exception as ex:
            print(f"[viz] failed to init pygame: {ex}")
            screen = None
            mixer_channel = None
            scroller = None

    # Small weight network
    U = len(Ns)
    weight_net = TinyWeightNet(U).to(device)
    opt = torch.optim.Adam(weight_net.parameters(), lr=args.lr)

    segment_buffer: List[dict] = []
    train_updates = 0
    stop_training = False

    # Training loop
    for epoch in range(args.epochs):
        if stop_training:
            break
        for step in range(args.steps_per_epoch):
            if stop_training:
                break
            if stream_iter is not None:
                try:
                    seg = next(stream_iter)
                except StopIteration:
                    stream_iter = chunk_stream(x_all, int(args.chunk), float(args.stream_overlap), loop=True)
                    seg = next(stream_iter)
            else:
                start = np.random.randint(0, total_len - args.chunk + 1)
                seg = x_all[start:start + args.chunk]
            seg_t = torch.from_numpy(seg).unsqueeze(0).to(device)  # [1, T]

            # Compute per-config STFTs from C (no grad path through these)
            specs: List[Tuple[torch.Tensor, torch.Tensor]] = []  # (real, imag) tensors in each node's native grid
            specs_np: List[Tuple[np.ndarray, np.ndarray]] = []    # matching numpy arrays for transport alignment
            frames_ref = None
            bins_ref = None
            ref_raw_np = None  # (real_np, imag_np, bins, frames) from direct A_ref(x)
            for u, plan in enumerate(forward_plans):
                r, i, m, bins, frames = c.stft(plan, seg)
                rt = torch.from_numpy(r).to(device)  # [F, T]
                it = torch.from_numpy(i).to(device)
                if u == ref_idx:
                    bins_ref, frames_ref = int(bins), int(frames)
                    # Keep an untouched copy of the exact reference analysis for audio 'ref'
                    ref_raw_np = (r.copy(), i.copy(), int(bins), int(frames))
                specs.append((rt, it))
                specs_np.append((r.copy(), i.copy()))

            assert frames_ref is not None and bins_ref is not None

            # Canonical grid (reference plan dims)
            F_ref, T_ref = bins_ref, frames_ref
            real_stack = []
            imag_stack = []
            mag_stack = []
            for u, ((rt, it), (r_np, i_np)) in enumerate(zip(specs, specs_np)):
                if args.align == "resize":
                    # Naive resize of complex to canonical
                    if rt.shape != (F_ref, T_ref):
                        rt2 = _resize_2d(rt.unsqueeze(0).unsqueeze(0), (F_ref, T_ref)).squeeze(0).squeeze(0)
                        it2 = _resize_2d(it.unsqueeze(0).unsqueeze(0), (F_ref, T_ref)).squeeze(0).squeeze(0)
                    else:
                        rt2, it2 = rt, it
                else:
                    # Transport: synthesize with node inverse, analyze with reference forward
                    y_u = c.istft(inverse_plans[u], r_np.astype(np.float32, copy=False), i_np.astype(np.float32, copy=False))
                    r2, i2, m2, bins2, frames2 = c.stft(forward_ref, y_u)
                    rt2 = torch.from_numpy(r2).to(device)
                    it2 = torch.from_numpy(i2).to(device)
                    # Align time frames by crop/pad to T_ref (bins should match F_ref)
                    if bins2 != F_ref:
                        # If mismatch, resize bins via bilinear (rare)
                        rt2 = _resize_2d(rt2.unsqueeze(0).unsqueeze(0), (F_ref, frames2)).squeeze(0).squeeze(0)
                        it2 = _resize_2d(it2.unsqueeze(0).unsqueeze(0), (F_ref, frames2)).squeeze(0).squeeze(0)
                    if frames2 < T_ref:
                        pad = T_ref - frames2
                        rt2 = torch.cat([rt2, torch.zeros(F_ref, pad, dtype=rt2.dtype, device=rt2.device)], dim=1)
                        it2 = torch.cat([it2, torch.zeros(F_ref, pad, dtype=it2.dtype, device=it2.device)], dim=1)
                    elif frames2 > T_ref:
                        rt2 = rt2[:, :T_ref]
                        it2 = it2[:, :T_ref]
                real_stack.append(rt2)
                imag_stack.append(it2)
                mag_stack.append(torch.sqrt(torch.clamp(rt2 ** 2 + it2 ** 2, min=0.0)))

            real_u = torch.stack(real_stack, dim=0).unsqueeze(0).contiguous()  # [1, U, F, T]
            imag_u = torch.stack(imag_stack, dim=0).unsqueeze(0).contiguous()
            mag_u = torch.stack(mag_stack, dim=0).unsqueeze(0).contiguous()

            real_u = _nan_guard("real_u", real_u)
            imag_u = _nan_guard("imag_u", imag_u)
            mag_u = _nan_guard("mag_u", mag_u, clamp=1e6)
            mag_u = torch.clamp(mag_u, min=0.0, max=1e6)

            ref_real_canon = real_u[0, ref_idx].detach()
            ref_imag_canon = imag_u[0, ref_idx].detach()

            segment_buffer.append({
                "real_u": real_u,
                "imag_u": imag_u,
                "mag_u": mag_u,
                "target": seg_t.detach(),
                "ref_real": ref_real_canon,
                "ref_imag": ref_imag_canon,
                "ref_raw_np": ref_raw_np,
            })

            if len(segment_buffer) < segment_batch_size:
                continue

            batch_items = segment_buffer[:segment_batch_size]
            real_u_batch = torch.cat([item["real_u"] for item in batch_items], dim=0).contiguous()
            imag_u_batch = torch.cat([item["imag_u"] for item in batch_items], dim=0).contiguous()
            mag_u_batch = torch.cat([item["mag_u"] for item in batch_items], dim=0).contiguous()
            ref_real_batch = torch.stack([item["ref_real"] for item in batch_items], dim=0).contiguous()
            ref_imag_batch = torch.stack([item["ref_imag"] for item in batch_items], dim=0).contiguous()

            real_u_batch = _nan_guard("real_u_batch", real_u_batch)
            imag_u_batch = _nan_guard("imag_u_batch", imag_u_batch)
            mag_u_batch = _nan_guard("mag_u_batch", mag_u_batch, clamp=1e6)
            mag_u_batch = torch.clamp(mag_u_batch, min=0.0, max=1e6)
            ref_raw_np_batch = [item["ref_raw_np"] for item in batch_items]

            B = real_u_batch.shape[0]
            F_ref = real_u_batch.shape[2]
            T_ref = real_u_batch.shape[3]
            out_len = (T_ref - 1) * ref_hop + ref_N

            target_batch_list = []
            for item in batch_items:
                tgt = item["target"]
                if tgt.shape[1] < out_len:
                    pad = out_len - tgt.shape[1]
                    tgt = F.pad(tgt, (0, pad))
                else:
                    tgt = tgt[:, :out_len]
                target_batch_list.append(tgt)
            x_target = torch.cat(target_batch_list, dim=0).contiguous()
            x_target = _nan_guard("x_target", x_target, clamp=1e3)

            with torch.enable_grad():
                mags_for_logits = log_mag(mag_u_batch, k=1.0)
                mags_for_logits = _nan_guard("mags_for_logits", mags_for_logits, clamp=20.0)
                mags_for_logits = mags_for_logits - mags_for_logits.mean(dim=(2, 3), keepdim=True)
                mags_for_logits = torch.clamp(mags_for_logits, min=-20.0, max=20.0)
                w_raw = weight_net(mags_for_logits).contiguous()
                if not torch.isfinite(w_raw).all():
                    print("[nan] weights produced non-finite values; applying uniform fallback")
                    w_clean = torch.where(torch.isfinite(w_raw), w_raw, torch.zeros_like(w_raw))
                else:
                    w_clean = w_raw
                w_clean = torch.clamp(w_clean, min=1e-6)
                norm = w_clean.sum(dim=1, keepdim=True)
                fallback_mask = norm <= 0
                if fallback_mask.any():
                    w_uniform = torch.full_like(w_clean, 1.0 / w_clean.shape[1])
                    w_clean = torch.where(fallback_mask, w_uniform, w_clean)
                    norm = w_clean.sum(dim=1, keepdim=True)
                w = w_clean / norm.clamp_min(1e-6)
                if args.phase_blend == "unit":
                    eps = float(args.phase_eps)
                    beta = float(args.phase_mag_exp)
                    M_u = torch.clamp(mag_u_batch, min=0.0, max=1e4)
                    cos_u = real_u_batch / (M_u + eps)
                    sin_u = imag_u_batch / (M_u + eps)
                    cos_u = _nan_guard("cos_u", cos_u)
                    sin_u = _nan_guard("sin_u", sin_u)
                    w_phase = w * torch.pow(M_u + eps, beta)
                    w_phase = _nan_guard("w_phase", w_phase)
                    cos_mix = torch.sum(w_phase * cos_u, dim=1)
                    sin_mix = torch.sum(w_phase * sin_u, dim=1)
                    phase_hat = torch.atan2(sin_mix, cos_mix)
                    if args.mag_combine == "powermean":
                        p = float(args.mag_p)
                        if abs(p) < 1e-6:
                            M_hat = torch.exp(torch.sum(w * torch.log(M_u + eps), dim=1))
                        else:
                            M_hat = torch.pow(torch.sum(w * torch.pow(M_u + eps, p), dim=1) + 1e-12, 1.0 / p)
                    else:
                        M_hat = torch.sum(w * M_u, dim=1)
                    real_blend = M_hat * torch.cos(phase_hat)
                    imag_blend = M_hat * torch.sin(phase_hat)
                else:
                    real_blend = torch.sum(w * real_u_batch, dim=1)
                    imag_blend = torch.sum(w * imag_u_batch, dim=1)

                real_blend = _nan_guard("real_blend", real_blend, clamp=1e4)
                imag_blend = _nan_guard("imag_blend", imag_blend, clamp=1e4)

                wav_hat = ISTFT_CFFI_Function.apply(real_blend, imag_blend, (c, inverse_plan, forward_ref))

                wav_hat = _nan_guard("wav_hat", wav_hat, clamp=1e3)
                l_snr = snr_loss(wav_hat, x_target)

                ref_mag = torch.sqrt(torch.clamp(ref_real_batch ** 2 + ref_imag_batch ** 2, min=0.0))
                blend_mag = torch.sqrt(torch.clamp(real_blend ** 2 + imag_blend ** 2, min=0.0))
                ref_mag = _nan_guard("ref_mag", ref_mag, clamp=1e4)
                blend_mag = _nan_guard("blend_mag", blend_mag, clamp=1e4)

                ref_log = log_mag(ref_mag)
                blend_log = log_mag(blend_mag)
                ref_log = _nan_guard("ref_log", ref_log, clamp=50.0)
                blend_log = _nan_guard("blend_log", blend_log, clamp=50.0)

                h_ref = soft_histogram(ref_log, bins=64, minv=float(ref_log.min().item()), maxv=float(ref_log.max().item()))
                h_blend = soft_histogram(blend_log, bins=64, minv=float(ref_log.min().item()), maxv=float(ref_log.max().item()))
                h_ref = _nan_guard("h_ref", h_ref)
                h_blend = _nan_guard("h_blend", h_blend)
                l_hist = F.mse_loss(h_blend, h_ref)

                lc_ref = local_contrast_map(ref_log)
                lc_blend = local_contrast_map(blend_log)
                lc_ref = _nan_guard("lc_ref", lc_ref)
                lc_blend = _nan_guard("lc_blend", lc_blend)
                l_contrast = F.l1_loss(lc_blend, lc_ref)

                tv_t = torch.mean(torch.abs(w[:, :, :, 1:] - w[:, :, :, :-1]))
                tv_f = torch.mean(torch.abs(w[:, :, 1:, :] - w[:, :, :-1, :]))
                l_tv = tv_t + tv_f

                loss = l_snr + args.lambda_hist * l_hist + args.lambda_contrast * l_contrast + args.tv * l_tv

            if not torch.isfinite(loss):
                print(f"[warn] non-finite loss detected at epoch {epoch+1} step {step+1}; skipping update")
                opt.zero_grad(set_to_none=True)
                segment_buffer.clear()
                continue

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(weight_net.parameters(), max_norm=5.0)
            opt.step()
            train_updates += 1

            wav_hat_np = wav_hat.detach().cpu().numpy()
            x_target_np = x_target.detach().cpu().numpy()

            if args.debug_check_ref:
                for idx, ref_entry in enumerate(ref_raw_np_batch):
                    if ref_entry is None:
                        continue
                    r_np, i_np, b_np, f_np = ref_entry
                    y_ref = c.istft(inverse_plan, r_np.astype(np.float32), i_np.astype(np.float32))
                    x_np = x_target_np[idx]
                    L = min(len(y_ref), len(x_np))
                    if L > 0:
                        err = np.mean((y_ref[:L] - x_np[:L]) ** 2)
                        sig = np.mean(x_np[:L] ** 2) + 1e-12
                        snr_ref = 10.0 * np.log10(sig / (err + 1e-12))
                        print(f"[debug] update {train_updates} seg {idx}: ref round-trip SNR {snr_ref:.2f} dB (bins={b_np}, frames={f_np}, out_len={L})")

            if train_updates % 10 == 0:
                print(f"epoch {epoch+1} step {step+1} update {train_updates}: loss={loss.item():.6f} snr={l_snr.item():.6f} hist={l_hist.item():.6f} contrast={l_contrast.item():.6f} tv={l_tv.item():.6f} batch={B}")

            if screen is not None and scroller is not None:
                try:
                    import pygame

                    F_total = ref_log.shape[1]
                    T_total = ref_log.shape[2]
                    fmin_hz = float(args.viz_fmin)
                    fmax_hz = float(args.viz_fmax)
                    nyq = sr * 0.5
                    if fmax_hz <= 0.0 or fmax_hz > nyq:
                        fmax_hz = nyq
                    if fmin_hz < 0.0:
                        fmin_hz = 0.0
                    kmin = int(round(fmin_hz * ref_N / sr))
                    kmax = int(round(fmax_hz * ref_N / sr))
                    kmin = max(0, min(F_total - 1, kmin))
                    kmax = max(kmin + 1, min(F_total, kmax))
                    tf = int(args.viz_time_frames)
                    t0 = max(0, T_total - tf) if tf > 0 else 0
                    t1 = T_total

                    last_idx = B - 1
                    ref_log_det = ref_log.detach()
                    blend_log_det = blend_log.detach()
                    weights_det = w.detach()
                    real_det = real_u_batch.detach()
                    imag_det = imag_u_batch.detach()

                    ref_log_crop = ref_log_det[last_idx][kmin:kmax, t0:t1]
                    blend_log_crop = blend_log_det[last_idx][kmin:kmax, t0:t1]

                    node_real = real_det[last_idx]
                    node_imag = imag_det[last_idx]
                    node_mag = torch.sqrt(torch.clamp(node_real ** 2 + node_imag ** 2, min=0.0))
                    node_log = torch.log1p(node_mag)
                    node_logs = [node_log[u][kmin:kmax, t0:t1] for u in range(node_log.shape[0])]

                    weights_slice = weights_det[last_idx][:, kmin:kmax, t0:t1]

                    scroller.update(ref_log_crop, blend_log_crop, node_logs, weights_slice)
                    scroller.render(screen)
                    pygame.display.flip()

                    for event in pygame.event.get():
                        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                            stop_training = True
                            pygame.quit()
                            screen = None
                            scroller = None
                            mixer_channel = None
                            if audio_streamer is not None:
                                audio_streamer.close()
                                audio_streamer = None
                            break
                        elif event.type == pygame.KEYDOWN and scroller is not None:
                            scroller.handle_key(event.key)
                    if stop_training:
                        break
                except Exception as _viz_ex:
                    print(f"[viz] disabled due to error: {_viz_ex}")
                    try:
                        import pygame
                        pygame.quit()
                    except Exception:
                        pass
                    screen = None
                    scroller = None
                    mixer_channel = None
                    if audio_streamer is not None:
                        audio_streamer.close()
                        audio_streamer = None

            if audio_streamer is not None and mixer_channel is not None and args.audio != "off":
                try:
                    import pygame
                    for idx in range(B):
                        y = wav_hat_np[idx]
                        x = x_target_np[idx]
                        if args.audio == "residual":
                            y = x - y
                        elif args.audio == "ref":
                            ref_entry = ref_raw_np_batch[idx]
                            if ref_entry is None:
                                ref_r_np = ref_real_batch[idx].detach().cpu().numpy().astype(np.float32)
                                ref_i_np = ref_imag_batch[idx].detach().cpu().numpy().astype(np.float32)
                                y_ref = c.istft(inverse_plan, ref_r_np, ref_i_np)
                            else:
                                r_np, i_np, b_np, f_np = ref_entry
                                plan_bins = inverse_plan.n // 2 + 1 if inverse_plan.half else inverse_plan.n
                                if b_np != plan_bins:
                                    r_t = torch.from_numpy(r_np)
                                    i_t = torch.from_numpy(i_np)
                                    r_t = _resize_2d(r_t.unsqueeze(0).unsqueeze(0), (plan_bins, f_np)).squeeze(0).squeeze(0)
                                    i_t = _resize_2d(i_t.unsqueeze(0).unsqueeze(0), (plan_bins, f_np)).squeeze(0).squeeze(0)
                                    r_np = r_t.numpy().astype(np.float32)
                                    i_np = i_t.numpy().astype(np.float32)
                                y_ref = c.istft(inverse_plan, r_np.astype(np.float32), i_np.astype(np.float32))
                            if y_ref.shape[0] >= x.shape[0]:
                                y = y_ref[: x.shape[0]]
                            else:
                                pad = x.shape[0] - y_ref.shape[0]
                                y = np.pad(y_ref, (0, pad), mode="constant")
                        audio_streamer.push(y)
                except Exception as _aud_ex:
                    print(f"[audio] disabled due to error: {_aud_ex}")
                    if audio_streamer is not None:
                        audio_streamer.close()
                        audio_streamer = None
                    mixer_channel = None

            del segment_buffer[:segment_batch_slide]

    # Save trained weights
    torch.save({
        "state_dict": weight_net.state_dict(),
        "Ns": Ns,
        "hops": hops,
        "ref_idx": ref_idx,
        "ref_N": ref_N,
        "ref_hop": ref_hop,
    }, args.save)
    print(f"Saved model to {args.save}")

    # Free contexts
    if audio_streamer is not None:
        audio_streamer.close()
    c.free_all()


if __name__ == "__main__":
    main()
