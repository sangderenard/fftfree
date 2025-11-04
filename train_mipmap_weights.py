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
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cffi import FFI
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
        produced = int(
            self.lib.fft_execute_batched(
                plan.handle,
                self.ffi.cast("float *", pcm.ctypes.data),
                int(pcm.size),
                self.ffi.cast("float *", out_real.ctypes.data),
                self.ffi.cast("float *", out_imag.ctypes.data),
                self.ffi.cast("float *", out_mag.ctypes.data),
                1,  # pad last
                1,  # enable_backup
                0,
            )
        )
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
        produced = int(
            self.lib.fft_execute_complex_batched(
                plan.handle,
                self.ffi.cast("float *", in_real.ctypes.data),
                self.ffi.cast("float *", in_imag.ctypes.data),
                int(frames),
                self.ffi.cast("float *", out_frames.ctypes.data),
                1,
                1,
                0,
            )
        )
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
        for b in range(B):
            r_np = real[b].detach().cpu().to(torch.float32).numpy()
            i_np = imag[b].detach().cpu().to(torch.float32).numpy()
            wav = cffi.istft(inv_plan, r_np, i_np)
            outs.append(torch.from_numpy(wav))
        wav_pad = torch.nn.utils.rnn.pad_sequence(outs, batch_first=True)
        ctx.save_for_backward(torch.tensor([wav.shape[0] for wav in outs], dtype=torch.long))
        ctx.plans = plans
        ctx.spec_shape = (B, Freq, Frames)
        return wav_pad  # [B, T_out] (same T_out per-batch if frames constant)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        # Map grad_out (waveform) -> grad w.r.t. input complex spectrogram via adjoint STFT
        lengths, = ctx.saved_tensors
        cffi, inv_plan, fwd_plan = ctx.plans
        B, Freq, Frames = ctx.spec_shape
        grad_real = torch.zeros((B, Freq, Frames), dtype=torch.float32)
        grad_imag = torch.zeros((B, Freq, Frames), dtype=torch.float32)
        for b in range(B):
            g = grad_out[b, : lengths[b].item()].detach().cpu().to(torch.float32).numpy()
            r, i, _m, bins, frames = cffi.stft(fwd_plan, g)
            # Align to expected (Freq, Frames)
            r_t = torch.from_numpy(r)
            i_t = torch.from_numpy(i)
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
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda-hist", type=float, default=0.1)
    p.add_argument("--lambda-contrast", type=float, default=0.1)
    p.add_argument("--tv", type=float, default=1e-3, help="Total-variation regularization on weights")
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

    # Load audio (mono)
    sr, data = wavfile.read(args.input)
    if data.ndim > 1:
        data = data[:, 0]
    x_all = data.astype(np.float32)
    total_len = int(x_all.shape[0])
    if total_len < args.chunk:
        raise ValueError("Input WAV too short for a training chunk")

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
    if args.viz or args.audio != "off":
        try:
            import pygame
            pygame.init()
            if args.viz:
                screen = pygame.display.set_mode((args.viz_width, args.viz_height))
                pygame.display.set_caption("Ref / Blend / Weights (low freq bottom)")
            if args.audio != "off":
                try:
                    pygame.mixer.init(frequency=int(sr), size=-16, channels=1, buffer=1024)
                    mixer_channel = pygame.mixer.Channel(0)
                except Exception as mex:
                    print(f"[audio] mixer init failed: {mex}")
                    mixer_channel = None
        except Exception as ex:
            print(f"[viz] failed to init pygame: {ex}")
            screen = None
            mixer_channel = None

    # Small weight network
    U = len(Ns)
    weight_net = TinyWeightNet(U)
    opt = torch.optim.Adam(weight_net.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(args.epochs):
        for step in range(args.steps_per_epoch):
            # Random chunk
            start = np.random.randint(0, total_len - args.chunk + 1)
            seg = x_all[start:start + args.chunk]
            seg_t = torch.from_numpy(seg).unsqueeze(0)  # [1, T]

            # Compute per-config STFTs from C (no grad path through these)
            specs: List[Tuple[torch.Tensor, torch.Tensor]] = []  # (real, imag) tensors in each node's native grid
            frames_ref = None
            bins_ref = None
            ref_raw_np = None  # (real_np, imag_np, bins, frames) from direct A_ref(x)
            for u, plan in enumerate(forward_plans):
                r, i, m, bins, frames = c.stft(plan, seg)
                rt = torch.from_numpy(r)  # [F, T]
                it = torch.from_numpy(i)
                if u == ref_idx:
                    bins_ref, frames_ref = int(bins), int(frames)
                    # Keep an untouched copy of the exact reference analysis for audio 'ref'
                    ref_raw_np = (r.copy(), i.copy(), int(bins), int(frames))
                specs.append((rt, it))

            assert frames_ref is not None and bins_ref is not None

            # Canonical grid (reference plan dims)
            F_ref, T_ref = bins_ref, frames_ref
            real_stack = []
            imag_stack = []
            mag_stack = []
            for u, (rt, it) in enumerate(specs):
                if args.align == "resize":
                    # Naive resize of complex to canonical
                    if rt.shape != (F_ref, T_ref):
                        rt2 = _resize_2d(rt.unsqueeze(0).unsqueeze(0), (F_ref, T_ref)).squeeze(0).squeeze(0)
                        it2 = _resize_2d(it.unsqueeze(0).unsqueeze(0), (F_ref, T_ref)).squeeze(0).squeeze(0)
                    else:
                        rt2, it2 = rt, it
                else:
                    # Transport: synthesize with node inverse, analyze with reference forward
                    y_u = c.istft(inverse_plans[u], rt.numpy(), it.numpy())
                    r2, i2, m2, bins2, frames2 = c.stft(forward_ref, y_u)
                    rt2 = torch.from_numpy(r2)
                    it2 = torch.from_numpy(i2)
                    # Align time frames by crop/pad to T_ref (bins should match F_ref)
                    if bins2 != F_ref:
                        # If mismatch, resize bins via bilinear (rare)
                        rt2 = _resize_2d(rt2.unsqueeze(0).unsqueeze(0), (F_ref, frames2)).squeeze(0).squeeze(0)
                        it2 = _resize_2d(it2.unsqueeze(0).unsqueeze(0), (F_ref, frames2)).squeeze(0).squeeze(0)
                    if frames2 < T_ref:
                        pad = T_ref - frames2
                        rt2 = torch.cat([rt2, torch.zeros(F_ref, pad, dtype=rt2.dtype)], dim=1)
                        it2 = torch.cat([it2, torch.zeros(F_ref, pad, dtype=it2.dtype)], dim=1)
                    elif frames2 > T_ref:
                        rt2 = rt2[:, :T_ref]
                        it2 = it2[:, :T_ref]
                real_stack.append(rt2)
                imag_stack.append(it2)
                mag_stack.append(torch.sqrt(torch.clamp(rt2 ** 2 + it2 ** 2, min=0.0)))

            real_u = torch.stack(real_stack, dim=0).unsqueeze(0)  # [1, U, F, T]
            imag_u = torch.stack(imag_stack, dim=0).unsqueeze(0)
            mag_u = torch.stack(mag_stack, dim=0).unsqueeze(0)

            # Weighting
            with torch.enable_grad():
                mags_for_logits = log_mag(mag_u, k=1.0)
                w = weight_net(mags_for_logits)  # [1, U, F, T]
                # Complex blending (unit circle or Cartesian)
                if args.phase_blend == "unit":
                    eps = float(args.phase_eps)
                    beta = float(args.phase_mag_exp)
                    M_u = torch.clamp(mag_u, min=0.0)
                    cos_u = real_u / (M_u + eps)
                    sin_u = imag_u / (M_u + eps)
                    w_phase = w * torch.pow(M_u + eps, beta)
                    cos_mix = torch.sum(w_phase * cos_u, dim=1)  # [1, F, T]
                    sin_mix = torch.sum(w_phase * sin_u, dim=1)
                    phase_hat = torch.atan2(sin_mix, cos_mix)
                    if args.mag_combine == "powermean":
                        p = float(args.mag_p)
                        if abs(p) < 1e-6:
                            # Geometric mean approximation
                            M_hat = torch.exp(torch.sum(w * torch.log(M_u + eps), dim=1))
                        else:
                            M_hat = torch.pow(torch.sum(w * torch.pow(M_u + eps, p), dim=1) + 1e-12, 1.0 / p)
                    else:
                        M_hat = torch.sum(w * M_u, dim=1)
                    real_blend = M_hat * torch.cos(phase_hat)
                    imag_blend = M_hat * torch.sin(phase_hat)
                else:
                    # Cartesian sum (baseline)
                    real_blend = torch.sum(w * real_u, dim=1)  # [1, F, T]
                    imag_blend = torch.sum(w * imag_u, dim=1)

                # Inverse through C with autograd-enabled wrapper
                wav_hat = ISTFT_CFFI_Function.apply(real_blend, imag_blend, (c, inverse_plan, forward_ref))  # [1, T_out]

                # Target must match output length. Compute actual out_len from frames/hop/N
                out_len = (T_ref - 1) * ref_hop + ref_N
                x_target = seg_t[:, :out_len]

                # Losses
                l_snr = snr_loss(wav_hat, x_target)

                # Spectrogram perceptual: compare log-mag maps to reference original STFT on canonical grid
                ref_real, ref_imag = specs[ref_idx]
                if ref_real.shape != (F_ref, T_ref):
                    ref_real = _resize_2d(ref_real.unsqueeze(0).unsqueeze(0), (F_ref, T_ref)).squeeze(0).squeeze(0)
                    ref_imag = _resize_2d(ref_imag.unsqueeze(0).unsqueeze(0), (F_ref, T_ref)).squeeze(0).squeeze(0)
                ref_mag = torch.sqrt(torch.clamp(ref_real ** 2 + ref_imag ** 2, min=0.0)).unsqueeze(0)  # [1,F,T]
                blend_mag = torch.sqrt(torch.clamp(real_blend ** 2 + imag_blend ** 2, min=0.0))      # [1,F,T]

                ref_log = log_mag(ref_mag)
                blend_log = log_mag(blend_mag)

                # Histogram loss (soft KDE histograms)
                h_ref = soft_histogram(ref_log, bins=64, minv=float(ref_log.min().item()), maxv=float(ref_log.max().item()))
                h_blend = soft_histogram(blend_log, bins=64, minv=float(ref_log.min().item()), maxv=float(ref_log.max().item()))
                l_hist = F.mse_loss(h_blend, h_ref)

                # Local contrast loss
                lc_ref = local_contrast_map(ref_log)
                lc_blend = local_contrast_map(blend_log)
                l_contrast = F.l1_loss(lc_blend, lc_ref)

                # TV regularization on weights (encourage smoothness in TF)
                tv_t = torch.mean(torch.abs(w[:, :, :, 1:] - w[:, :, :, :-1]))
                tv_f = torch.mean(torch.abs(w[:, :, 1:, :] - w[:, :, :-1, :]))
                l_tv = tv_t + tv_f

                loss = l_snr + args.lambda_hist * l_hist + args.lambda_contrast * l_contrast + args.tv * l_tv

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if args.debug_check_ref and (step + 1) % 1 == 0 and ref_raw_np is not None:
                # Sanity-check reference round-trip SNR
                r_np, i_np, b_np, f_np = ref_raw_np
                y_ref = c.istft(inverse_plan, r_np.astype(np.float32), i_np.astype(np.float32))
                x_np = x_target.detach().cpu().squeeze(0).numpy()
                L = min(len(y_ref), len(x_np))
                if L > 0:
                    err = np.mean((y_ref[:L] - x_np[:L]) ** 2)
                    sig = np.mean(x_np[:L] ** 2) + 1e-12
                    snr_ref = 10.0 * np.log10(sig / (err + 1e-12))
                    print(f"[debug] ref round-trip SNR: {snr_ref:.2f} dB (bins={b_np}, frames={f_np}, out_len={L})")

            if (step + 1) % 10 == 0:
                print(f"epoch {epoch+1} step {step+1}: loss={loss.item():.6f} snr={l_snr.item():.6f} hist={l_hist.item():.6f} contrast={l_contrast.item():.6f} tv={l_tv.item():.6f}")

            # Update pygame window with interim spectrogram image (3 panels: ref / blend / weights)
            if screen is not None:
                try:
                    import pygame
                    # Select frequency crop and time window on canonical grid
                    F_ref = ref_log.shape[1]
                    T_ref = ref_log.shape[2]
                    fmin_hz = float(args.viz_fmin)
                    fmax_hz = float(args.viz_fmax)
                    nyq = sr * 0.5
                    if fmax_hz <= 0.0 or fmax_hz > nyq:
                        fmax_hz = nyq
                    if fmin_hz < 0.0:
                        fmin_hz = 0.0
                    kmin = int(round(fmin_hz * ref_N / sr))
                    kmax = int(round(fmax_hz * ref_N / sr))
                    kmin = max(0, min(F_ref - 1, kmin))
                    kmax = max(kmin + 1, min(F_ref, kmax))
                    # Time window
                    tf = int(args.viz_time_frames)
                    t0 = max(0, T_ref - tf) if tf > 0 else 0
                    t1 = T_ref

                    def to_u8_gray(ft: torch.Tensor) -> np.ndarray:
                        # ft: [F,T] torch -> uint8 HxW with 0..255, low freq at bottom
                        v = ft.detach().cpu()
                        v = v - v.min()
                        denom = float(v.max().item()) if v.numel() > 0 else 0.0
                        if denom <= 1e-12:
                            v = torch.zeros_like(v)
                        else:
                            v = (v / denom) * 255.0
                        img = v.numpy().astype(np.uint8)
                        img = np.flipud(img)  # low freq at bottom
                        return img  # [H=F, W=T]

                    def to_u8_rgb(real_ft: torch.Tensor, imag_ft: torch.Tensor) -> np.ndarray:
                        # real_ft/imag_ft: [F,T]; map real/imag linearly per-channel; B=mag
                        def _norm01(x: torch.Tensor) -> torch.Tensor:
                            xx = x.detach().cpu()
                            mn = xx.min()
                            rng = xx.max() - mn
                            return torch.zeros_like(xx) if float(rng.item()) <= 1e-12 else (xx - mn) / (rng + 1e-12)
                        R = _norm01(real_ft)
                        I = _norm01(imag_ft)
                        M = torch.sqrt(torch.clamp(real_ft**2 + imag_ft**2, min=0.0))
                        # log1p for magnitude then normalize
                        M = torch.log1p(M)
                        M = _norm01(M)
                        r = (R.numpy() * 255.0).astype(np.uint8)
                        g = (I.numpy() * 255.0).astype(np.uint8)
                        b = (M.numpy() * 255.0).astype(np.uint8)
                        # Flip vertically (low freq bottom)
                        r = np.flipud(r)
                        g = np.flipud(g)
                        b = np.flipud(b)
                        rgb = np.stack([r.T, g.T, b.T], axis=2)  # (W,H,3)
                        return rgb

                    # Prepare three panels (apply crop)
                    if args.viz_mode == "rgb":
                        # ref: use ref_real/imag; blend: use real_blend/imag_blend
                        ref_r = ref_real[kmin:kmax, t0:t1]
                        ref_i = ref_imag[kmin:kmax, t0:t1]
                        blend_r = real_blend.squeeze(0)[kmin:kmax, t0:t1]
                        blend_i = imag_blend.squeeze(0)[kmin:kmax, t0:t1]
                        ref_rgb = to_u8_rgb(ref_r, ref_i)
                        blend_rgb = to_u8_rgb(blend_r, blend_i)
                    else:
                        ref_c = ref_log[:, kmin:kmax, t0:t1].squeeze(0)
                        blend_c = blend_log[:, kmin:kmax, t0:t1].squeeze(0)
                        ref_gray = to_u8_gray(ref_c)    # [H,W]
                        blend_gray = to_u8_gray(blend_c)
                    # Weights heatmap: average over frequency -> [U,T]
                    w_cpu = w.detach().cpu().squeeze(0)        # [U,F,T]
                    w_crop = w_cpu[:, kmin:kmax, t0:t1]
                    w_mean = w_crop.mean(dim=1)                 # [U,T]
                    # Normalize across all nodes/time for a common color scale
                    wv = w_mean - w_mean.min()
                    wd = float(wv.max().item()) if wv.numel() > 0 else 0.0
                    if wd <= 1e-12:
                        w_u8 = (wv * 0).numpy().astype(np.uint8)
                    else:
                        w_u8 = (wv / wd * 255.0).numpy().astype(np.uint8)
                    # Expand each node row to a small height block
                    U = w_u8.shape[0]
                    per_row = max(1, args.viz_height // 12)  # heuristic row height
                    rows = [np.tile(w_u8[u:u+1, :], (per_row, 1)) for u in range(U)]
                    weights_vis = np.vstack(rows)  # [U*per_row, T]
                    # Convert each panel to RGB surface scaled to thirds
                    def make_surface(img: np.ndarray, width: int, height: int) -> pygame.Surface:
                        # Accept either [H,W] uint8 or [W,H,3] uint8
                        if img.ndim == 2:
                            g = np.ascontiguousarray(img)
                            rgb = np.stack([g.T, g.T, g.T], axis=2)
                            rgb = np.ascontiguousarray(rgb)
                            surf = pygame.surfarray.make_surface(rgb)
                        else:
                            surf = pygame.surfarray.make_surface(np.ascontiguousarray(img))
                        return pygame.transform.smoothscale(surf, (width, height))

                    W = args.viz_width
                    H = args.viz_height
                    h_third = H // 3
                    if args.viz_mode == "rgb":
                        ref_s = make_surface(ref_rgb, W, h_third)
                        blend_s = make_surface(blend_rgb, W, h_third)
                    else:
                        ref_s = make_surface(ref_gray, W, h_third)
                        blend_s = make_surface(blend_gray, W, h_third)
                    w_s = make_surface(weights_vis, W, H - 2 * h_third)
                    screen.blit(ref_s, (0, 0))
                    screen.blit(blend_s, (0, h_third))
                    screen.blit(w_s, (0, 2 * h_third))

                    # Overlay labels and per-node specs on bottom panel
                    try:
                        font = pygame.font.SysFont(None, 16)
                        label = font.render("Ref / Blend / Weights ({} mode)".format(args.viz_mode.upper()), True, (255, 255, 0))
                        screen.blit(label, (6, 6))
                        # Specs for each node along the left of bottom panel
                        y0 = 2 * h_third
                        row_h = (H - 2 * h_third) / max(1, w_u8.shape[0])
                        for idx, (n, hhop) in enumerate(zip(Ns, hops)):
                            y = int(y0 + idx * row_h + 1)
                            txt = font.render(f"N={n} H={hhop}", True, (200, 200, 200))
                            screen.blit(txt, (6, y))
                    except Exception:
                        pass

                    pygame.display.flip()
                    # Basic event pump to keep window responsive
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            screen = None
                            pygame.quit()
                            break
                except Exception as _viz_ex:
                    # Disable viz on first failure
                    print(f"[viz] disabled due to error: {_viz_ex}")
                    try:
                        import pygame
                        pygame.quit()
                    except Exception:
                        pass
                    screen = None

            # Optional audio monitoring
            if mixer_channel is not None and args.audio != "off":
                try:
                    import pygame
                    # Extract numpy arrays
                    y = wav_hat.detach().cpu().squeeze(0).numpy()  # blend reconstruction
                    x = x_target.detach().cpu().squeeze(0).numpy()  # reference-length target
                    if args.audio == "residual":
                        y = x - y
                    elif args.audio == "ref":
                        # Reconstruct reference from the exact forward output of A_ref(x)
                        if ref_raw_np is None:
                            # Fallback: use current tensors if not captured (should not happen)
                            ref_r_np = ref_real.detach().cpu().numpy().astype(np.float32)
                            ref_i_np = ref_imag.detach().cpu().numpy().astype(np.float32)
                            y_ref = c.istft(inverse_plan, ref_r_np, ref_i_np)
                        else:
                            r_np, i_np, b_np, f_np = ref_raw_np
                            # Sanity: bins should match inverse plan bins
                            plan_bins = inverse_plan.n // 2 + 1 if inverse_plan.half else inverse_plan.n
                            if b_np != plan_bins:
                                # Resize safely if mismatch (defensive; should not occur)
                                r_t = torch.from_numpy(r_np)
                                i_t = torch.from_numpy(i_np)
                                r_t = _resize_2d(r_t.unsqueeze(0).unsqueeze(0), (plan_bins, f_np)).squeeze(0).squeeze(0)
                                i_t = _resize_2d(i_t.unsqueeze(0).unsqueeze(0), (plan_bins, f_np)).squeeze(0).squeeze(0)
                                r_np = r_t.numpy().astype(np.float32)
                                i_np = i_t.numpy().astype(np.float32)
                            y_ref = c.istft(inverse_plan, r_np.astype(np.float32), i_np.astype(np.float32))
                        # Match target length
                        if y_ref.shape[0] >= x.shape[0]:
                            y = y_ref[: x.shape[0]]
                        else:
                            pad = x.shape[0] - y_ref.shape[0]
                            y = np.pad(y_ref, (0, pad), mode="constant")
                    # Normalize to int16
                    mx = np.max(np.abs(y)) if y.size else 0.0
                    if mx > 0:
                        y = y / mx
                    y = np.clip(y * float(args.audio_gain), -1.0, 1.0)
                    y16 = (y * 32767.0).astype(np.int16)
                    init = pygame.mixer.get_init()
                    ch = int(init[2]) if init else 1
                    arr = y16.copy()
                    if ch > 1:
                        # Expand mono -> stereo/ multichannel by duplication
                        arr = np.repeat(arr[:, None], ch, axis=1)
                    snd = pygame.sndarray.make_sound(arr)
                    mixer_channel.play(snd)
                except Exception as _aud_ex:
                    print(f"[audio] disabled due to error: {_aud_ex}")
                    mixer_channel = None

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
    c.free_all()


if __name__ == "__main__":
    main()
