
# pygame_fft_viz.py
# Minimal dependency: numpy, pygame
# Usage example:
#   from pygame_fft_viz import animate_from_flat
#   animate_from_flat(out_real, out_imag, frames, N_eff, sample_rate, hop=STFT_H)
#
# Or:
#   from pygame_fft_viz import PygameFFTViz
#   viz = PygameFFTViz(X_complex, sample_rate, hop); viz.run()
#
# Controls:
#   ESC / Q  -> quit
#   SPACE    -> pause/resume
#   [ / ]    -> decrease/increase draw decimation
#   - / =    -> slower / faster playback
#   1..6     -> choose reference band preset (auto / bass / low-mid / mid / high / very-high)
#   A        -> toggle auto band-seek
#   G        -> toggle ghosting
#   R        -> reset scale
#   H        -> toggle help overlay

from __future__ import annotations
import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import pygame
except Exception as e:
    raise RuntimeError("pygame is required. Install with: pip install pygame") from e


def _principal_angle(delta: float) -> float:
    # Map any angle to (-pi, pi]
    return (delta + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class VizParams:
    width: int = 1280
    height: int = 720
    top_px: int = 110               # height of phasor strip
    bg_color: Tuple[int, int, int] = (8, 10, 12)
    phasor_color: Tuple[int, int, int] = (230, 235, 240)
    whip_color: Tuple[int, int, int] = (120, 200, 255)
    ghost_alpha_top: int = 28       # 0..255 alpha for trail fade per frame (top layer)
    ghost_alpha_whip: int = 22      # 0..255 (bottom layer)
    gamma: float = 0.5              # magnitude compression
    speed: float = 1.0              # 1.0 == realtime (fps ~= fs/hop)
    decimate: int = 4               # draw every Nth bin / point
    target_radius_frac: float = 0.70  # whip radius target as fraction of bottom min(width,height)
    scale_ema: float = 0.12         # smoothing for auto scale updates
    theta_ema: float = 0.15         # smoothing for instantaneous frequency / angle
    max_fps: int = 120
    show_help: bool = True
    auto_band_seek: bool = True     # automatically select best band by coherence
    min_ref_hz: float = 50.0        # ignore DC / sub-bass for auto-seek (Hz)
    max_ref_hz: float = None        # None => Nyquist


class PygameFFTViz:
    def __init__(self,
                 X_complex: np.ndarray,
                 sample_rate: float,
                 hop: int,
                 params: VizParams = VizParams(),
                 band_hz: Optional[Tuple[float, float]] = None):
        # X_complex: np.ndarray of shape (frames, N_bins) complex64/complex128
        # sample_rate: input audio sample rate (Hz)
        # hop: STFT hop size (samples) => nominal frames per second = fs / hop
        # band_hz: optional (low_hz, high_hz) for spin-lock reference
        #          If None, and params.auto_band_seek=True, picks best band per-frame.
        if X_complex.ndim != 2:
            raise ValueError(f"X_complex must be 2D (frames, bins); got shape {X_complex.shape}")
        self.X = np.asarray(X_complex)
        self.frames, self.N = self.X.shape
        self.fs = float(sample_rate)
        self.hop = int(hop)
        self.params = params
        self.band_hz = band_hz

        # State
        self._paused = False
        self._ghosting = True
        self._decimate = max(1, int(self.params.decimate))
        self._speed = float(self.params.speed)
        self._theta_prev = None
        self._theta_acc = 0.0
        self._scale = 1.0
        self._scale_ready = False

        # Derived
        self._bins_hz = (np.arange(self.N) * self.fs) / self.N
        self._k_min_auto = int(np.ceil(self.params.min_ref_hz * self.N / self.fs))
        self._k_max_auto = int(self.N // 2) if self.params.max_ref_hz is None else int(
            min(self.N // 2, math.floor(self.params.max_ref_hz * self.N / self.fs))
        )

    # ---------- Band / reference helpers ----------

    def _band_indices(self, band_hz: Tuple[float, float]) -> np.ndarray:
        lo, hi = band_hz
        lo_k = max(0, int(math.floor(lo * self.N / self.fs)))
        hi_k = min(self.N - 1, int(math.ceil(hi * self.N / self.fs)))
        if hi_k <= lo_k:
            hi_k = min(self.N - 1, lo_k + 1)
        return np.arange(lo_k, hi_k + 1, dtype=np.int32)

    def _choose_auto_band(self, Xf: np.ndarray) -> Tuple[np.ndarray, float]:
        # Pick the strongest single bin in [k_min_auto, k_max_auto]
        k0 = int(np.argmax(np.abs(Xf[self._k_min_auto:self._k_max_auto+1])) + self._k_min_auto)
        # Use a narrow band around it (± one bin broadened at high bins)
        width = max(1, int(1 + 0.002 * k0))  # slightly wider at higher freq
        k_lo = max(self._k_min_auto, k0 - width)
        k_hi = min(self._k_max_auto, k0 + width)
        band = np.arange(k_lo, k_hi + 1, dtype=np.int32)
        # Coherence metric rho
        R = np.sum(Xf[band])
        denom = np.sum(np.abs(Xf[band])) + 1e-12
        rho = float(np.abs(R) / denom) if denom > 0 else 0.0
        return band, rho

    def _spin_lock_rotation(self, Xf: np.ndarray) -> Tuple[complex, float]:
        # Determine reference band
        if self.params.auto_band_seek and self.band_hz is None:
            band, rho = self._choose_auto_band(Xf)
        else:
            band = self._band_indices(self.band_hz) if self.band_hz is not None else np.arange(
                self._k_min_auto, self._k_max_auto + 1, dtype=np.int32
            )
            R = np.sum(Xf[band])
            denom = np.sum(np.abs(Xf[band])) + 1e-12
            rho = float(np.abs(R) / denom) if denom > 0 else 0.0

        R = np.sum(Xf[band])
        phi = float(np.angle(R)) if R != 0 else 0.0

        # Unwrapped angle accumulator for smooth rotation
        if self._theta_prev is None:
            self._theta_prev = phi
            self._theta_acc = phi
        else:
            d = _principal_angle(phi - self._theta_prev)
            # Low-pass angle increments (PLL-ish smoothing)
            d = (1.0 - self.params.theta_ema) * 0.0 + self.params.theta_ema * d
            self._theta_acc += d
            self._theta_prev = phi

        rot = complex(math.cos(-self._theta_acc), math.sin(-self._theta_acc))
        return rot, rho

    # ---------- Drawing helpers ----------

    def _fade_surface(self, surf: "pygame.Surface", alpha: int) -> None:
        if alpha <= 0:
            return
        # Overlay a translucent black rect to darken existing pixels => ghosting
        fade_rect = pygame.Surface(surf.get_size(), flags=pygame.SRCALPHA)
        fade_rect.fill((0, 0, 0, alpha))
        surf.blit(fade_rect, (0, 0))

    def _draw_phasor_strip(self, surf: "pygame.Surface", Xf_rot: np.ndarray) -> None:
        h = surf.get_height()
        w = surf.get_width()
        mid_y = h // 2
        step = self._decimate
        # Precompute x positions
        xs = np.linspace(0, w, num=self.N, endpoint=False, dtype=np.float32)[::step]
        # Lengths (compressed) and angles
        mags = np.power(np.abs(Xf_rot), self.params.gamma)[::step]
        phis = np.angle(Xf_rot)[::step]

        # Normalize lengths to top strip height
        if mags.size > 0:
            mmax = float(np.percentile(mags, 99.0)) + 1e-12
        else:
            mmax = 1.0
        scale = 0.45 * h / mmax

        color = self.params.phasor_color
        for i in range(min(xs.size, mags.size)):
            length = float(mags[i] * scale)
            ang = float(phis[i])
            dx = length * math.cos(ang)
            dy = length * math.sin(ang)
            x0 = int(xs[i])
            y0 = int(mid_y)
            x1 = int(x0 + dx)
            y1 = int(y0 - dy)
            pygame.draw.line(surf, color, (x0, y0), (x1, y1), width=1)

    def _draw_whip(self, surf: "pygame.Surface", Xf_rot: np.ndarray) -> None:
        w = surf.get_width()
        h = surf.get_height()
        cx, cy = w // 2, h // 2
        step = self._decimate

        # Cumulative complex sum over bins (low -> high)
        C = np.cumsum(Xf_rot)
        C = C[::step]
        if C.size < 2:
            return

        # Auto scale to target radius
        radii = np.abs(C)
        r95 = float(np.percentile(radii, 95.0)) + 1e-9
        target_radius = self.params.target_radius_frac * min(w, h) * 0.5
        scale_now = target_radius / r95
        if not self._scale_ready:
            self._scale = scale_now
            self._scale_ready = True
        else:
            self._scale = (1.0 - self.params.scale_ema) * self._scale + self.params.scale_ema * scale_now

        pts = np.empty((C.size, 2), dtype=np.int32)
        pts[:, 0] = (cx + (C.real * self._scale)).astype(np.int32)
        pts[:, 1] = (cy - (C.imag * self._scale)).astype(np.int32)

        color = self.params.whip_color
        pygame.draw.lines(surf, color, False, pts.tolist(), 2)

    def _draw_help(self, screen: "pygame.Surface") -> None:
        if not self.params.show_help:
            return
        font = pygame.font.SysFont("consolas,monospace", 14)
        lines = [
            "[ESC/Q] quit   [SPACE] pause   [-/=] slower/faster   [</>] decimate   [A] auto-band   [G] ghost   [1..6] band preset   [H] help",
            f"frames={self.frames} N={self.N} fs={self.fs:.1f}Hz hop={self.hop}  fps_target≈{self.fs/self.hop:.1f}  speed×{self._speed:.2f}  decim={self._decimate}",
        ]
        y = 6
        for text in lines:
            img = font.render(text, True, (200, 200, 210))
            screen.blit(img, (10, y))
            y += img.get_height() + 2

    # ---------- Main loop ----------

    def run(self) -> None:
        pygame.init()
        pygame.display.set_caption("FFT Whip Visualizer (spin-locked)")
        screen = pygame.display.set_mode((self.params.width, self.params.height))
        clock = pygame.time.Clock()

        # Layers for ghosting
        top_rect = pygame.Rect(0, 0, self.params.width, self.params.top_px)
        bot_rect = pygame.Rect(0, self.params.top_px, self.params.width, self.params.height - self.params.top_px)

        top_layer = pygame.Surface((top_rect.w, top_rect.h), flags=pygame.SRCALPHA)
        whip_layer = pygame.Surface((bot_rect.w, bot_rect.h), flags=pygame.SRCALPHA)

        idx = 0
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self._paused = not self._paused
                    elif event.key == pygame.K_MINUS:
                        self._speed = max(0.05, self._speed * 0.8)
                    elif event.key == pygame.K_EQUALS:
                        self._speed = min(8.0, self._speed * 1.25)
                    elif event.key == pygame.K_LEFTBRACKET:
                        self._decimate = min(max(1, self._decimate - 1), self.N)
                    elif event.key == pygame.K_RIGHTBRACKET:
                        self._decimate = min(max(1, self._decimate + 1), self.N)
                    elif event.key == pygame.K_r:
                        self._scale_ready = False
                    elif event.key == pygame.K_g:
                        self._ghosting = not self._ghosting
                    elif event.key == pygame.K_h:
                        self.params.show_help = not self.params.show_help
                    elif event.key == pygame.K_a:
                        self.params.auto_band_seek = not self.params.auto_band_seek
                    elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6):
                        # Band presets (roughly): auto / bass / low-mid / mid / high / very-high
                        preset = event.key - pygame.K_1
                        if preset == 0:
                            self.band_hz = None  # auto
                            self.params.auto_band_seek = True
                        else:
                            self.params.auto_band_seek = False
                            nyq = 0.5 * self.fs
                            # Define 5 bands across log space
                            edges = np.geomspace(40.0, nyq, num=7)  # 6 intervals
                            lo = float(edges[preset])
                            hi = float(edges[preset + 1])
                            self.band_hz = (lo, hi)

            if self._paused:
                clock.tick(self.params.max_fps)
                continue

            # Clear background
            screen.fill(self.params.bg_color)

            # Advance frame index according to speed
            idx_int = int(idx) % self.frames
            Xf = self.X[idx_int]

            # Spin lock rotation
            rot, rho = self._spin_lock_rotation(Xf)
            Xf_rot = Xf * rot

            # Fade layers for ghosting
            if self._ghosting:
                self._fade_surface(top_layer, self.params.ghost_alpha_top)
                self._fade_surface(whip_layer, self.params.ghost_alpha_whip)
            else:
                top_layer.fill((0, 0, 0, 0))
                whip_layer.fill((0, 0, 0, 0))

            # Draw top phasor strip
            self._draw_phasor_strip(top_layer, Xf_rot)
            # Draw whip in bottom area
            self._draw_whip(whip_layer, Xf_rot)

            # Blit layers
            screen.blit(top_layer, top_rect.topleft)
            screen.blit(whip_layer, bot_rect.topleft)

            # HUD / help text
            self._draw_help(screen)

            pygame.display.flip()

            # Timing: nominal frames per second = fs / hop
            target_fps = (self.fs / self.hop) * self._speed
            # Clamp to max_fps to avoid burning CPU
            capped_fps = min(self.params.max_fps, max(1, int(target_fps)))
            clock.tick_busy_loop(capped_fps)
            idx += target_fps / max(1.0, self.params.max_fps)

        pygame.quit()


def animate_from_flat(out_real: np.ndarray,
                      out_imag: np.ndarray,
                      frames: int,
                      N_bins: int,
                      sample_rate: float,
                      hop: int,
                      **kwargs) -> None:
    """
    Convenience wrapper for flattened arrays (frames*N) like those produced by fft_to_png.py.

    Example integration (after computing out_real/out_imag/out_mag in your script):
        from pygame_fft_viz import animate_from_flat
        animate_from_flat(out_real, out_imag, frames, N_eff, sample_rate=_sample_rate, hop=STFT_H)

    kwargs are forwarded into VizParams() or band_hz, e.g.:
        animate_from_flat(..., width=1600, height=900, band_hz=(80, 200))
    """
    out_real = np.asarray(out_real, dtype=np.float32).reshape(frames, N_bins)
    out_imag = np.asarray(out_imag, dtype=np.float32).reshape(frames, N_bins)
    X = out_real.astype(np.float32) + 1j * out_imag.astype(np.float32)

    # Split kwargs: VizParams fields vs band_hz
    params_fields = set(VizParams().__dict__.keys())
    params_kwargs = {k: v for k, v in kwargs.items() if k in params_fields}
    other_kwargs = {k: v for k, v in kwargs.items() if k not in params_fields}

    params = VizParams(**params_kwargs)
    band_hz = other_kwargs.get("band_hz", None)

    viz = PygameFFTViz(X, sample_rate=sample_rate, hop=hop, params=params, band_hz=band_hz)
    viz.run()
