#!/usr/bin/env python3
"""Synthesize a mono WAV file composed of randomly-weighted sine waves."""

from __future__ import annotations

import argparse
import math
import random
import struct
import wave
from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class Partial:
    """Represents a single sine component used in waveform synthesis."""

    frequency: float
    phase: float
    amplitude: float


def _build_partials(rng: random.Random, count: int, min_freq: float, max_freq: float) -> List[Partial]:
    if count <= 0:
        raise ValueError("component count must be positive")
    if max_freq <= min_freq:
        raise ValueError("maximum frequency must be greater than minimum frequency")

    weights = [rng.random() for _ in range(count)]
    total_weight = sum(weights)
    if total_weight == 0.0:
        weights = [1.0 for _ in range(count)]
        total_weight = float(count)

    partials: List[Partial] = []
    for weight in weights:
        frequency = rng.uniform(min_freq, max_freq)
        phase = rng.uniform(0.0, 2.0 * math.pi)
        amplitude = weight / total_weight
        partials.append(Partial(frequency=frequency, phase=phase, amplitude=amplitude))
    return partials


def _render_samples(
    sample_rate: int,
    duration: float,
    partials: Iterable[Partial],
    amplitude: float,
) -> List[int]:
    if duration <= 0.0:
        raise ValueError("duration must be positive")
    if sample_rate <= 0:
        raise ValueError("sample rate must be positive")
    if not 0.0 < amplitude <= 1.0:
        raise ValueError("amplitude must be in the range (0.0, 1.0]")

    sample_count = max(1, int(round(duration * sample_rate)))
    envelope_denom = max(sample_count - 1, 1)
    samples: List[int] = []
    for n in range(sample_count):
        t = float(n) / float(sample_rate)
        # Use a Hann window so the waveform starts and ends at zero, avoiding clicks.
        envelope = 0.5 * (1.0 - math.cos(2.0 * math.pi * float(n) / float(envelope_denom)))
        value = 0.0
        for partial in partials:
            value += partial.amplitude * math.sin(2.0 * math.pi * partial.frequency * t + partial.phase)
        value *= amplitude * envelope
        quantized = int(round(value * 32767.0))
        quantized = max(-32768, min(32767, quantized))
        samples.append(quantized)
    return samples


def synthesize(
    output_path: str,
    *,
    sample_rate: int,
    duration: float,
    components: int,
    amplitude: float,
    min_frequency: float,
    max_frequency: float,
    seed: int | None,
) -> None:
    rng = random.Random(seed)
    partials = _build_partials(rng, components, min_frequency, max_frequency)
    samples = _render_samples(sample_rate, duration, partials, amplitude)

    with wave.open(output_path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        payload = bytearray()
        for sample in samples:
            payload.extend(struct.pack("<h", sample))
        wav.writeframes(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="Destination WAV file path")
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Sample rate in Hz (default: %(default)s)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Duration of the generated audio in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--components",
        type=int,
        default=4,
        help="Number of sine components to blend (default: %(default)s)",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.9,
        help="Overall amplitude scaling in the range (0, 1] (default: %(default)s)",
    )
    parser.add_argument(
        "--min-frequency",
        type=float,
        default=55.0,
        help="Minimum component frequency in Hz (default: %(default)s)",
    )
    parser.add_argument(
        "--max-frequency",
        type=float,
        default=None,
        help="Maximum component frequency in Hz (default: 45% of Nyquist)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for the random number generator (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nyquist = args.sample_rate * 0.5
    max_frequency = args.max_frequency if args.max_frequency is not None else nyquist * 0.9
    synthesize(
        args.output,
        sample_rate=args.sample_rate,
        duration=args.duration,
        components=args.components,
        amplitude=args.amplitude,
        min_frequency=args.min_frequency,
        max_frequency=max_frequency,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
