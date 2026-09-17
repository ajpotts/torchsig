"""Measure phase displacement produced by TorchSIG's clock-jitter function.

A known complex tone is passed through ``F.clock_jitter`` and compared with
the zero-impairment output. The phase difference is converted to an equivalent
sampling-position shift.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from torchsig.transforms.functional import clock_jitter

MIN_TONE_MAGNITUDE = 0.5
MAX_EDGE_GUARD = 32


def generate_tone(num_samples: int, cycles_per_sample: float) -> np.ndarray:
    """Generate a unit-amplitude complex tone with a known phase slope."""
    sample_indices = np.arange(num_samples)
    return np.exp(2j * np.pi * cycles_per_sample * sample_indices).astype(
        np.complex64
    )


def measure_position_shift(
    reference: np.ndarray,
    impaired: np.ndarray,
    cycles_per_sample: float,
) -> np.ndarray:
    """Convert measured tone-phase differences to equivalent sample shifts."""
    valid = (
        (np.abs(reference) > MIN_TONE_MAGNITUDE)
        & (np.abs(impaired) > MIN_TONE_MAGNITUDE)
    )
    edge_guard = min(MAX_EDGE_GUARD, len(valid) // 4)
    if edge_guard > 0:
        valid[:edge_guard] = False
        valid[-edge_guard:] = False
    shift = np.full(len(reference), np.nan, dtype=np.float64)
    phase_difference = np.angle(impaired[valid] * np.conj(reference[valid]))
    shift[valid] = np.unwrap(phase_difference) / (
        2.0 * np.pi * cycles_per_sample
    )
    return shift


def plot_clock_jitter(
    num_samples: int = 1_000,
    seed: int = 42,
    cycles_per_sample: float = 0.03125,
) -> plt.Figure:
    """Measure and plot the actual clock-jitter transform output."""
    tone = generate_tone(num_samples, cycles_per_sample)
    reference = clock_jitter(
        data=tone,
        jitter_ppm=0.0,
        rng=np.random.default_rng(seed),
    )
    sample_indices = np.arange(num_samples)
    modest_values = (0.0, 10.0, 100.0, 1_000.0)
    exaggerated_values = (10_000.0, 100_000.0, 500_000.0, 1_000_000.0)

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for jitter_ppm in modest_values:
        impaired = clock_jitter(
            data=tone,
            jitter_ppm=jitter_ppm,
            rng=np.random.default_rng(seed),
        )
        shift = measure_position_shift(reference, impaired, cycles_per_sample)
        axes[0].plot(
            sample_indices,
            shift * 1e6,
            linewidth=1,
            label=f"{jitter_ppm:,.0f} PPM",
        )

    axes[0].set_title("Measured output for small jitter values")
    axes[0].set_ylabel("Equivalent shift (millionths of a sample)")
    axes[0].grid(visible=True, alpha=0.3)
    axes[0].legend(ncols=2)

    for jitter_ppm in exaggerated_values:
        impaired = clock_jitter(
            data=tone,
            jitter_ppm=jitter_ppm,
            rng=np.random.default_rng(seed),
        )
        shift = measure_position_shift(reference, impaired, cycles_per_sample)
        axes[1].plot(
            sample_indices,
            shift,
            linewidth=1,
            label=f"{jitter_ppm:,.0f} PPM",
        )

    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axes[1].set_title("Measured output for exaggerated jitter values")
    axes[1].set_xlabel("Output sample index")
    axes[1].set_ylabel("Equivalent shift (input-sample periods)")
    axes[1].grid(visible=True, alpha=0.3)
    axes[1].legend(ncols=2)

    figure.suptitle(
        "Clock jitter measured from the phase of F.clock_jitter() output"
    )
    figure.tight_layout()
    return figure


def main() -> None:
    """Display the plot or save it to a command-line output path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional output image path.")
    parser.add_argument("--num-samples", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.num_samples <= 0:
        parser.error("--num-samples must be positive")

    figure = plot_clock_jitter(args.num_samples, args.seed)
    if args.output is None:
        plt.show()
    else:
        figure.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved clock-jitter plot to {args.output}")


if __name__ == "__main__":
    main()
