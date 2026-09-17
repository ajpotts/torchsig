"""Measure phase displacement produced by TorchSIG's clock-drift function.

A known complex tone is passed through ``F.clock_drift`` and compared with the
zero-impairment output. This directly measures the public function, including
polyphase resampling and output-length normalization.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from torchsig.transforms.functional import clock_drift

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

    # F.clock_drift() crops or pads its raw resampler output back to the input
    # length. Remove that constant alignment offset at the center so the plot
    # isolates the rate-induced change in phase across the signal.
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) > 0:
        center_idx = valid_indices[np.argmin(np.abs(valid_indices - len(shift) // 2))]
        shift -= shift[center_idx]
    return shift


def drift_label(drift_ppm: float) -> str:
    """Format a signed PPM value for a plot legend."""
    return "0 PPM" if drift_ppm == 0.0 else f"{drift_ppm:+,.0f} PPM"


def plot_clock_drift(
    num_samples: int = 1_000,
    cycles_per_sample: float = 0.03125,
) -> plt.Figure:
    """Measure and plot the actual clock-drift transform output."""
    tone = generate_tone(num_samples, cycles_per_sample)
    reference = clock_drift(data=tone, drift_ppm=0.0)
    sample_indices = np.arange(num_samples)
    modest_values = (-1_000.0, -100.0, -10.0, 0.0, 10.0, 100.0, 1_000.0)
    exaggerated_values = (
        -500_000.0,
        -100_000.0,
        -10_000.0,
        10_000.0,
        100_000.0,
        500_000.0,
    )

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for drift_ppm in modest_values:
        impaired = clock_drift(data=tone, drift_ppm=drift_ppm)
        shift = measure_position_shift(reference, impaired, cycles_per_sample)
        axes[0].plot(
            sample_indices,
            shift * 1e6,
            linewidth=1.5,
            label=drift_label(drift_ppm),
        )

    axes[0].set_title("Measured output for small fixed rate offsets")
    axes[0].set_ylabel("Equivalent shift (millionths of a sample)")
    axes[0].grid(visible=True, alpha=0.3)
    axes[0].legend(ncols=4)

    for drift_ppm in exaggerated_values:
        impaired = clock_drift(data=tone, drift_ppm=drift_ppm)
        shift = measure_position_shift(reference, impaired, cycles_per_sample)
        axes[1].plot(
            sample_indices,
            shift,
            linewidth=1.5,
            label=drift_label(drift_ppm),
        )

    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axes[1].set_title("Measured output for exaggerated fixed rate offsets")
    axes[1].set_xlabel("Output sample index")
    axes[1].set_ylabel("Equivalent shift (input-sample periods)")
    axes[1].grid(visible=True, alpha=0.3)
    axes[1].legend(ncols=3)

    figure.suptitle(
        "Clock drift measured from the phase of F.clock_drift() output"
    )
    figure.tight_layout()
    return figure


def main() -> None:
    """Display the plot or save it to a command-line output path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional output image path.")
    parser.add_argument("--num-samples", type=int, default=1_000)
    args = parser.parse_args()

    if args.num_samples <= 0:
        parser.error("--num-samples must be positive")

    figure = plot_clock_drift(args.num_samples)
    if args.output is None:
        plt.show()
    else:
        figure.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved clock-drift plot to {args.output}")


if __name__ == "__main__":
    main()
