"""Visualize TorchSig sampling-clock drift models and their statistics.

Run from the TorchSig checkout with its editable environment. The script uses
the same sampling-position generator as the clock transforms and writes two
PNG files without opening an interactive window.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from torchsig.utils.dsp import SamplingPositions, generate_sampling_positions

MODELS = ("constant", "linear", "random_walk")
MIN_SAMPLES = 64
MIN_REALIZATIONS = 20
MIN_DRIFT_PPM = -1_000_000.0


def generate_positions(
    args: argparse.Namespace,
    model: str,
    seed: int,
) -> SamplingPositions:
    """Generate one jitter-free sampling-clock drift trajectory."""
    return generate_sampling_positions(
        num_input_samples=args.samples,
        taps_per_phase=1,
        uprate=args.uprate,
        drate=float(args.uprate),
        jitter_ppm=0.0,
        drift_ppm=args.drift_ppm,
        rng=np.random.default_rng(seed),
        drift_model=model,
        drift_rate_ppm_per_sec=args.drift_rate_ppm_per_sec,
        drift_noise_ppm_per_sqrt_sec=args.drift_noise_ppm_per_sqrt_sec,
        sample_rate_hz=args.sample_rate_hz,
    )


def period_error_ppm(positions: SamplingPositions, uprate: int) -> np.ndarray:
    """Recover period error from consecutive accepted sampling positions."""
    consecutive = np.diff(positions.nominal_indices) == 1
    return (np.diff(positions.actual)[consecutive] / uprate - 1.0) * 1e6


def plot_model_trajectories(
    trajectories: dict[str, SamplingPositions],
    args: argparse.Namespace,
) -> None:
    """Plot period and accumulated timing errors for all drift models."""
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    figure.suptitle("TorchSig sampling-clock drift models (jitter disabled)")

    for column, model in enumerate(MODELS):
        positions = trajectories[model]
        time_seconds = positions.nominal_indices / args.sample_rate_hz
        errors_ppm = period_error_ppm(positions, args.uprate)
        error_time = positions.nominal_indices[:-1] / args.sample_rate_hz

        axes[0, column].plot(error_time[: len(errors_ppm)], errors_ppm, lw=0.8)
        axes[0, column].set_title(f"{model}: instantaneous period error")
        axes[0, column].set_xlabel("Elapsed time (s)")
        axes[0, column].set_ylabel("ppm")
        axes[0, column].grid(alpha=0.25)

        axes[1, column].plot(time_seconds, positions.timing_error, lw=0.8)
        axes[1, column].set_title(f"{model}: accumulated timing error")
        axes[1, column].set_xlabel("Elapsed time (s)")
        axes[1, column].set_ylabel("Input-sample periods")
        axes[1, column].grid(alpha=0.25)

    figure.savefig(args.output_dir / "drift_model_trajectories.png", dpi=150)
    plt.close(figure)


def random_walk_ensemble(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect innovations and endpoint errors across seeded realizations."""
    innovations = []
    final_period_errors = np.empty(args.realizations)
    final_timing_errors = np.empty(args.realizations)

    for realization in range(args.realizations):
        positions = generate_positions(args, "random_walk", args.seed + realization)
        errors_ppm = period_error_ppm(positions, args.uprate)
        innovations.append(np.diff(errors_ppm))
        final_period_errors[realization] = errors_ppm[-1]
        final_timing_errors[realization] = positions.timing_error[-1]

    return np.concatenate(innovations), final_period_errors, final_timing_errors


def normal_pdf(x: np.ndarray, mean: float, standard_deviation: float) -> np.ndarray:
    """Evaluate a Gaussian density without adding a SciPy dependency."""
    if standard_deviation == 0.0:
        return np.zeros_like(x)
    normalized = (x - mean) / standard_deviation
    return np.exp(-0.5 * normalized**2) / (standard_deviation * np.sqrt(2.0 * np.pi))


def plot_random_walk_distributions(
    innovations: np.ndarray,
    final_period_errors: np.ndarray,
    final_timing_errors: np.ndarray,
    args: argparse.Namespace,
) -> None:
    """Plot random-walk innovation and endpoint ensemble distributions."""
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    figure.suptitle(f"Random-walk drift ensemble ({args.realizations} realizations)")

    innovation_sigma = args.drift_noise_ppm_per_sqrt_sec / np.sqrt(args.sample_rate_hz)
    innovation_grid = np.linspace(np.min(innovations), np.max(innovations), 400)
    axes[0].hist(innovations, bins=65, density=True, alpha=0.65, label="Empirical")
    axes[0].plot(
        innovation_grid,
        normal_pdf(innovation_grid, 0.0, innovation_sigma),
        label="Theoretical Gaussian",
    )
    axes[0].set_title("Period-error innovations")
    axes[0].set_xlabel("ppm per sample")
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=8)

    axes[1].hist(final_period_errors, bins=40, density=True)
    axes[1].axvline(args.drift_ppm, color="black", ls="--", label="Initial offset")
    axes[1].set_title("Final period error")
    axes[1].set_xlabel("ppm")
    axes[1].set_ylabel("Density")
    axes[1].legend(fontsize=8)

    axes[2].hist(final_timing_errors, bins=40, density=True)
    axes[2].set_title("Final accumulated timing error")
    axes[2].set_xlabel("Input-sample periods")
    axes[2].set_ylabel("Density")

    figure.savefig(args.output_dir / "random_walk_distributions.png", dpi=150)
    plt.close(figure)


def print_statistics(
    innovations: np.ndarray,
    final_period_errors: np.ndarray,
    final_timing_errors: np.ndarray,
    args: argparse.Namespace,
) -> None:
    """Print empirical random-walk statistics and theoretical expectations."""
    innovation_variance = args.drift_noise_ppm_per_sqrt_sec**2 / args.sample_rate_hz
    steps = args.samples - 2
    expected_final_period_variance = steps * innovation_variance
    expected_final_timing_variance = (
        innovation_variance * (steps - 1) * steps * (2 * steps - 1) / 6.0 * 1e-12
    )
    print(f"Innovation mean: {np.mean(innovations):.6g} ppm")
    print(
        "Innovation variance: "
        f"empirical={np.var(innovations, ddof=1):.6g}, theoretical={innovation_variance:.6g} ppm^2"
    )
    print(
        "Final period-error variance: "
        f"empirical={np.var(final_period_errors, ddof=1):.6g}, "
        f"theoretical~={expected_final_period_variance:.6g} ppm^2"
    )
    print(
        "Final timing-error variance: "
        f"empirical={np.var(final_timing_errors, ddof=1):.6g}, "
        f"theoretical~={expected_final_timing_variance:.6g} sample-periods^2"
    )


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--realizations", type=int, default=500)
    parser.add_argument("--drift-ppm", type=float, default=100.0)
    parser.add_argument("--drift-rate-ppm-per-sec", type=float, default=1000.0)
    parser.add_argument("--drift-noise-ppm-per-sqrt-sec", type=float, default=500.0)
    parser.add_argument("--sample-rate-hz", type=float, default=1000.0)
    parser.add_argument("--uprate", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("drift_diagnostics"))
    args = parser.parse_args()

    numeric_values = np.array(
        [args.drift_ppm, args.drift_rate_ppm_per_sec, args.drift_noise_ppm_per_sqrt_sec, args.sample_rate_hz]
    )
    if np.any(~np.isfinite(numeric_values)):
        parser.error("Drift parameters and sample rate must be finite")
    if args.samples < MIN_SAMPLES or args.realizations < MIN_REALIZATIONS or args.uprate < 1:
        parser.error("Require samples>=64, realizations>=20, and uprate>=1")
    if args.drift_ppm <= MIN_DRIFT_PPM:
        parser.error("drift-ppm must be greater than -1000000")
    if args.drift_noise_ppm_per_sqrt_sec < 0.0 or args.sample_rate_hz <= 0.0:
        parser.error("Require nonnegative drift noise and positive sample rate")
    return args


def main() -> None:
    """Generate drift-model trajectory and distribution diagnostics."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trajectories = {model: generate_positions(args, model, args.seed) for model in MODELS}
    plot_model_trajectories(trajectories, args)
    ensemble = random_walk_ensemble(args)
    plot_random_walk_distributions(*ensemble, args)
    print_statistics(*ensemble, args)
    print(f"Saved 2 PNGs in {args.output_dir}")


if __name__ == "__main__":
    main()
