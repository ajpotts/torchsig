# ruff: noqa: INP001, S101
"""Verify that the sampling-position refactor preserves legacy IQ output.

Run from the repository root with::

    python scripts/verify_sampling_clock_refactor_equivalence.py
"""

from dataclasses import dataclass

import numpy as np

from torchsig.utils.dsp import polyphase_resample_at_positions, sampling_clock_impairments


@dataclass(frozen=True)
class Example:
    """One deterministic sampling-clock comparison case."""

    name: str
    h: np.ndarray
    x: np.ndarray
    uprate: int
    drate: float
    jitter_ppm: float
    drift_ppm: float
    initial_phase: float
    seed: int


def previous_sampling_clock_impairments(
    h: np.ndarray,
    x: np.ndarray,
    uprate: int,
    drate: float,
    jitter_ppm: float,
    drift_ppm: float,
    rng: np.random.Generator,
    initial_phase: float = 0.0,
) -> np.ndarray:
    """Run the pre-refactor sampling-position loop."""
    taps_per_phase = int(np.ceil(len(h) / uprate))
    padded_len = len(x) + 2 * taps_per_phase - 1
    max_input_idx = padded_len - taps_per_phase
    max_sample_position = max_input_idx * uprate + (uprate - 1)
    nominal_increment = drate * (1.0 + drift_ppm * 1e-6)
    nominal_position = uprate / drate + initial_phase * uprate
    position_offset = 0.0
    positions: list[float] = []
    clock_drift = 0.0
    jitter_std = jitter_ppm * 1e-6
    drift_std = drift_ppm * 1e-6

    while nominal_position <= max_sample_position:
        sample_position = nominal_position + position_offset
        if 0.0 <= sample_position <= max_sample_position:
            positions.append(sample_position)

        if jitter_ppm != 0.0 or drift_ppm != 0.0:
            clock_jitter = rng.normal(0.0, jitter_std)
            clock_drift += rng.normal(0.0, drift_std)
            position_offset += clock_jitter + clock_drift

        nominal_position += nominal_increment

    return polyphase_resample_at_positions(
        h=h,
        x=x,
        uprate=uprate,
        positions=np.asarray(positions, dtype=np.float64),
    )


def examples() -> list[Example]:
    """Build examples spanning signal, filter, rate, and impairment choices."""
    rng = np.random.default_rng(2026)
    random_iq = (rng.standard_normal(257) + 1j * rng.standard_normal(257)).astype(np.complex64)
    tone = np.exp(2j * np.pi * 0.137 * np.arange(193)).astype(np.complex64)
    impulse = np.zeros(65, dtype=np.complex64)
    impulse[0] = 1.0 + 0.5j

    return [
        Example("zero impairments", np.array([1.0], np.float32), random_iq, 1, 1.0, 0.0, 0.0, 0.0, 1),
        Example("jitter only", np.array([1.0, 0.5, -0.25, 0.125], np.float32), random_iq, 4, 4.0, 250_000.0, 0.0, 0.0, 2),
        Example("drift only", np.hanning(24).astype(np.float32), tone, 8, 8.0, 0.0, 50_000.0, 0.0, 3),
        Example("jitter and drift", np.hanning(32).astype(np.float32), random_iq, 8, 7.5, 100_000.0, 20_000.0, 0.0, 4),
        Example("fractional phase", np.hanning(16).astype(np.float32), tone, 4, 3.25, 10_000.0, 5_000.0, 0.375, 5),
        Example("short impulse", np.array([0.5, 0.25, 0.125], np.float32), impulse, 2, 1.5, 5_000.0, 0.0, 0.75, 6),
        Example("empty input", np.array([1.0], np.float32), np.array([], np.complex64), 1, 1.0, 0.0, 0.0, 0.0, 7),
    ]


def main() -> None:
    """Run every example and assert that legacy and refactored IQ agree."""
    for example in examples():
        kwargs = {
            "h": example.h,
            "x": example.x,
            "uprate": example.uprate,
            "drate": example.drate,
            "jitter_ppm": example.jitter_ppm,
            "drift_ppm": example.drift_ppm,
            "initial_phase": example.initial_phase,
        }
        previous = previous_sampling_clock_impairments(
            **kwargs,
            rng=np.random.default_rng(example.seed),
        )
        current = sampling_clock_impairments(
            **kwargs,
            rng=np.random.default_rng(example.seed),
        )

        assert previous.dtype == current.dtype == np.dtype(np.complex64)
        np.testing.assert_array_equal(current, previous, err_msg=example.name)
        print(f"PASS: {example.name} ({len(current)} output samples)")

    print(f"All {len(examples())} sampling-clock refactor examples passed.")


if __name__ == "__main__":
    main()
