"""Tests for extracted sampling-clock position generation."""

import numpy as np

from torchsig.utils.dsp import generate_sampling_positions, sampling_clock_impairments
from torchsig.utils.dsp_numba import sampling_clock_impairments_numba_wrapper


def test_zero_impairments_and_input_sample_accessors() -> None:
    positions = generate_sampling_positions(8, 2, 4, 4.0, 0.0, 0.0, initial_phase=0.25)

    np.testing.assert_array_equal(positions.actual, positions.nominal)
    np.testing.assert_array_equal(positions.nominal_indices, np.arange(len(positions.nominal)))
    np.testing.assert_array_equal(positions.nominal_input_samples, positions.nominal / 4)
    np.testing.assert_array_equal(positions.actual_input_samples, positions.actual / 4)


def test_historical_jitter_accumulation_and_skipped_alignment() -> None:
    class OffsetRng:
        def __init__(self) -> None:
            self.values = iter((100.0, 0.0, -100.0, 0.0, 0.0, 0.0, 0.0, 0.0))

        def normal(self, _mean, scale):
            return next(self.values) * scale

    positions = generate_sampling_positions(4, 1, 1, 1.0, 1_000_000.0, 0.0, rng=OffsetRng())

    np.testing.assert_array_equal(positions.nominal_indices, [0, 2, 3])
    np.testing.assert_array_equal(positions.nominal, [1.0, 3.0, 4.0])
    np.testing.assert_array_equal(positions.actual, [1.0, 3.0, 4.0])


def test_positions_are_reproducible() -> None:
    kwargs = {
        "num_input_samples": 128,
        "taps_per_phase": 3,
        "uprate": 8,
        "drate": 8.0,
        "jitter_ppm": 10.0,
        "drift_ppm": 5.0,
    }

    first = generate_sampling_positions(rng=np.random.default_rng(123), **kwargs)
    second = generate_sampling_positions(rng=np.random.default_rng(123), **kwargs)

    np.testing.assert_array_equal(first.nominal_indices, second.nominal_indices)
    np.testing.assert_array_equal(first.actual, second.actual)


def test_python_numba_parity_with_extracted_positions() -> None:
    kwargs = {
        "h": np.array([1.0, 0.5, 0.25, 0.125], dtype=np.float32),
        "x": np.arange(32, dtype=np.complex64),
        "uprate": 4,
        "drate": 4.0,
        "jitter_ppm": 50_000.0,
        "drift_ppm": 10.0,
    }

    expected = sampling_clock_impairments(rng=np.random.default_rng(91), **kwargs)
    actual = sampling_clock_impairments_numba_wrapper(rng=np.random.default_rng(91), **kwargs)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)
