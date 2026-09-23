"""Core behavioral tests for the sampling-clock impairment engines."""

import numpy as np

from torchsig.utils.dsp import sampling_clock_impairments
from torchsig.utils.dsp_numba import sampling_clock_impairments_numba_wrapper


def _kwargs() -> dict:
    return {
        "h": np.array([1.0, 0.5, -0.25, 0.125], dtype=np.float32),
        "x": np.arange(64, dtype=np.float32).astype(np.complex64),
        "uprate": 4,
        "drate": 4.0,
        "jitter_ppm": 10_000.0,
        "drift_ppm": -100_000.0,
    }


def test_sampling_clock_engines_match_for_offset_and_jitter() -> None:
    kwargs = _kwargs()

    reference = sampling_clock_impairments(
        rng=np.random.default_rng(123),
        **kwargs,
    )
    accelerated = sampling_clock_impairments_numba_wrapper(
        rng=np.random.default_rng(123),
        **kwargs,
    )

    np.testing.assert_allclose(accelerated, reference, rtol=1e-6, atol=1e-6)


def test_sampling_clock_engines_are_seed_reproducible() -> None:
    kwargs = _kwargs()

    first = sampling_clock_impairments_numba_wrapper(
        rng=np.random.default_rng(7),
        **kwargs,
    )
    second = sampling_clock_impairments_numba_wrapper(
        rng=np.random.default_rng(7),
        **kwargs,
    )

    np.testing.assert_array_equal(first, second)


def test_rate_offset_changes_raw_output_length() -> None:
    kwargs = {
        "h": np.array([1.0], dtype=np.float32),
        "x": np.ones(100, dtype=np.complex64),
        "uprate": 1,
        "drate": 1.0,
        "jitter_ppm": 0.0,
        "rng": np.random.default_rng(123),
    }

    nominal = sampling_clock_impairments(drift_ppm=0.0, **kwargs)
    faster = sampling_clock_impairments(drift_ppm=500_000.0, **kwargs)
    slower = sampling_clock_impairments(drift_ppm=-500_000.0, **kwargs)

    assert len(faster) < len(nominal) < len(slower)


def test_independent_jitter_does_not_change_output_length() -> None:
    kwargs = {
        "h": np.array([1.0], dtype=np.float32),
        "x": np.ones(100, dtype=np.complex64),
        "uprate": 8,
        "drate": 8.0,
        "drift_ppm": 0.0,
    }

    nominal = sampling_clock_impairments(
        jitter_ppm=0.0,
        rng=np.random.default_rng(123),
        **kwargs,
    )
    jittered = sampling_clock_impairments(
        jitter_ppm=100_000.0,
        rng=np.random.default_rng(123),
        **kwargs,
    )

    assert len(jittered) == len(nominal)
