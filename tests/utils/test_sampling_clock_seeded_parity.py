"""Seeded parity tests for NumPy and Numba sampling-clock engines."""

import numpy as np
import pytest

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


def test_seeded_engines_match_for_combined_drift_and_jitter() -> None:
    kwargs = _kwargs()
    reference = sampling_clock_impairments(rng=np.random.default_rng(123), **kwargs)
    accelerated = sampling_clock_impairments_numba_wrapper(
        rng=np.random.default_rng(123),
        **kwargs,
    )

    np.testing.assert_allclose(accelerated, reference, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "implementation",
    [sampling_clock_impairments, sampling_clock_impairments_numba_wrapper],
)
def test_seeded_engine_is_reproducible(implementation) -> None:
    kwargs = _kwargs()
    first = implementation(rng=np.random.default_rng(7), **kwargs)
    second = implementation(rng=np.random.default_rng(7), **kwargs)

    np.testing.assert_array_equal(first, second)
