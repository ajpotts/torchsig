"""Tests for fixed sampling-clock rate-offset behavior."""

import numpy as np
import pytest

from torchsig.utils.dsp import sampling_clock_impairments
from torchsig.utils.dsp_numba import sampling_clock_impairments_numba_wrapper

IMPLEMENTATIONS = (
    sampling_clock_impairments,
    sampling_clock_impairments_numba_wrapper,
)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_signed_drift_changes_raw_output_length(implementation) -> None:
    kwargs = {
        "h": np.array([1.0], dtype=np.float32),
        "x": np.ones(100, dtype=np.complex64),
        "uprate": 1,
        "drate": 1.0,
        "jitter_ppm": 0.0,
        "rng": np.random.default_rng(7),
    }

    nominal = implementation(drift_ppm=0.0, **kwargs)
    faster = implementation(drift_ppm=500_000.0, **kwargs)
    slower = implementation(drift_ppm=-500_000.0, **kwargs)

    assert len(faster) < len(nominal) < len(slower)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_fixed_drift_does_not_consume_random_values(implementation) -> None:
    class ExplodingRng:
        def normal(self, *_args, **_kwargs):
            raise AssertionError("fixed drift must not consume random values")

    output = implementation(
        h=np.array([1.0], dtype=np.float32),
        x=np.ones(16, dtype=np.complex64),
        uprate=1,
        drate=1.0,
        jitter_ppm=0.0,
        drift_ppm=10.0,
        rng=ExplodingRng(),
    )

    assert output.dtype == np.complex64
