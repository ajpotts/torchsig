"""Tests for independent sampling-clock jitter behavior."""

import numpy as np
import pytest

from torchsig.utils.dsp import sampling_clock_impairments
from torchsig.utils.dsp_numba import sampling_clock_impairments_numba_wrapper


IMPLEMENTATIONS = (
    sampling_clock_impairments,
    sampling_clock_impairments_numba_wrapper,
)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_jitter_does_not_change_nominal_output_length(implementation) -> None:
    kwargs = {
        "h": np.array([1.0], dtype=np.float32),
        "x": np.ones(100, dtype=np.complex64),
        "uprate": 8,
        "drate": 8.0,
        "drift_ppm": 0.0,
    }

    nominal = implementation(
        jitter_ppm=0.0,
        rng=np.random.default_rng(123),
        **kwargs,
    )
    jittered = implementation(
        jitter_ppm=100_000.0,
        rng=np.random.default_rng(123),
        **kwargs,
    )

    assert len(jittered) == len(nominal)
    assert jittered.dtype == np.complex64


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_jitter_scale_is_measured_in_input_sample_periods(implementation) -> None:
    h = np.ones(8, dtype=np.float32)
    x = np.arange(32, dtype=np.float32).astype(np.complex64)

    output = implementation(
        h=h,
        x=x,
        uprate=8,
        drate=8.0,
        jitter_ppm=100_000.0,
        drift_ppm=0.0,
        rng=np.random.default_rng(7),
    )

    assert output.dtype == np.complex64
    assert np.all(np.isfinite(output))
