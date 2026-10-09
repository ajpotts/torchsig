"""Tests for sampling-clock output-capacity calculation."""

import numpy as np
import pytest

from torchsig.utils.dsp import sampling_clock_impairments
from torchsig.utils.dsp_numba import sampling_clock_impairments_numba_wrapper

IMPLEMENTATIONS = (
    sampling_clock_impairments,
    sampling_clock_impairments_numba_wrapper,
)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_small_drate_with_initial_position_past_bound_returns_empty(implementation) -> None:
    output = implementation(
        h=np.array([1.0], dtype=np.float32),
        x=np.ones(4, dtype=np.complex64),
        uprate=1,
        drate=1e-12,
        jitter_ppm=0.0,
        drift_ppm=0.0,
        rng=np.random.default_rng(7),
    )

    assert output.shape == (0,)
    assert output.dtype == np.complex64
