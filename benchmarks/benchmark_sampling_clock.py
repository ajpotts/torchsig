"""Benchmarks for Python and Numba sampling-clock implementations."""

import numpy as np
import pytest

from torchsig.utils.dsp import sampling_clock_impairments
from torchsig.utils.dsp_numba import sampling_clock_impairments_numba_wrapper


@pytest.mark.parametrize(
    "implementation",
    [sampling_clock_impairments, sampling_clock_impairments_numba_wrapper],
    ids=["python", "numba"],
)
def test_benchmark_sampling_clock_implementation(benchmark, implementation) -> None:
    rng = np.random.default_rng(7)
    data = (rng.standard_normal(4096) + 1j * rng.standard_normal(4096)).astype(np.complex64)
    taps = np.hanning(64).astype(np.float32)
    kwargs = {
        "h": taps,
        "x": data,
        "uprate": 8,
        "drate": 8.0,
        "jitter_ppm": 10.0,
        "drift_ppm": 0.0,
    }

    def run():
        return implementation(rng=np.random.default_rng(11), **kwargs)

    output = benchmark(run)
    assert output.dtype == np.complex64
