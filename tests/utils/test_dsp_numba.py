"""Tests for the numba-accelerated sampling clock impairments.

Verifies the numba implementation matches the pure-NumPy reference (to float32
precision) and that the clock_drift / clock_jitter transforms remain correct
and reproducible.
"""

import numpy as np
import pytest

pytest.importorskip("numba")

from torchsig.transforms import functional as F
from torchsig.utils.dsp import (
    TorchSigRealDataType,
    prototype_polyphase_filter,
    sampling_clock_impairments,
)
from torchsig.utils.dsp_numba import sampling_clock_impairments_numba_wrapper

UPRATE = 5000


def _filter_and_data(seed=123, n=4096):
    h = prototype_polyphase_filter(num_branches=UPRATE).astype(TorchSigRealDataType)
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    return h, x


@pytest.mark.parametrize(
    "jitter_ppm,drift_ppm",
    [(0.0, 10.0), (10.0, 0.0), (0.0, 0.0)],
)
def test_numba_matches_reference(jitter_ppm, drift_ppm):
    """Numba output matches the NumPy reference to float32 precision."""
    h, x = _filter_and_data()
    kw = dict(h=h, x=x, uprate=UPRATE, drate=UPRATE, jitter_ppm=jitter_ppm, drift_ppm=drift_ppm)

    ref = sampling_clock_impairments(rng=np.random.default_rng(42), **kw)
    out = sampling_clock_impairments_numba_wrapper(rng=np.random.default_rng(42), **kw)

    assert len(out) == len(ref)
    assert out.dtype == np.complex64
    np.testing.assert_allclose(out, ref, atol=1e-4)


def test_numba_reproducible():
    """Same seed yields identical numba output."""
    h, x = _filter_and_data()
    kw = dict(h=h, x=x, uprate=UPRATE, drate=UPRATE, jitter_ppm=0.0, drift_ppm=10.0)
    a = sampling_clock_impairments_numba_wrapper(rng=np.random.default_rng(7), **kw)
    b = sampling_clock_impairments_numba_wrapper(rng=np.random.default_rng(7), **kw)
    np.testing.assert_array_equal(a, b)


def test_functional_uses_numba():
    """functional.py wired the accelerated implementation in."""
    assert F._sampling_clock_impairments is sampling_clock_impairments_numba_wrapper


@pytest.mark.parametrize("transform", [F.clock_drift, F.clock_jitter])
def test_clock_transforms_preserve_length(transform):
    """clock_drift / clock_jitter return the same number of samples as the input."""
    _, x = _filter_and_data(n=4096)
    out = transform(x, 10.0, np.random.default_rng(5))
    assert len(out) == len(x)
    assert out.dtype == np.complex64
    assert np.all(np.isfinite(out))


# --------------------------------------------------------------------------- #
# digital_agc
# --------------------------------------------------------------------------- #

from torchsig.utils.dsp_numba import digital_agc_numba  # noqa: E402

_AGC_ARGS = (0.0, 1e-4, 1e-3, 0.1, 1e-3, 0.0, 1.0, -80.0, 10.0)


def _agc_data(seed=11, n=4096):
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    x[5] = 0  # exercise the zero-sample branch
    return x


def test_digital_agc_matches_reference():
    """Numba digital_agc matches the pure-NumPy reference to float precision."""
    x = _agc_data()
    ref = F._digital_agc_python(x, *_AGC_ARGS)
    out = digital_agc_numba(x, *_AGC_ARGS)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)


def test_digital_agc_functional_uses_numba_and_preserves_length():
    """functional.digital_agc uses numba and preserves length/dtype."""
    assert F._digital_agc_numba is digital_agc_numba
    x = _agc_data()
    out = F.digital_agc(x)
    assert len(out) == len(x)
    assert out.dtype == np.complex64
    assert np.all(np.isfinite(out))
