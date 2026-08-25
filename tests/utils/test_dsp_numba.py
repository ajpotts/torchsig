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
from torchsig.utils.dsp_numba import (
    digital_agc_numba,
    partition_polyphase_numba,
    sampling_clock_impairments_numba,
    sampling_clock_impairments_numba_wrapper,
)

UPRATE = 5000


def _filter_and_data(seed=123, n=4096, uprate=UPRATE):
    h = prototype_polyphase_filter(num_branches=uprate).astype(TorchSigRealDataType)
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    return h, x


@pytest.mark.slow
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
    kw = {
        "h": h,
        "x": x,
        "uprate": UPRATE,
        "drate": UPRATE,
        "jitter_ppm": 10.0,
        "drift_ppm": 10.0,
    }
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


def test_partition_polyphase_numba_scales_and_zero_pads_taps():
    h = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)

    out = partition_polyphase_numba(h, up_rate=3, taps_per_phase=2)

    expected = np.array(
        [
            [3.0, 12.0],
            [6.0, 15.0],
            [9.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(out, expected)


def test_sampling_clock_numba_wrapper_does_not_consume_rng_when_no_jitter_or_drift():
    class ExplodingRng:
        def normal(self, *args, **kwargs):
            raise AssertionError("rng.normal should not be called")

    h = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    x = np.array([1 + 1j, 2 - 1j, -1 + 0.5j], dtype=np.complex64)

    out = sampling_clock_impairments_numba_wrapper(
        h=h,
        x=x,
        uprate=2,
        drate=2,
        jitter_ppm=0.0,
        drift_ppm=0.0,
        rng=ExplodingRng(),
    )

    assert out.dtype == np.complex64
    assert np.all(np.isfinite(out))


def test_sampling_clock_reference_scales_jitter_by_input_sample_period():
    """Timing jitter is converted from PPM to polyphase-bank units."""

    class RecordingRng:
        def __init__(self):
            self.scales = []

        def normal(self, location, scale):
            assert location == 0.0
            self.scales.append(scale)
            return 0.0

    rng = RecordingRng()
    sampling_clock_impairments(
        h=np.array([1.0, 0.0], dtype=np.float32),
        x=np.ones(4, dtype=np.complex64),
        uprate=2,
        drate=2.0,
        jitter_ppm=20.0,
        drift_ppm=30.0,
        rng=rng,
    )

    np.testing.assert_allclose(rng.scales, 2.0 * 20.0e-6)


def test_sampling_clock_numba_wrapper_scales_jitter_by_input_sample_period(
    monkeypatch,
):
    """The Numba jitter pool uses the same timing-unit conversion."""
    captured = {}

    def capture_pool(*args):
        captured["pool"] = args[3]
        return np.zeros(1, dtype=np.complex64)

    class OnesRng:
        def normal(self, location, scale, size):
            assert location == 0.0
            assert np.isclose(scale, 4.0 * 20.0e-6)
            return np.full(size, scale)

    monkeypatch.setattr(
        "torchsig.utils.dsp_numba.sampling_clock_impairments_numba",
        capture_pool,
    )

    sampling_clock_impairments_numba_wrapper(
        h=np.array([1.0, 0.0], dtype=np.float32),
        x=np.ones(4, dtype=np.complex64),
        uprate=4,
        drate=4.0,
        jitter_ppm=20.0,
        drift_ppm=30.0,
        rng=OnesRng(),
    )

    np.testing.assert_allclose(captured["pool"], 4.0 * 20.0e-6)


def test_sampling_clock_numba_wrapper_handles_real_valued_complex_input():
    h = np.array([0.5, 0.25, -0.125, 0.0], dtype=np.float32)
    x = np.array([1, 2, 3, 4], dtype=np.complex64)

    out = sampling_clock_impairments_numba_wrapper(
        h=h,
        x=x,
        uprate=2,
        drate=2,
        jitter_ppm=0.0,
        drift_ppm=0.0,
        rng=np.random.default_rng(123),
    )

    assert out.dtype == np.complex64
    np.testing.assert_array_equal(out.imag, np.zeros_like(out.imag))


def test_sampling_clock_rate_offset_changes_raw_output_length():
    """Positive rate offset advances faster; negative offset advances slower."""
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
    faster_numba = sampling_clock_impairments_numba_wrapper(drift_ppm=500_000.0, **kwargs)
    slower_numba = sampling_clock_impairments_numba_wrapper(drift_ppm=-500_000.0, **kwargs)

    assert len(faster) < len(nominal) < len(slower)
    assert len(faster_numba) == len(faster)
    assert len(slower_numba) == len(slower)


def test_independent_jitter_does_not_change_output_length():
    kwargs = {
        "h": np.array([1.0], dtype=np.float32),
        "x": np.ones(100, dtype=np.complex64),
        "uprate": 8,
        "drate": 8.0,
        "drift_ppm": 0.0,
    }

    nominal = sampling_clock_impairments(jitter_ppm=0.0, rng=np.random.default_rng(123), **kwargs)
    jittered = sampling_clock_impairments(jitter_ppm=100_000.0, rng=np.random.default_rng(123), **kwargs)

    assert len(jittered) == len(nominal)


def test_sampling_clock_implementations_match_with_fixed_rate_offset_and_jitter():
    kwargs = {
        "h": np.array([1.0, 0.5, -0.25, 0.125], dtype=np.float32),
        "x": np.arange(32, dtype=np.float32).astype(np.complex64),
        "uprate": 4,
        "drate": 4.0,
        "jitter_ppm": 10_000.0,
        "drift_ppm": -100_000.0,
    }

    reference = sampling_clock_impairments(rng=np.random.default_rng(123), **kwargs)
    actual = sampling_clock_impairments_numba_wrapper(rng=np.random.default_rng(123), **kwargs)

    np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=1e-6)


def test_sampling_clock_implementations_match_with_initial_phase():
    kwargs = {
        "h": np.array([1.0, 0.5, -0.25, 0.125], dtype=np.float32),
        "x": np.arange(32, dtype=np.float32).astype(np.complex64),
        "uprate": 4,
        "drate": 4.0,
        "jitter_ppm": 0.0,
        "drift_ppm": 0.0,
        "initial_phase": 0.75,
    }

    reference = sampling_clock_impairments(rng=np.random.default_rng(123), **kwargs)
    actual = sampling_clock_impairments_numba_wrapper(rng=np.random.default_rng(123), **kwargs)

    np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=1e-6)
    legacy = sampling_clock_impairments(
        **{**kwargs, "initial_phase": 0.0},
        rng=np.random.default_rng(123),
    )
    assert not np.array_equal(reference, legacy)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"jitter_ppm": -1.0}, "jitter_ppm"),
        ({"jitter_ppm": np.nan}, "jitter_ppm"),
        ({"drift_ppm": np.inf}, "drift_ppm"),
        ({"drift_ppm": -1_000_000.0}, "nonpositive sampling-position"),
        ({"uprate": 0}, "uprate must be a positive integer"),
        ({"drate": 0.0}, "drate must be finite and positive"),
        ({"drate": np.nan}, "drate must be finite and positive"),
        ({"initial_phase": -0.1}, "initial_phase"),
        ({"initial_phase": 1.0}, "initial_phase"),
        ({"initial_phase": np.nan}, "initial_phase"),
    ],
)
@pytest.mark.parametrize(
    "implementation",
    [sampling_clock_impairments, sampling_clock_impairments_numba_wrapper],
)
def test_sampling_clock_wrapper_rejects_invalid_parameters(overrides, match, implementation):
    kwargs = {
        "h": np.array([1.0], dtype=np.float32),
        "x": np.ones(4, dtype=np.complex64),
        "uprate": 1,
        "drate": 1.0,
        "jitter_ppm": 0.0,
        "drift_ppm": 0.0,
        "rng": np.random.default_rng(123),
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=match):
        implementation(**kwargs)


def test_sampling_clock_jitter_cannot_select_negative_polyphase_branch():
    class LargeNegativeJitterRng:
        def normal(self, location, _scale, size=None):
            assert location == 0.0
            if size is None:
                return -1e9
            return np.full(size, -1e9, dtype=np.float32)

    kwargs = {
        "h": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "x": np.ones(4, dtype=np.complex64),
        "uprate": 4,
        "drate": 4.0,
        "jitter_ppm": 1.0,
        "drift_ppm": 0.0,
    }

    reference = sampling_clock_impairments(rng=LargeNegativeJitterRng(), **kwargs)
    actual = sampling_clock_impairments_numba_wrapper(rng=LargeNegativeJitterRng(), **kwargs)

    np.testing.assert_array_equal(actual, reference)


def test_sampling_clock_preserves_legacy_initial_phase_alignment():
    out = sampling_clock_impairments(
        h=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        x=np.array([1.0, 2.0], dtype=np.complex64),
        uprate=4,
        drate=4.0,
        jitter_ppm=0.0,
        drift_ppm=0.0,
    )

    np.testing.assert_array_equal(out, np.array([8.0, 16.0, 0.0], dtype=np.complex64))


@pytest.mark.parametrize("length", [0, 1])
def test_sampling_clock_empty_and_short_inputs_are_safe(length):
    x = np.ones(length, dtype=np.complex64)
    kwargs = {
        "h": np.array([1.0], dtype=np.float32),
        "x": x,
        "uprate": 1,
        "drate": 1.0,
        "jitter_ppm": 10.0,
        "drift_ppm": -10.0,
    }

    reference = sampling_clock_impairments(rng=np.random.default_rng(123), **kwargs)
    actual = sampling_clock_impairments_numba_wrapper(rng=np.random.default_rng(123), **kwargs)

    np.testing.assert_array_equal(actual, reference)


def test_sampling_clock_accepts_none_rng():
    out = sampling_clock_impairments_numba_wrapper(
        h=np.array([1.0], dtype=np.float32),
        x=np.ones(4, dtype=np.complex64),
        uprate=1,
        drate=1.0,
        jitter_ppm=1.0,
        drift_ppm=0.0,
        rng=None,
    )

    assert np.all(np.isfinite(out))


def test_realistic_impairments_do_not_exhaust_nominal_output_capacity():
    """Realistic offsets complete without exercising emergency buffer growth."""
    kwargs = {
        "h": np.array([1.0], dtype=np.float32),
        "x": np.ones(4096, dtype=np.complex64),
        "uprate": 1,
        "drate": 1.0,
        "jitter_ppm": 10.0,
        "drift_ppm": -10.0,
    }

    actual = sampling_clock_impairments_numba_wrapper(rng=np.random.default_rng(123), **kwargs)

    assert len(actual) <= len(kwargs["x"]) + 1


def test_sampling_clock_numba_kernel_raises_before_output_overflow():
    with pytest.raises(RuntimeError, match="output capacity exhausted"):
        sampling_clock_impairments_numba(
            np.ones(4, dtype=np.float32),
            np.zeros(4, dtype=np.float32),
            1,
            np.zeros(1, dtype=np.float64),
            np.array([[1.0]], dtype=np.float32),
            1,
            5,
            4,
            1.0,
            1.0,
            1,
        )


@pytest.mark.parametrize(
    "x",
    [
        np.zeros(16, dtype=np.complex64),
        np.ones(16, dtype=np.complex64),
        np.linspace(-1, 1, 16).astype(np.complex64),
    ],
)
def test_digital_agc_handles_special_input_patterns(x):
    out = digital_agc_numba(x, *_AGC_ARGS)

    assert out.shape == x.shape
    assert out.dtype == np.complex64
    assert np.all(np.isfinite(out))


def test_digital_agc_empty_input_returns_empty_complex64_array():
    x = np.array([], dtype=np.complex64)

    out = digital_agc_numba(x, *_AGC_ARGS)

    assert out.shape == (0,)
    assert out.dtype == np.complex64


@pytest.mark.parametrize(
    "sample,expected_finite",
    [
        (0 + 0j, True),
        (1 + 0j, True),
        (1 + 1j, True),
        (-1 - 1j, True),
    ],
)
def test_digital_agc_single_sample_inputs(sample, expected_finite):
    x = np.array([sample], dtype=np.complex64)

    out = digital_agc_numba(x, *_AGC_ARGS)

    assert out.shape == (1,)
    assert out.dtype == np.complex64
    assert np.isfinite(out[0])


def test_partition_polyphase_numba_py_func_covers_kernel_body():
    h = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)

    out = partition_polyphase_numba.py_func(h, 3, 2)

    expected = np.array(
        [
            [3.0, 12.0],
            [6.0, 15.0],
            [9.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(out, expected)


@pytest.mark.slow
def test_sampling_clock_impairments_numba_py_func_covers_kernel_body():
    uprate = 8
    drate = 8
    h, x = _filter_and_data(n=128, uprate=uprate)
    taps_per_phase = int(np.ceil(len(h) / uprate))
    h_pfb = partition_polyphase_numba.py_func(h, uprate, taps_per_phase)
    h_pfb_reversed = np.ascontiguousarray(np.flip(h_pfb, axis=1))

    padded_len = len(x) + 2 * taps_per_phase - 1
    max_start = padded_len - taps_per_phase
    num_output_samples = int(np.ceil(padded_len * uprate / drate)) + 1

    rng = np.random.default_rng(123)

    jitter_values = rng.normal(0.0, 1.0, num_output_samples * 2).astype(np.float32) * 1e-6

    out = sampling_clock_impairments_numba.py_func(
        x.real.astype(np.float32),
        x.imag.astype(np.float32),
        uprate,
        jitter_values,
        h_pfb_reversed,
        taps_per_phase,
        padded_len,
        max_start,
        float(drate),
        uprate / drate,
        num_output_samples,
    )

    assert out.dtype == np.complex64
    assert len(out) > 0
    assert np.all(np.isfinite(out))


def test_sampling_clock_impairments_numba_py_func_handles_empty_output():
    x = np.array([], dtype=np.complex64)

    out = sampling_clock_impairments_numba.py_func(
        x.real.astype(np.float32),
        x.imag.astype(np.float32),
        1,
        np.zeros(1, dtype=np.float64),
        np.array([[1.0]], dtype=np.float32),
        1,
        1,
        0,
        1.0,
        1.0,
        1,
    )

    assert out.shape == (0,)
    assert out.dtype == np.complex64


def test_digital_agc_numba_py_func_covers_kernel_body():
    x = _agc_data(n=64)

    out = digital_agc_numba.py_func(x, *_AGC_ARGS)

    ref = F._digital_agc_python(x, *_AGC_ARGS)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)


def test_digital_agc_numba_py_func_empty_input():
    x = np.array([], dtype=np.complex64)

    out = digital_agc_numba.py_func(x, *_AGC_ARGS)

    assert out.shape == (0,)
    assert out.dtype == np.complex64
