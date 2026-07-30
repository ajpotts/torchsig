"""Unit tests for the FSK signal builder and modulator."""

from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from torchsig.signals.builders.fsk import (
    FSKSignalGenerator,
    fsk_modulator,
    fsk_modulator_baseband,
    gaussian_taps,
    get_fsk_freq_map,
    get_fsk_mod_index,
)
from torchsig.utils.dsp import TorchSigComplexDataType


MODULE_PATH = "torchsig.signals.builders.fsk"


@pytest.mark.parametrize("constellation_size", [1, 2, 4, 8, 16])
def test_get_fsk_freq_map_returns_expected_values(constellation_size):
    """The frequency map should contain the documented evenly spaced values."""
    result = get_fsk_freq_map(constellation_size)

    expected = np.linspace(
        -1 + (1 / constellation_size),
        1 - (1 / constellation_size),
        constellation_size,
        endpoint=True,
    )

    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("constellation_size", [2, 4, 8, 16])
def test_get_fsk_freq_map_is_symmetric(constellation_size):
    """Even-order FSK maps should be symmetric around zero."""
    result = get_fsk_freq_map(constellation_size)

    np.testing.assert_allclose(result, -result[::-1])


@pytest.mark.parametrize("constellation_size", [2, 4, 8, 16])
def test_get_fsk_freq_map_has_expected_size(constellation_size):
    """The frequency map should contain the requested number of points."""
    result = get_fsk_freq_map(constellation_size)

    assert result.shape == (constellation_size,)


def test_get_fsk_mod_index_creates_default_rng():
    """A default random generator should be created when none is supplied."""
    rng = MagicMock(spec=np.random.Generator)
    rng.uniform.return_value = 0.3

    with patch(
        f"{MODULE_PATH}.np.random.default_rng",
        return_value=rng,
    ) as default_rng:
        result = get_fsk_mod_index("gfsk")

    default_rng.assert_called_once_with()
    assert result == 0.3


def test_get_fsk_mod_index_gfsk():
    """GFSK should draw its modulation index from the Bluetooth range."""
    rng = MagicMock(spec=np.random.Generator)
    rng.uniform.return_value = 0.3

    result = get_fsk_mod_index("gfsk", rng)

    assert result == 0.3
    rng.uniform.assert_called_once_with(0.1, 0.5)


@pytest.mark.parametrize("fsk_type", ["msk", "gmsk"])
def test_get_fsk_mod_index_msk_variants(fsk_type):
    """MSK and GMSK should use a fixed modulation index of 0.5."""
    rng = MagicMock(spec=np.random.Generator)

    result = get_fsk_mod_index(fsk_type, rng)

    assert result == 0.5
    rng.uniform.assert_not_called()


def test_get_fsk_mod_index_fsk_orthogonal_branch():
    """FSK should return one when the orthogonal branch is selected."""
    rng = MagicMock(spec=np.random.Generator)
    rng.uniform.return_value = 0.25

    result = get_fsk_mod_index("fsk", rng)

    assert result == 1.0
    rng.uniform.assert_called_once_with(0, 1)


def test_get_fsk_mod_index_fsk_nonorthogonal_branch():
    """Nonorthogonal FSK should draw an index from the configured range."""
    rng = MagicMock(spec=np.random.Generator)
    rng.uniform.side_effect = [0.75, 0.9]

    result = get_fsk_mod_index("fsk", rng)

    assert result == 0.9
    assert rng.uniform.call_args_list == [
        call(0, 1),
        call(0.7, 1.01),
    ]


def test_get_fsk_mod_index_rejects_unknown_type():
    """Unsupported FSK types should raise ValueError."""
    with pytest.raises(
        ValueError,
        match="Unexpected fsk_type: invalid",
    ):
        get_fsk_mod_index(
            "invalid",
            np.random.default_rng(42),
        )


@pytest.mark.parametrize("bt", [-1.0, -0.1, 1.1, 2.0])
def test_gaussian_taps_rejects_invalid_bt(bt):
    """The Gaussian time-bandwidth product must lie between zero and one."""
    with pytest.raises(
        ValueError,
        match="bt must be between 0.0 and 1.0",
    ):
        gaussian_taps(
            samples_per_symbol=4,
            bt=bt,
            rng=np.random.default_rng(42),
        )


@pytest.mark.parametrize("bt", [0.0, 0.1, 0.5, 1.0])
def test_gaussian_taps_accepts_boundary_and_interior_bt(bt):
    """The documented closed interval for BT should be accepted."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = 2

    result = gaussian_taps(
        samples_per_symbol=4,
        bt=bt,
        rng=rng,
    )

    assert result.ndim == 1
    assert np.all(np.isfinite(result))


def test_gaussian_taps_creates_default_rng():
    """A default generator should be created when none is provided."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = 2

    with patch(
        f"{MODULE_PATH}.np.random.default_rng",
        return_value=rng,
    ) as default_rng:
        gaussian_taps(
            samples_per_symbol=4,
            bt=0.3,
        )

    default_rng.assert_called_once_with()


@pytest.mark.parametrize("filter_span", [1, 2, 3, 4])
def test_gaussian_taps_has_expected_length(filter_span):
    """The tap count should follow the selected two-sided filter span."""
    samples_per_symbol = 4
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = filter_span

    result = gaussian_taps(
        samples_per_symbol=samples_per_symbol,
        bt=0.3,
        rng=rng,
    )

    expected_length = 2 * filter_span * samples_per_symbol + 1

    assert result.shape == (expected_length,)
    rng.integers.assert_called_once_with(1, 5)


def test_gaussian_taps_sum_to_one():
    """Gaussian taps should be normalized to unit sum."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = 3

    result = gaussian_taps(
        samples_per_symbol=4,
        bt=0.3,
        rng=rng,
    )

    assert np.sum(result) == pytest.approx(1.0)


def test_gaussian_taps_are_symmetric():
    """The Gaussian filter should be symmetric around its center."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = 3

    result = gaussian_taps(
        samples_per_symbol=4,
        bt=0.3,
        rng=rng,
    )

    np.testing.assert_allclose(result, result[::-1])


def test_gaussian_taps_are_nonnegative():
    """All Gaussian pulse-shaping coefficients should be nonnegative."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = 2

    result = gaussian_taps(
        samples_per_symbol=4,
        bt=0.3,
        rng=rng,
    )

    assert np.all(result >= 0)


@pytest.mark.parametrize("max_num_samples", [0, -1, -100])
def test_fsk_modulator_baseband_rejects_nonpositive_max_samples(
    max_num_samples,
):
    """Nonpositive output lengths should be rejected."""
    with pytest.raises(
        ValueError,
        match="max_num_samples must be positive",
    ):
        fsk_modulator_baseband(
            constellation_size=4,
            fsk_type="fsk",
            max_num_samples=max_num_samples,
            oversampling_rate_nominal=4,
            rng=np.random.default_rng(42),
        )


@pytest.mark.parametrize("oversampling_rate", [0, -1, -10])
def test_fsk_modulator_baseband_rejects_nonpositive_oversampling_rate(
    oversampling_rate,
):
    """Nonpositive nominal oversampling rates should be rejected."""
    with pytest.raises(
        ValueError,
        match="oversampling_rate_nominal must be positive",
    ):
        fsk_modulator_baseband(
            constellation_size=4,
            fsk_type="fsk",
            max_num_samples=128,
            oversampling_rate_nominal=oversampling_rate,
            rng=np.random.default_rng(42),
        )


def test_fsk_modulator_baseband_creates_default_rng():
    """A default random generator should be created when none is supplied."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = np.array([0])

    filtered = np.ones(16)

    with (
        patch(
            f"{MODULE_PATH}.np.random.default_rng",
            return_value=rng,
        ) as default_rng,
        patch(
            f"{MODULE_PATH}.get_fsk_mod_index",
            return_value=1.0,
        ),
        patch(
            f"{MODULE_PATH}.sp.upfirdn",
            return_value=filtered,
        ),
        patch(
            f"{MODULE_PATH}.pad_head_tail_to_length",
            return_value=np.ones(32, dtype=np.complex64),
        ),
    ):
        result = fsk_modulator_baseband(
            constellation_size=4,
            fsk_type="fsk",
            max_num_samples=32,
            oversampling_rate_nominal=4,
        )

    default_rng.assert_called_once_with()
    assert result.shape == (32,)


def test_fsk_modulator_baseband_uses_modulation_index_and_frequency_map():
    """The baseband modulator should request its helper parameters."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = np.array([0])
    frequency_map = np.array([-0.5, 0.5])

    with (
        patch(
            f"{MODULE_PATH}.get_fsk_mod_index",
            return_value=0.75,
        ) as get_mod_index,
        patch(
            f"{MODULE_PATH}.get_fsk_freq_map",
            return_value=frequency_map,
        ) as get_freq_map,
        patch(
            f"{MODULE_PATH}.sp.upfirdn",
            return_value=np.ones(8),
        ),
        patch(
            f"{MODULE_PATH}.pad_head_tail_to_length",
            return_value=np.ones(16, dtype=np.complex64),
        ),
    ):
        fsk_modulator_baseband(
            constellation_size=2,
            fsk_type="fsk",
            max_num_samples=16,
            oversampling_rate_nominal=4,
            rng=rng,
        )

    get_mod_index.assert_called_once_with("fsk", rng)
    get_freq_map.assert_called_once_with(2)


def test_fsk_modulator_baseband_rectangular_pulse_shape():
    """Plain FSK should use a rectangular pulse without Gaussian filtering."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = np.array([0])

    filtered = np.ones(16)

    with (
        patch(
            f"{MODULE_PATH}.get_fsk_mod_index",
            return_value=1.0,
        ),
        patch(
            f"{MODULE_PATH}.gaussian_taps",
        ) as gaussian,
        patch(
            f"{MODULE_PATH}.sp.convolve",
        ) as convolve,
        patch(
            f"{MODULE_PATH}.sp.upfirdn",
            return_value=filtered,
        ) as upfirdn,
        patch(
            f"{MODULE_PATH}.pad_head_tail_to_length",
            return_value=np.ones(32, dtype=np.complex64),
        ),
    ):
        fsk_modulator_baseband(
            constellation_size=4,
            fsk_type="fsk",
            max_num_samples=32,
            oversampling_rate_nominal=4,
            rng=rng,
        )

    gaussian.assert_not_called()
    convolve.assert_not_called()

    pulse_shape = upfirdn.call_args.args[0]

    # samples_per_symbol = constellation size * oversampling rate
    np.testing.assert_array_equal(
        pulse_shape,
        np.ones(16),
    )


@pytest.mark.parametrize("fsk_type", ["gfsk", "gmsk"])
def test_fsk_modulator_baseband_gaussian_filter_branch(fsk_type):
    """GFSK and GMSK should apply Gaussian pulse shaping."""
    rng = MagicMock(spec=np.random.Generator)
    rng.uniform.return_value = 0.3
    rng.integers.return_value = np.array([0])

    taps = np.array([0.25, 0.5, 0.25])
    rectangular_pulse = np.ones(8)
    gaussian_pulse = np.ones(10)
    filtered = np.ones(10)

    with (
        patch(
            f"{MODULE_PATH}.get_fsk_mod_index",
            return_value=0.5,
        ),
        patch(
            f"{MODULE_PATH}.get_fsk_freq_map",
            return_value=np.array([-0.5, 0.5]),
        ),
        patch(
            f"{MODULE_PATH}.gaussian_taps",
            return_value=taps,
        ) as gaussian,
        patch(
            f"{MODULE_PATH}.sp.convolve",
            return_value=gaussian_pulse,
        ) as convolve,
        patch(
            f"{MODULE_PATH}.sp.upfirdn",
            return_value=filtered,
        ),
        patch(
            f"{MODULE_PATH}.pad_head_tail_to_length",
            return_value=np.ones(32, dtype=np.complex64),
        ),
    ):
        fsk_modulator_baseband(
            constellation_size=2,
            fsk_type=fsk_type,
            max_num_samples=32,
            oversampling_rate_nominal=4,
            rng=rng,
        )

    rng.uniform.assert_called_once_with(0.1, 0.5)
    gaussian.assert_called_once_with(8, 0.3, rng)

    np.testing.assert_array_equal(
        convolve.call_args.args[0],
        taps,
    )
    np.testing.assert_array_equal(
        convolve.call_args.args[1],
        rectangular_pulse,
    )


def test_fsk_modulator_baseband_generates_expected_symbol_count():
    """The number of generated symbols should account for pulse length."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = np.array([0, 1, 2, 3])

    frequency_map = np.array([-0.75, -0.25, 0.25, 0.75])
    filtered = np.ones(64)

    with (
        patch(
            f"{MODULE_PATH}.get_fsk_mod_index",
            return_value=1.0,
        ),
        patch(
            f"{MODULE_PATH}.get_fsk_freq_map",
            return_value=frequency_map,
        ),
        patch(
            f"{MODULE_PATH}.sp.upfirdn",
            return_value=filtered,
        ),
    ):
        fsk_modulator_baseband(
            constellation_size=4,
            fsk_type="fsk",
            max_num_samples=80,
            oversampling_rate_nominal=4,
            rng=rng,
        )

    # samples_per_symbol = 4 * 4 = 16
    # pulse length = 16
    # floor((80 - 16 + 1) / 16) = floor(65 / 16) = 4
    rng.integers.assert_called_once_with(
        0,
        4,
        4,
    )


def test_fsk_modulator_baseband_generates_at_least_one_symbol():
    """Short signals should still generate at least one symbol."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = np.array([0])

    with (
        patch(
            f"{MODULE_PATH}.get_fsk_mod_index",
            return_value=1.0,
        ),
        patch(
            f"{MODULE_PATH}.sp.upfirdn",
            return_value=np.ones(16),
        ),
        patch(
            f"{MODULE_PATH}.slice_tail_to_length",
            return_value=np.ones(4, dtype=np.complex64),
        ),
    ):
        fsk_modulator_baseband(
            constellation_size=4,
            fsk_type="fsk",
            max_num_samples=4,
            oversampling_rate_nominal=4,
            rng=rng,
        )

    rng.integers.assert_called_once_with(
        0,
        4,
        1,
    )


def test_fsk_modulator_baseband_oversamples_frequency_map():
    """Frequency symbols should be divided by the nominal oversampling rate."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = np.array([0, 3])

    frequency_map = np.array([-0.75, -0.25, 0.25, 0.75])
    filtered = np.ones(32)

    with (
        patch(
            f"{MODULE_PATH}.get_fsk_mod_index",
            return_value=1.0,
        ),
        patch(
            f"{MODULE_PATH}.get_fsk_freq_map",
            return_value=frequency_map,
        ),
        patch(
            f"{MODULE_PATH}.sp.upfirdn",
            return_value=filtered,
        ) as upfirdn,
    ):
        fsk_modulator_baseband(
            constellation_size=4,
            fsk_type="fsk",
            max_num_samples=47,
            oversampling_rate_nominal=4,
            rng=rng,
        )

    expected_symbols = np.array(
        [
            frequency_map[0] / 4,
            frequency_map[3] / 4,
        ]
    )

    np.testing.assert_allclose(
        upfirdn.call_args.args[1],
        expected_symbols,
    )
    assert upfirdn.call_args.kwargs == {
        "up": 16,
        "down": 1,
    }


def test_fsk_modulator_baseband_matches_phase_formula():
    """The modulated output should match cumulative phase integration."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = np.array([0])

    modulation_index = 0.5
    filtered = np.array([0.1, -0.2, 0.3, -0.4])

    with (
        patch(
            f"{MODULE_PATH}.get_fsk_mod_index",
            return_value=modulation_index,
        ),
        patch(
            f"{MODULE_PATH}.sp.upfirdn",
            return_value=filtered,
        ),
        patch(
            f"{MODULE_PATH}.pad_head_tail_to_length",
        ) as pad,
        patch(
            f"{MODULE_PATH}.slice_tail_to_length",
        ) as slice_tail,
    ):
        result = fsk_modulator_baseband(
            constellation_size=1,
            fsk_type="msk",
            max_num_samples=4,
            oversampling_rate_nominal=4,
            rng=rng,
        )

    expected_phase = np.cumsum(
        filtered * 1j * modulation_index * np.pi
    )
    expected = np.exp(expected_phase)

    pad.assert_not_called()
    slice_tail.assert_not_called()

    np.testing.assert_allclose(result, expected)


def test_fsk_modulator_baseband_has_unit_magnitude():
    """Continuous-phase FSK samples should lie on the unit circle."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = np.array([0])

    filtered = np.linspace(-0.25, 0.25, 16)

    with (
        patch(
            f"{MODULE_PATH}.get_fsk_mod_index",
            return_value=0.5,
        ),
        patch(
            f"{MODULE_PATH}.sp.upfirdn",
            return_value=filtered,
        ),
    ):
        result = fsk_modulator_baseband(
            constellation_size=4,
            fsk_type="msk",
            max_num_samples=16,
            oversampling_rate_nominal=4,
            rng=rng,
        )

    np.testing.assert_allclose(
        np.abs(result),
        np.ones(16),
        rtol=1e-12,
        atol=1e-12,
    )


def test_fsk_modulator_baseband_slices_long_signal():
    """An oversized modulated signal should be sliced from its tail."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = np.array([0])

    filtered = np.ones(20)
    sliced = np.ones(16, dtype=np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.get_fsk_mod_index",
            return_value=1.0,
        ),
        patch(
            f"{MODULE_PATH}.sp.upfirdn",
            return_value=filtered,
        ),
        patch(
            f"{MODULE_PATH}.slice_tail_to_length",
            return_value=sliced,
        ) as slice_tail,
        patch(
            f"{MODULE_PATH}.pad_head_tail_to_length",
        ) as pad,
    ):
        result = fsk_modulator_baseband(
            constellation_size=4,
            fsk_type="fsk",
            max_num_samples=16,
            oversampling_rate_nominal=4,
            rng=rng,
        )

    slice_tail.assert_called_once()

    original_modulated = np.exp(
        np.cumsum(filtered * 1j * np.pi)
    )
    np.testing.assert_allclose(
        slice_tail.call_args.args[0],
        original_modulated,
    )
    assert slice_tail.call_args.args[1] == 16

    pad.assert_not_called()
    assert result is sliced


def test_fsk_modulator_baseband_pads_short_signal():
    """A short modulated signal should be padded to the target length."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = np.array([0])

    filtered = np.ones(8)
    padded = np.ones(16, dtype=np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.get_fsk_mod_index",
            return_value=1.0,
        ),
        patch(
            f"{MODULE_PATH}.sp.upfirdn",
            return_value=filtered,
        ),
        patch(
            f"{MODULE_PATH}.pad_head_tail_to_length",
            return_value=padded,
        ) as pad,
        patch(
            f"{MODULE_PATH}.slice_tail_to_length",
        ) as slice_tail,
    ):
        result = fsk_modulator_baseband(
            constellation_size=4,
            fsk_type="fsk",
            max_num_samples=16,
            oversampling_rate_nominal=4,
            rng=rng,
        )

    pad.assert_called_once()
    assert pad.call_args.args[1] == 16
    slice_tail.assert_not_called()
    assert result is padded


@pytest.mark.parametrize(
    ("bandwidth", "sample_rate", "num_samples", "expected_message"),
    [
        (0, 10_000, 128, "bandwidth must be positive"),
        (-1, 10_000, 128, "bandwidth must be positive"),
        (1_000, 0, 128, "sample_rate must be positive"),
        (1_000, -1, 128, "sample_rate must be positive"),
        (
            5_001,
            10_000,
            128,
            "bandwidth must be less than sample_rate/2",
        ),
        (1_000, 10_000, 0, "num_samples must be positive"),
        (1_000, 10_000, -1, "num_samples must be positive"),
    ],
)
def test_fsk_modulator_rejects_invalid_inputs(
    bandwidth,
    sample_rate,
    num_samples,
    expected_message,
):
    """Invalid top-level modulation parameters should be rejected."""
    with pytest.raises(ValueError, match=expected_message):
        fsk_modulator(
            constellation_size=4,
            fsk_type="fsk",
            bandwidth=bandwidth,
            sample_rate=sample_rate,
            num_samples=num_samples,
            rng=np.random.default_rng(42),
        )


def test_fsk_modulator_creates_default_rng():
    """A default random generator should be created when none is supplied."""
    rng = MagicMock(spec=np.random.Generator)
    baseband = np.ones(40, dtype=np.complex64)
    resampled = np.ones(100, dtype=np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.np.random.default_rng",
            return_value=rng,
        ) as default_rng,
        patch(
            f"{MODULE_PATH}.fsk_modulator_baseband",
            return_value=baseband,
        ),
        patch(
            f"{MODULE_PATH}.multistage_polyphase_resampler",
            return_value=resampled,
        ),
        patch(
            f"{MODULE_PATH}.pad_head_tail_to_length",
            return_value=resampled,
        ),
    ):
        fsk_modulator(
            constellation_size=4,
            fsk_type="fsk",
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=100,
        )

    default_rng.assert_called_once_with()


def test_fsk_modulator_calculates_resampling_parameters():
    """The wrapper should calculate the expected baseband length and rate."""
    rng = np.random.default_rng(42)
    baseband = np.ones(40, dtype=np.complex64)
    resampled = np.ones(100, dtype=np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.fsk_modulator_baseband",
            return_value=baseband,
        ) as baseband_modulator,
        patch(
            f"{MODULE_PATH}.multistage_polyphase_resampler",
            return_value=resampled,
        ) as resampler,
        patch(
            f"{MODULE_PATH}.pad_head_tail_to_length",
            return_value=resampled,
        ),
    ):
        fsk_modulator(
            constellation_size=4,
            fsk_type="fsk",
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=100,
            rng=rng,
        )

    # oversampling_rate = 10
    # resample_rate_ideal = 10 / 4 = 2.5
    # baseband length = floor(100 / 2.5) = 40
    baseband_modulator.assert_called_once_with(
        4,
        "fsk",
        40,
        4,
        rng,
    )
    resampler.assert_called_once_with(
        baseband,
        2.5,
    )


def test_fsk_modulator_uses_minimum_baseband_length():
    """The wrapper should request at least four baseband samples."""
    rng = np.random.default_rng(42)
    baseband = np.ones(4, dtype=np.complex64)
    resampled = np.ones(1, dtype=np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.fsk_modulator_baseband",
            return_value=baseband,
        ) as baseband_modulator,
        patch(
            f"{MODULE_PATH}.multistage_polyphase_resampler",
            return_value=resampled,
        ),
        patch(
            f"{MODULE_PATH}.pad_head_tail_to_length",
            return_value=np.ones(1, dtype=np.complex64),
        ),
    ):
        fsk_modulator(
            constellation_size=2,
            fsk_type="msk",
            bandwidth=1,
            sample_rate=10_000,
            num_samples=1,
            rng=rng,
        )

    baseband_modulator.assert_called_once_with(
        2,
        "msk",
        4,
        4,
        rng,
    )


def test_fsk_modulator_applies_resampling_amplitude_correction():
    """The resampled signal should be scaled by the inverse resample rate."""
    rng = np.random.default_rng(42)
    baseband = np.ones(40, dtype=np.complex64)
    resampled = np.full(100, 10 + 5j, dtype=np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.fsk_modulator_baseband",
            return_value=baseband,
        ),
        patch(
            f"{MODULE_PATH}.multistage_polyphase_resampler",
            return_value=resampled.copy(),
        ),
        patch(
            f"{MODULE_PATH}.pad_head_tail_to_length",
            side_effect=lambda signal, _length: signal,
        ),
    ):
        result = fsk_modulator(
            constellation_size=4,
            fsk_type="fsk",
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=100,
            rng=rng,
        )

    resample_rate = 2.5
    expected = resampled / resample_rate

    np.testing.assert_allclose(
        result,
        expected.astype(TorchSigComplexDataType),
    )


def test_fsk_modulator_slices_long_resampled_signal():
    """An oversized resampled signal should be sliced to the target length."""
    rng = np.random.default_rng(42)
    baseband = np.ones(40, dtype=np.complex64)
    resampled = np.ones(110, dtype=np.complex64)
    original_resampled = resampled.copy()
    sliced = np.ones(100, dtype=np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.fsk_modulator_baseband",
            return_value=baseband,
        ),
        patch(
            f"{MODULE_PATH}.multistage_polyphase_resampler",
            return_value=resampled,
        ),
        patch(
            f"{MODULE_PATH}.slice_head_tail_to_length",
            return_value=sliced,
        ) as slice_signal,
        patch(
            f"{MODULE_PATH}.pad_head_tail_to_length",
        ) as pad_signal,
    ):
        result = fsk_modulator(
            constellation_size=4,
            fsk_type="fsk",
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=100,
            rng=rng,
        )

    expected_scaled = original_resampled / 2.5

    np.testing.assert_allclose(
        slice_signal.call_args.args[0],
        expected_scaled,
    )
    assert slice_signal.call_args.args[1] == 100

    pad_signal.assert_not_called()
    assert result.dtype == np.dtype(TorchSigComplexDataType)


@pytest.mark.parametrize("resampled_length", [90, 100])
def test_fsk_modulator_pads_signal_not_longer_than_target(
    resampled_length,
):
    """A signal no longer than the target should use the padding helper."""
    rng = np.random.default_rng(42)
    baseband = np.ones(40, dtype=np.complex64)
    resampled = np.ones(resampled_length, dtype=np.complex64)
    original_resampled = resampled.copy()
    padded = np.ones(100, dtype=np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.fsk_modulator_baseband",
            return_value=baseband,
        ),
        patch(
            f"{MODULE_PATH}.multistage_polyphase_resampler",
            return_value=resampled,
        ),
        patch(
            f"{MODULE_PATH}.pad_head_tail_to_length",
            return_value=padded,
        ) as pad_signal,
        patch(
            f"{MODULE_PATH}.slice_head_tail_to_length",
        ) as slice_signal,
    ):
        result = fsk_modulator(
            constellation_size=4,
            fsk_type="fsk",
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=100,
            rng=rng,
        )

    expected_scaled = original_resampled / 2.5

    np.testing.assert_allclose(
        pad_signal.call_args.args[0],
        expected_scaled,
    )
    assert pad_signal.call_args.args[1] == 100

    slice_signal.assert_not_called()
    assert result.shape == (100,)


def test_fsk_modulator_returns_torchsig_complex_dtype():
    """The wrapper should return the configured TorchSig complex dtype."""
    rng = np.random.default_rng(42)
    baseband = np.ones(40)
    resampled = np.ones(100)

    with (
        patch(
            f"{MODULE_PATH}.fsk_modulator_baseband",
            return_value=baseband,
        ),
        patch(
            f"{MODULE_PATH}.multistage_polyphase_resampler",
            return_value=resampled,
        ),
        patch(
            f"{MODULE_PATH}.pad_head_tail_to_length",
            return_value=resampled,
        ),
    ):
        result = fsk_modulator(
            constellation_size=4,
            fsk_type="fsk",
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=100,
            rng=rng,
        )

    assert result.dtype == np.dtype(TorchSigComplexDataType)


def test_fsk_signal_generator_initialization():
    """The generator should configure required fields and its class name."""
    metadata = {
        "fsk_type": "gfsk",
        "constellation_size": 4,
    }

    with (
        patch(
            f"{MODULE_PATH}.BaseSignalGenerator.__init__",
            autospec=True,
        ) as base_init,
        patch.object(
            FSKSignalGenerator,
            "__getitem__",
            side_effect=metadata.__getitem__,
        ),
        patch.object(
            FSKSignalGenerator,
            "set_default_class_name",
        ) as set_class_name,
    ):
        generator = FSKSignalGenerator(**metadata)

    base_init.assert_called_once_with(
        generator,
        **metadata,
    )
    set_class_name.assert_called_once_with("4gfsk")

    assert generator.required_metadata_fields == [
        "sample_rate",
        "bandwidth_min",
        "bandwidth_max",
        "fsk_type",
        "constellation_size",
        "signal_duration_in_samples_min",
        "signal_duration_in_samples_max",
    ]


def test_fsk_signal_generator_generate():
    """The generator should sample parameters and construct a Signal."""
    metadata = {
        "sample_rate": 10_000,
        "bandwidth_min": 500,
        "bandwidth_max": 1_000,
        "fsk_type": "gmsk",
        "constellation_size": 2,
        "signal_duration_in_samples_min": 100,
        "signal_duration_in_samples_max": 200,
    }

    rng = MagicMock(spec=np.random.Generator)
    rng.integers.side_effect = [
        150,
        800,
    ]

    class GeneratorStub:
        random_generator = rng

        def __getitem__(self, key):
            return metadata[key]

    signal_data = np.ones(150, dtype=TorchSigComplexDataType)
    expected_signal = MagicMock()

    with (
        patch(
            f"{MODULE_PATH}.fsk_modulator",
            return_value=signal_data,
        ) as modulator,
        patch(
            f"{MODULE_PATH}.Signal",
            return_value=expected_signal,
        ) as signal_class,
    ):
        result = FSKSignalGenerator.generate(GeneratorStub())

    assert rng.integers.call_args_list == [
        call(low=100, high=201),
        call(low=500, high=1_001),
    ]

    modulator.assert_called_once_with(
        2,
        "gmsk",
        800,
        10_000,
        150,
        rng,
    )

    signal_class.assert_called_once_with(
        data=signal_data,
        center_freq=0,
        bandwidth=800,
    )

    assert result is expected_signal
