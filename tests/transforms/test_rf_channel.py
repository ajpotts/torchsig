# ruff: noqa: INP001
"""Tests for deterministic generation-time RF/channel augmentation."""

import numpy as np
import pytest

from torchsig.signals.signal_types import Signal
from torchsig.transforms.rf_channel import FadingConfig, RFChannelImpairmentConfig, RFChannelImpairmentPipeline, UniformRange


def tone(length: int = 4096) -> Signal:
    data = np.exp(2j * np.pi * 0.1 * np.arange(length)).astype(np.complex64)
    return Signal(data=data)


def test_disabled_pipeline_preserves_signal_and_metadata():
    signal = tone()
    original = signal.data.copy()

    result = RFChannelImpairmentPipeline().apply(signal, record_identity=7, sample_rate=1_000.0)

    assert result is signal
    np.testing.assert_array_equal(result.data, original)
    assert "rf_channel_impairments" not in result.keys()  # noqa: SIM118


def test_frequency_offset_is_applied_in_hz_and_recorded():
    config = RFChannelImpairmentConfig(frequency_offset_hz=UniformRange(50.0, 50.0))
    result = RFChannelImpairmentPipeline(config, global_seed=3).apply(tone(), record_identity="record-1", sample_rate=1_000.0)

    phase_step = np.angle(result.data[1] * np.conj(result.data[0])) / (2 * np.pi)
    assert phase_step == pytest.approx(0.15, abs=1e-6)
    assert result["rf_channel_impairments"]["applied"][0]["offset_hz"] == 50.0


def test_phase_noise_is_deterministic_and_changes_phase():
    config = RFChannelImpairmentConfig(phase_noise_degrees=UniformRange(4.0, 4.0))
    pipeline = RFChannelImpairmentPipeline(config, global_seed=9)
    first = pipeline.apply(tone(), record_identity=11, sample_rate=1_000.0)
    second = pipeline.apply(tone(), record_identity=11, sample_rate=1_000.0)

    np.testing.assert_array_equal(first.data, second.data)
    assert not np.array_equal(first.data, tone().data)


def test_fading_is_deterministic_and_recorded():
    config = RFChannelImpairmentConfig(fading=FadingConfig(UniformRange(0.2, 0.2), (1.0, 0.25)))
    pipeline = RFChannelImpairmentPipeline(config, global_seed=12)
    first = pipeline.apply(tone(), record_identity="a", sample_rate=1_000.0)
    second = pipeline.apply(tone(), record_identity="a", sample_rate=1_000.0)

    np.testing.assert_array_equal(first.data, second.data)
    assert first["rf_channel_impairments"]["applied"][0]["model"] == "rayleigh"


def test_combined_order_and_post_normalization():
    config = RFChannelImpairmentConfig(
        frequency_offset_hz=UniformRange(-10.0, 10.0),
        phase_noise_degrees=UniformRange(1.0, 2.0),
        fading=FadingConfig(UniformRange(0.1, 0.2)),
        noise_power_db=UniformRange(-40.0, -30.0),
        normalization="after",
    )
    result = RFChannelImpairmentPipeline(config, global_seed=77).apply(tone(), record_identity=4, sample_rate=2_000.0)

    metadata = result["rf_channel_impairments"]
    assert metadata["order"] == ["frequency_offset", "phase_noise", "fading", "awgn", "normalize"]
    assert np.mean(np.abs(result.data) ** 2) == pytest.approx(1.0, abs=1e-6)


def test_per_effect_sampling_does_not_depend_on_other_enabled_effects():
    phase = UniformRange(1.0, 5.0)
    phase_only = RFChannelImpairmentPipeline(RFChannelImpairmentConfig(phase_noise_degrees=phase), global_seed=8)
    combined = RFChannelImpairmentPipeline(
        RFChannelImpairmentConfig(frequency_offset_hz=UniformRange(1.0, 2.0), phase_noise_degrees=phase),
        global_seed=8,
    )
    phase_value = phase_only.apply(tone(), record_identity=2, sample_rate=1_000.0)["rf_channel_impairments"]["applied"][0]
    combined_value = combined.apply(tone(), record_identity=2, sample_rate=1_000.0)["rf_channel_impairments"]["applied"][1]

    assert phase_value == combined_value


@pytest.mark.parametrize(
    ("value", "error"),
    [
        ({"phase_noise_degrees": [-1, 1]}, "phase noise"),
        ({"fading": {"coherence_bandwidth": [0, 0.1]}}, "coherence bandwidth"),
        ({"normalization": "sometimes"}, "normalization"),
        ({"unknown": 1}, "unknown"),
    ],
)
def test_invalid_configuration_fails_eagerly(value, error):
    with pytest.raises((TypeError, ValueError), match=error):
        RFChannelImpairmentConfig.from_dict(value)


def test_frequency_offset_outside_nyquist_fails_before_mutation():
    signal = tone()
    original = signal.data.copy()
    pipeline = RFChannelImpairmentPipeline(RFChannelImpairmentConfig(frequency_offset_hz=UniformRange(500.0, 500.0)))

    with pytest.raises(ValueError, match="Nyquist"):
        pipeline.apply(signal, record_identity=1, sample_rate=1_000.0)
    np.testing.assert_array_equal(signal.data, original)
