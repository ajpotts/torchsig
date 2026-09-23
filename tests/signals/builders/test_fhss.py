"""Tests for the basic frequency-hopping signal builder."""

import numpy as np
import pytest

from torchsig.signals.builders.fhss import (
    FrequencyHoppingSignalGenerator,
    frequency_hopping_modulator,
)
from torchsig.utils.dsp import TorchSigComplexDataType


@pytest.mark.parametrize(
    ("num_samples", "sample_rate", "bandwidth", "message"),
    [
        (0, 100.0, 20.0, "num_samples must be positive"),
        (16, 0.0, 20.0, "sample_rate must be positive"),
        (16, 100.0, 0.0, "bandwidth must be positive"),
        (16, 100.0, 101.0, "bandwidth must not exceed sample_rate"),
    ],
)
def test_frequency_hopping_modulator_rejects_invalid_inputs(num_samples, sample_rate, bandwidth, message):
    with pytest.raises(ValueError, match=message):
        frequency_hopping_modulator(num_samples, sample_rate, bandwidth)


def test_frequency_hopping_modulator_shape_dtype_and_magnitude():
    data = frequency_hopping_modulator(
        257,
        1_000.0,
        400.0,
        np.random.default_rng(4),
    )

    assert data.shape == (257,)
    assert data.dtype == np.dtype(TorchSigComplexDataType)
    np.testing.assert_allclose(np.abs(data), 1.0, rtol=1e-6, atol=1e-6)


def test_frequency_hopping_modulator_is_seeded_and_hops():
    first = frequency_hopping_modulator(512, 1_000.0, 400.0, np.random.default_rng(9))
    second = frequency_hopping_modulator(512, 1_000.0, 400.0, np.random.default_rng(9))

    np.testing.assert_array_equal(first, second)
    phase_steps = np.diff(np.unwrap(np.angle(first)))
    assert np.unique(np.round(phase_steps, decimals=5)).size > 1


def test_frequency_hopping_signal_generator_uses_no_hopping_parameters():
    generator = FrequencyHoppingSignalGenerator(
        metadata={
            "sample_rate": 1_000.0,
            "bandwidth_min": 200.0,
            "bandwidth_max": 400.0,
            "signal_duration_in_samples_min": 128,
            "signal_duration_in_samples_max": 256,
        },
        seed=3,
    )

    generator.validate_metadata_fields()
    signal = generator()

    assert signal.class_name == "fhss"
    assert 128 <= len(signal.data) <= 256
    assert 200.0 <= signal.bandwidth <= 400.0
    assert signal.center_freq == 0
