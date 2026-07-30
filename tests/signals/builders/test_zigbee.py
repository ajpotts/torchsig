"""Unit tests for Zigbee signal builder."""

import numpy as np
import pytest

from torchsig.signals.builders.zigbee import (
    ZIGBEE_CHIP_SEQS,
    ZigBeeSignalGenerator,
    build_zigbee_chip_stream,
    zigbee_modulator,
)

from torchsig.signals.builders.constellation_maps import all_symbol_maps
from torchsig.signals.signal_lists import CLASS_FAMILY_DICT, TorchSigSignalLists
from torchsig.utils.dsp import TorchSigComplexDataType
from torchsig.utils.signal_building import lookup_signal_generator_by_string

ZIGBEE_METADATA = {
    "sample_rate": 10_000_000,
    "bandwidth_min": 1_500_000,
    "bandwidth_max": 2_500_000,
    "signal_duration_in_samples_min": 4096,
    "signal_duration_in_samples_max": 4096,
}


def test_zigbee_chip_seqs_shape():
    """There are 16 chip sequences, each 32 chips."""
    assert ZIGBEE_CHIP_SEQS.shape == (16, 32)
    assert set(np.unique(ZIGBEE_CHIP_SEQS)).issubset({0, 1})


def test_zigbee_chip_stream_length():
    """Chip stream is exactly the requested length."""
    rng = np.random.default_rng(0)
    stream = build_zigbee_chip_stream(500, rng)
    assert len(stream) == 500


def test_zigbee_modulator_output():
    """The modulator returns finite complex IQ of the right length."""
    rng = np.random.default_rng(42)
    num_samples = 4096
    iq = zigbee_modulator(2_000_000, 10_000_000, num_samples, rng)
    assert iq.dtype == TorchSigComplexDataType
    assert len(iq) == num_samples
    assert np.all(np.isfinite(iq))


def test_zigbee_modulator_invalid_args():
    """Invalid bandwidth/sample-rate raise."""
    with pytest.raises(ValueError):
        zigbee_modulator(0, 10_000_000, 4096)
    with pytest.raises(ValueError):
        zigbee_modulator(6_000_000, 10_000_000, 4096)


def test_zigbee_generator_generate():
    """Generator produces a Signal with correct metadata."""
    signal = ZigBeeSignalGenerator(metadata=ZIGBEE_METADATA, seed=1)()
    assert signal.class_name == "zigbee"
    assert len(signal.data) == ZIGBEE_METADATA["signal_duration_in_samples_min"]


def test_zigbee_generator_reproducible():
    """Same seed yields identical IQ."""
    a = ZigBeeSignalGenerator(metadata=ZIGBEE_METADATA, seed=6).generate()
    b = ZigBeeSignalGenerator(metadata=ZIGBEE_METADATA, seed=6).generate()
    np.testing.assert_array_equal(a.data, b.data)


def test_zigbee_registered_and_in_signal_lists():
    """'zigbee' resolves through lookup and is its own family."""
    assert isinstance(lookup_signal_generator_by_string("zigbee"), ZigBeeSignalGenerator)
    assert CLASS_FAMILY_DICT["zigbee"] == "zigbee"
    lists = TorchSigSignalLists()
    assert "zigbee" in lists.zigbee_signals
