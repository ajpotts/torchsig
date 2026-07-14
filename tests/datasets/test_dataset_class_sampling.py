"""Unit tests for configurable class-selection probabilities."""

from collections import Counter

import numpy as np
import pytest

from torchsig.datasets.datasets import TorchSigIterableDataset
from torchsig.signals.builder import BaseSignalGenerator
from torchsig.signals.signal_types import Signal
from torchsig.utils.defaults import TorchSigDefaults


class DummySignalGenerator(BaseSignalGenerator):
    """Minimal generator used to test sampling probabilities only."""

    def __init__(self, class_name: str, **kwargs):
        super().__init__(**kwargs)
        self["class_name"] = class_name

    def generate(self) -> Signal:
        return Signal(
            data=np.ones(32, dtype=np.complex64),
            center_freq=0.0,
            bandwidth=1.0,
        )


def make_dataset() -> TorchSigIterableDataset:
    """Create a minimal dataset with no default generators."""
    metadata = TorchSigDefaults().default_dataset_metadata
    metadata["num_iq_samples_dataset"] = 128
    metadata["fft_size"] = 32
    metadata["fft_stride"] = 32
    metadata["num_signals_min"] = 1
    metadata["num_signals_max"] = 1
    metadata["cochannel_overlap_probability"] = 0.0
    metadata["noise_power_db"] = 0.0
    metadata["snr_db_min"] = 0.0
    metadata["snr_db_max"] = 0.0
    return TorchSigIterableDataset(signal_generators=[], metadata=metadata)


def sample_generator_frequencies(
    dataset: TorchSigIterableDataset,
    num_draws: int = 1000,
) -> dict[str, float]:
    """Draw many class selections and return empirical frequencies."""
    dataset.seed(2026)
    counts = Counter(
        dataset._random_signal_generator().class_name for _ in range(num_draws)
    )
    return {
        class_name: counts[class_name] / num_draws
        for class_name in dataset["class_names"]
    }


def test_default_add_signal_generator_probabilities_are_uniform():
    dataset = make_dataset()
    for class_name in ["bpsk", "qpsk", "8psk"]:
        dataset.add_signal_generator(
            DummySignalGenerator(class_name),
            class_name=class_name,
        )

    expected = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
    assert np.allclose(dataset.signal_probabilities, expected)

    empirical = sample_generator_frequencies(dataset)
    for class_name in ["bpsk", "qpsk", "8psk"]:
        assert abs(empirical[class_name] - (1 / 3)) < 0.05


def test_likelihoods_remain_backward_compatible():
    dataset = make_dataset()
    dataset.add_signal_generator(
        DummySignalGenerator("bpsk"),
        class_name="bpsk",
        likelihood=2,
    )
    dataset.add_signal_generator(
        DummySignalGenerator("qpsk"),
        class_name="qpsk",
        likelihood=1,
    )
    dataset.add_signal_generator(
        DummySignalGenerator("8psk"),
        class_name="8psk",
        likelihood=1,
    )

    expected = np.array([0.50, 0.25, 0.25], dtype=float)
    assert np.allclose(dataset.signal_probabilities, expected)

    empirical = sample_generator_frequencies(dataset)
    assert abs(empirical["bpsk"] - 0.50) < 0.05
    assert abs(empirical["qpsk"] - 0.25) < 0.05
    assert abs(empirical["8psk"] - 0.25) < 0.05


def test_explicit_probabilities_are_honored():
    dataset = make_dataset()
    dataset.add_signal_generator(
        DummySignalGenerator("bpsk"),
        class_name="bpsk",
        probability=0.60,
    )
    dataset.add_signal_generator(
        DummySignalGenerator("qpsk"),
        class_name="qpsk",
        probability=0.25,
    )
    dataset.add_signal_generator(
        DummySignalGenerator("8psk"),
        class_name="8psk",
        probability=0.15,
    )

    expected = np.array([0.60, 0.25, 0.15], dtype=float)
    assert np.allclose(dataset.signal_probabilities, expected)

    empirical = sample_generator_frequencies(dataset)
    assert abs(empirical["bpsk"] - 0.60) < 0.05
    assert abs(empirical["qpsk"] - 0.25) < 0.05
    assert abs(empirical["8psk"] - 0.15) < 0.05


def test_explicit_probabilities_must_sum_to_one_before_sampling():
    dataset = make_dataset()
    dataset.add_signal_generator(
        DummySignalGenerator("bpsk"),
        class_name="bpsk",
        probability=0.60,
    )
    dataset.add_signal_generator(
        DummySignalGenerator("qpsk"),
        class_name="qpsk",
        probability=0.25,
    )

    with pytest.raises(ValueError, match="sum to 1.0 before sampling"):
        dataset._random_signal_generator()
