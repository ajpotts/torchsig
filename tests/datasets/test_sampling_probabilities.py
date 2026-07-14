import pytest

from torchsig.datasets.datasets import TorchSigIterableDataset
from torchsig.signals.builder import BaseSignalGenerator
from torchsig.signals.signal_types import Signal
from torchsig.utils.defaults import TorchSigDefaults


class DummyGenerator(BaseSignalGenerator):
    def __call__(self):
        return Signal(class_name=self.class_name)


def _metadata():
    metadata = TorchSigDefaults().default_dataset_metadata.copy()
    metadata["num_signals_min"] = 1
    metadata["num_signals_max"] = 1
    return metadata


def test_add_signal_generator_rejects_probability_sum_greater_than_one():
    dataset = TorchSigIterableDataset(
        metadata=_metadata(),
        signal_generators=[],
        validate_init=False,
    )

    dataset.add_signal_generator(
        DummyGenerator(class_name="a"),
        probability=0.75,
    )

    with pytest.raises(ValueError, match="sum to 1.0 or less"):
        dataset.add_signal_generator(
            DummyGenerator(class_name="b"),
            probability=0.50,
        )


def test_add_signal_generator_rejects_missing_probability_after_probability_mode():
    dataset = TorchSigIterableDataset(
        metadata=_metadata(),
        signal_generators=[],
        validate_init=False,
    )

    dataset.add_signal_generator(
        DummyGenerator(class_name="a"),
        probability=0.5,
    )

    with pytest.raises(ValueError, match="must specify probability"):
        dataset.add_signal_generator(
            DummyGenerator(class_name="b"),
        )


def test_add_signal_generator_rejects_probability_after_likelihood_mode():
    dataset = TorchSigIterableDataset(
        metadata=_metadata(),
        signal_generators=[],
        validate_init=False,
    )

    dataset.add_signal_generator(
        DummyGenerator(class_name="a"),
        likelihood=1.0,
    )

    with pytest.raises(ValueError, match="Cannot mix explicit probability"):
        dataset.add_signal_generator(
            DummyGenerator(class_name="b"),
            probability=0.5,
        )


def test_add_signal_generator_accepts_complete_explicit_probabilities():
    dataset = TorchSigIterableDataset(
        metadata=_metadata(),
        signal_generators=[],
        validate_init=False,
    )

    dataset.add_signal_generator(
        DummyGenerator(class_name="a"),
        probability=0.25,
    )
    dataset.add_signal_generator(
        DummyGenerator(class_name="b"),
        probability=0.75,
    )

    dataset._validate_signal_sampling_configuration(require_complete=True)

    assert dataset.signal_probabilities.tolist() == [0.25, 0.75]
