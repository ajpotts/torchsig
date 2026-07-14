# tests/datasets/test_datasets.py

import warnings
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from torchsig.datasets.datasets import (
    SafeTorchSigIterableDataset,
    StaticTorchSigDataset,
    TorchSigIterableDataset,
    apply_label_to_signal,
    apply_transforms_and_labels_to_signal,
)
from torchsig.signals.builder import BaseSignalGenerator
from torchsig.signals.signal_types import Signal
from torchsig.transforms.transforms import ComplexTo2D
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from torchsig.utils.defaults import TorchSigDefaults
from torchsig.utils.writer import DatasetCreator

# =============================================================================
# Helpers
# =============================================================================

class DummyGenerator(BaseSignalGenerator):
    def __call__(self):
        return Signal(class_name=self.class_name)


class ValidatingGenerator(BaseSignalGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.validate_metadata_fields = MagicMock()

    def __call__(self):
        return Signal()


class NonValidatingGenerator(BaseSignalGenerator):
    def __call__(self):
        return Signal()


def _dataset_with_empty_generators():
    return TorchSigIterableDataset(
        metadata=TorchSigDefaults().default_dataset_metadata.copy(),
        signal_generators=[],
        validate_init=False,
    )


def _safe_dataset():
    return SafeTorchSigIterableDataset(
        metadata=TorchSigDefaults().default_dataset_metadata.copy(),
        signal_generators=[],
        validate_init=False,
    )


def _small_metadata():
    metadata = TorchSigDefaults().default_dataset_metadata.copy()
    metadata.update(
        {
            "num_iq_samples_dataset": 16,
            "fft_size": 4,
            "fft_stride": 4,
            "num_signals_min": 1,
            "num_signals_max": 1,
            "sample_rate": 16,
            "frequency_min": -8,
            "frequency_max": 8,
            "signal_center_freq_min": -4,
            "signal_center_freq_max": 4,
            "bandwidth_min": 1,
            "bandwidth_max": 4,
        }
    )
    return metadata


def _parent_with_components(num_signals_max=2):
    parent = Signal(
        data=np.zeros(100, dtype=np.complex64),
        component_signals=[
            Signal(
                data=np.zeros(20, dtype=np.complex64),
                class_name="bpsk",
                class_index=3,
                start_in_samples=10,
                center_freq=100.0,
                bandwidth=40.0,
                _lower_frequency=80.0,
                _upper_frequency=120.0,
            ),
            Signal(
                data=np.zeros(30, dtype=np.complex64),
                class_name="qpsk",
                class_index=4,
                start_in_samples=50,
                center_freq=-200.0,
                bandwidth=60.0,
                _lower_frequency=-230.0,
                _upper_frequency=-170.0,
            ),
        ],
        num_iq_samples_dataset=100,
        num_signals_max=num_signals_max,
        class_names=np.array(["ook", "fm", "am", "bpsk", "qpsk"]),
    )

    for component in parent.component_signals:
        component.add_parent(parent, register=False)

    return parent


# =============================================================================
# apply_label_to_signal / apply_transforms_and_labels_to_signal
# =============================================================================

def test_apply_label_to_signal_uses_signal_properties():
    sample = _parent_with_components()

    assert apply_label_to_signal(sample, "class_name") == ["bpsk", "qpsk"]
    assert apply_label_to_signal(sample, "class_index") == [3, 4]
    assert apply_label_to_signal(sample, "start") == [0.1, 0.5]
    assert apply_label_to_signal(sample, "stop") == [0.3, 0.8]
    assert apply_label_to_signal(sample, "lower_freq") == [80.0, -230.0]
    assert apply_label_to_signal(sample, "upper_freq") == [120.0, -170.0]


def test_apply_label_to_signal_does_not_duplicate_parent_class_index():
    sample = _parent_with_components()
    sample["class_index"] = 99

    assert apply_label_to_signal(sample, "class_index") == [3, 4]


def test_apply_label_to_signal_leaf_signal_fallback():
    sample = Signal(
        data=np.zeros(50, dtype=np.complex64),
        class_name="bpsk",
        class_index=3,
        start_in_samples=25,
        num_iq_samples_dataset=100,
        num_signals_max=1,
        _lower_frequency=-10.0,
        _upper_frequency=10.0,
    )

    assert apply_label_to_signal(sample, "class_name") == ["bpsk"]
    assert apply_label_to_signal(sample, "class_index") == [3]
    assert apply_label_to_signal(sample, "start") == [0.25]
    assert apply_label_to_signal(sample, "stop") == [0.75]
    assert apply_label_to_signal(sample, "lower_freq") == [-10.0]
    assert apply_label_to_signal(sample, "upper_freq") == [10.0]


def test_apply_transforms_and_labels_none_returns_signal():
    sample = _parent_with_components()

    assert apply_transforms_and_labels_to_signal(sample, [], None) is sample


def test_apply_transforms_and_labels_empty_returns_data():
    sample = _parent_with_components()
    result = apply_transforms_and_labels_to_signal(sample, [], [])

    assert isinstance(result, np.ndarray)
    assert result.shape == sample.data.shape


def test_apply_transforms_and_labels_single_target_multi_signal_returns_list():
    sample = _parent_with_components(num_signals_max=2)

    data, targets = apply_transforms_and_labels_to_signal(
        sample,
        [],
        ["class_index"],
    )

    assert data is sample.data
    assert targets == [3, 4]


def test_apply_transforms_and_labels_single_signal_squeezes_single_target():
    sample = Signal(
        data=np.zeros(100, dtype=np.complex64),
        component_signals=[
            Signal(
                data=np.zeros(20, dtype=np.complex64),
                class_index=3,
                num_iq_samples_dataset=100,
            )
        ],
        num_iq_samples_dataset=100,
        num_signals_max=1,
    )
    sample.component_signals[0].add_parent(sample, register=False)

    _, target = apply_transforms_and_labels_to_signal(sample, [], ["class_index"])

    assert target == 3


def test_apply_transforms_and_labels_multiple_targets_parallel_lists():
    sample = _parent_with_components(num_signals_max=2)

    _, targets = apply_transforms_and_labels_to_signal(
        sample,
        [],
        ["class_name", "class_index", "start", "stop", "lower_freq", "upper_freq"],
    )

    assert targets == [
        ["bpsk", "qpsk"],
        [3, 4],
        [0.1, 0.5],
        [0.3, 0.8],
        [80.0, -230.0],
        [120.0, -170.0],
    ]


def test_apply_label_to_signal_class_index_from_class_name_fallback():
    sample = Signal(
        data=np.zeros(100, dtype=np.complex64),
        component_signals=[
            Signal(
                data=np.zeros(20, dtype=np.complex64),
                class_name="qpsk",
                start_in_samples=0,
            )
        ],
        num_iq_samples_dataset=100,
        num_signals_max=1,
        class_names=np.array(["bpsk", "qpsk", "8psk"]),
    )
    sample.component_signals[0].add_parent(sample, register=False)

    assert apply_label_to_signal(sample, "class_index") == [1]


def test_apply_label_to_signal_leaf_class_index_from_class_name_fallback():
    sample = Signal(
        data=np.zeros(100, dtype=np.complex64),
        class_name="8psk",
        num_iq_samples_dataset=100,
        num_signals_max=1,
        class_names=np.array(["bpsk", "qpsk", "8psk"]),
    )

    assert apply_label_to_signal(sample, "class_index") == [2]


# =============================================================================
# TorchSigIterableDataset generation helpers
# =============================================================================

def test_insert_component_signal_uses_relative_signal_slice():
    dataset = TorchSigIterableDataset(
        metadata=_small_metadata(),
        signal_generators=[],
        validate_init=False,
    )

    iq_samples = np.zeros(8, dtype=np.complex64)
    signal = Signal(
        data=np.arange(10, dtype=np.float32).astype(np.complex64),
        center_freq=0,
        bandwidth=1,
    )

    dataset._insert_component_signal(iq_samples, signal, start_sample=5)

    expected = np.zeros(8, dtype=np.complex64)
    expected[5:8] = np.array([0, 1, 2], dtype=np.complex64)

    assert np.array_equal(iq_samples, expected)
    assert signal.start_in_samples == 5
    assert signal.duration_in_samples == 3


def test_insert_component_signal_does_not_truncate_when_signal_fits():
    dataset = TorchSigIterableDataset(
        metadata=_small_metadata(),
        signal_generators=[],
        validate_init=False,
    )

    iq_samples = np.zeros(8, dtype=np.complex64)
    signal = Signal(
        data=np.array([1, 2, 3], dtype=np.complex64),
        center_freq=0,
        bandwidth=1,
    )

    dataset._insert_component_signal(iq_samples, signal, start_sample=2)

    expected = np.zeros(8, dtype=np.complex64)
    expected[2:5] = np.array([1, 2, 3], dtype=np.complex64)

    assert np.array_equal(iq_samples, expected)
    assert signal.start_in_samples == 2
    assert signal.duration_in_samples == 3


def test_choose_start_sample_warns_when_signal_is_too_large():
    dataset = TorchSigIterableDataset(
        metadata=_small_metadata(),
        signal_generators=[],
        validate_init=False,
    )

    iq_samples = np.zeros(8, dtype=np.complex64)
    signal = Signal(data=np.zeros(10, dtype=np.complex64), center_freq=0, bandwidth=1)

    with pytest.warns(UserWarning, match="too large"):
        start_sample = dataset._choose_start_sample(iq_samples, signal)

    assert start_sample == 0


def test_generate_new_signal_sets_component_start_and_clips_duration(monkeypatch):
    dataset = TorchSigIterableDataset(
        metadata=_small_metadata(),
        signal_generators=[],
        validate_init=False,
    )
    dataset["cochannel_overlap_probability"] = 1.0

    component = Signal(
        data=np.ones(20, dtype=np.complex64),
        center_freq=0,
        bandwidth=1,
        class_name="dummy",
        class_index=0,
    )

    monkeypatch.setattr(dataset, "_build_noise_floor", lambda: np.zeros(8, dtype=np.complex64))
    monkeypatch.setattr(dataset, "_generate_component_signal", lambda: component.copy())
    monkeypatch.setattr(dataset, "_choose_start_sample", lambda iq_samples, signal: 5)
    monkeypatch.setattr(dataset, "_map_to_coordinates", lambda signal, start_sample: object())
    monkeypatch.setattr(dataset, "_check_if_overlap", lambda rectangle, rectangles: False)

    sample = dataset.__generate_new_signal__()

    placed = sample.component_signals[0]
    assert placed.start_in_samples == 5
    assert placed.duration_in_samples == 3
    assert np.all(sample.data[5:8] == 1)


def test_iterable_dataset_validate_warns_when_signal_duration_exceeds_dataset_length():
    dataset = TorchSigIterableDataset(
        metadata={
            **TorchSigDefaults().default_dataset_metadata.copy(),
            "num_iq_samples_dataset": 4096,
            "signal_duration_in_samples_min": 3276,
            "signal_duration_in_samples_max": 8192,
        },
        signal_generators=[],
        validate_init=False,
    )

    with pytest.warns(
        UserWarning,
        match="signal_duration_in_samples_max exceeds num_iq_samples_dataset",
    ):
        dataset.validate()


def test_iterable_dataset_validate_allows_signal_duration_equal_to_dataset_length():
    dataset = TorchSigIterableDataset(
        metadata={
            **TorchSigDefaults().default_dataset_metadata.copy(),
            "num_iq_samples_dataset": 4096,
            "signal_duration_in_samples_min": 3276,
            "signal_duration_in_samples_max": 4096,
        },
        signal_generators=[],
        validate_init=False,
    )

    with warnings.catch_warnings(record=True) as warning_records:
        warnings.simplefilter("always")
        dataset.validate()

    assert warning_records == []


# =============================================================================
# Sampling weights / probabilities
# =============================================================================

@pytest.mark.parametrize(
    "value, expected",
    [(1, 1.0), (1.5, 1.5), (np.int64(2), 2.0), (np.float64(2.5), 2.5)],
)
def test_validate_positive_weight_accepts_positive_real_numbers(value, expected):
    assert TorchSigIterableDataset._validate_positive_weight(value, "likelihood") == expected


@pytest.mark.parametrize("value", [0, -1, -0.5, np.int64(0), np.float64(-1.0)])
def test_validate_positive_weight_rejects_nonpositive_values(value):
    with pytest.raises(ValueError, match="likelihood must be > 0"):
        TorchSigIterableDataset._validate_positive_weight(value, "likelihood")


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
def test_validate_positive_weight_rejects_nonfinite_values(value):
    with pytest.raises(ValueError, match="probability must be finite"):
        TorchSigIterableDataset._validate_positive_weight(value, "probability")


@pytest.mark.parametrize("value", ["1.0", None, object(), [1.0]])
def test_validate_positive_weight_rejects_non_numeric_values(value):
    with pytest.raises(TypeError, match="probability must be a real number"):
        TorchSigIterableDataset._validate_positive_weight(value, "probability")


def test_validate_signal_sampling_configuration_allows_empty_dataset():
    _dataset_with_empty_generators()._validate_signal_sampling_configuration()


def test_validate_signal_sampling_configuration_accepts_valid_likelihoods():
    dataset = _dataset_with_empty_generators()
    dataset.add_signal_generator(DummyGenerator(class_name="a"), likelihood=1.0)
    dataset.add_signal_generator(DummyGenerator(class_name="b"), likelihood=2.0)

    dataset._validate_signal_sampling_configuration()


def test_validate_signal_sampling_configuration_rejects_likelihood_count_mismatch():
    dataset = _dataset_with_empty_generators()
    dataset.add_signal_generator(DummyGenerator(class_name="a"), likelihood=1.0)
    dataset.signal_likelihoods = []

    with pytest.raises(ValueError, match="signal likelihood count does not match"):
        dataset._validate_signal_sampling_configuration()


def test_validate_signal_sampling_configuration_rejects_nonpositive_likelihoods():
    dataset = _dataset_with_empty_generators()
    dataset.add_signal_generator(DummyGenerator(class_name="a"), likelihood=1.0)
    dataset.signal_likelihoods = [0.0]

    with pytest.raises(ValueError, match="all signal likelihoods must be > 0"):
        dataset._validate_signal_sampling_configuration()


def test_validate_signal_sampling_configuration_accepts_complete_probabilities():
    dataset = _dataset_with_empty_generators()
    dataset.add_signal_generator(DummyGenerator(class_name="a"), probability=0.25)
    dataset.add_signal_generator(DummyGenerator(class_name="b"), probability=0.75)

    dataset._validate_signal_sampling_configuration(require_complete=True)


def test_validate_signal_sampling_configuration_accepts_incomplete_probabilities_when_not_required():
    dataset = _dataset_with_empty_generators()
    dataset.add_signal_generator(DummyGenerator(class_name="a"), probability=0.25)
    dataset.add_signal_generator(DummyGenerator(class_name="b"), probability=0.25)

    dataset._validate_signal_sampling_configuration(require_complete=False)


def test_validate_signal_sampling_configuration_rejects_incomplete_probabilities_when_required():
    dataset = _dataset_with_empty_generators()
    dataset.add_signal_generator(DummyGenerator(class_name="a"), probability=0.25)
    dataset.add_signal_generator(DummyGenerator(class_name="b"), probability=0.25)

    with pytest.raises(ValueError, match="must sum to 1.0 before sampling"):
        dataset._validate_signal_sampling_configuration(require_complete=True)


def test_validate_signal_sampling_configuration_rejects_probability_sum_greater_than_one():
    dataset = _dataset_with_empty_generators()
    dataset._signal_probability_mode = "probability"
    dataset.signal_generators = [DummyGenerator(class_name="a"), DummyGenerator(class_name="b")]
    dataset.signal_probabilities = np.array([0.75, 0.50])

    with pytest.raises(ValueError, match="signal probabilities must sum to 1.0"):
        dataset._validate_signal_sampling_configuration(require_complete=True)


def test_validate_signal_sampling_configuration_rejects_probability_count_mismatch():
    dataset = _dataset_with_empty_generators()
    dataset._signal_probability_mode = "probability"
    dataset.signal_generators = [DummyGenerator(class_name="a"), DummyGenerator(class_name="b")]
    dataset.signal_probabilities = np.array([1.0])

    with pytest.raises(ValueError, match="signal probability count does not match"):
        dataset._validate_signal_sampling_configuration()


def test_validate_signal_sampling_configuration_rejects_nonpositive_probabilities():
    dataset = _dataset_with_empty_generators()
    dataset._signal_probability_mode = "probability"
    dataset.signal_generators = [DummyGenerator(class_name="a"), DummyGenerator(class_name="b")]
    dataset.signal_probabilities = np.array([0.5, 0.0])

    with pytest.raises(ValueError, match="all signal probabilities must be > 0"):
        dataset._validate_signal_sampling_configuration()


def test_refresh_signal_probabilities_empty():
    dataset = _dataset_with_empty_generators()
    dataset.signal_probabilities = np.array([1.0])

    dataset._refresh_signal_probabilities()

    assert dataset.signal_probabilities.shape == (0,)
    assert dataset.signal_probabilities.dtype == float


def test_refresh_signal_probabilities_probability_mode():
    dataset = _dataset_with_empty_generators()
    dataset.add_signal_generator(DummyGenerator(class_name="a"), probability=0.25)
    dataset.add_signal_generator(DummyGenerator(class_name="b"), probability=0.75)

    dataset._refresh_signal_probabilities()

    np.testing.assert_array_equal(dataset.signal_probabilities, np.array([0.25, 0.75]))


def test_refresh_signal_probabilities_likelihood_mode():
    dataset = _dataset_with_empty_generators()
    dataset.add_signal_generator(DummyGenerator(class_name="a"), likelihood=1.0)
    dataset.add_signal_generator(DummyGenerator(class_name="b"), likelihood=3.0)

    dataset._refresh_signal_probabilities()

    np.testing.assert_allclose(dataset.signal_probabilities, np.array([0.25, 0.75]))


def test_add_signal_generator_rejects_likelihood_and_probability():
    dataset = _dataset_with_empty_generators()

    with pytest.raises(ValueError, match="Specify only one of likelihood or probability"):
        dataset.add_signal_generator(
            DummyGenerator(class_name="a"),
            likelihood=1.0,
            probability=0.5,
        )


def test_add_signal_generator_rejects_missing_probability_after_probability_mode():
    dataset = _dataset_with_empty_generators()
    dataset.add_signal_generator(DummyGenerator(class_name="a"), probability=0.5)

    with pytest.raises(ValueError, match="All signal generators must specify probability"):
        dataset.add_signal_generator(DummyGenerator(class_name="b"))


def test_add_signal_generator_rejects_probability_after_likelihood_mode():
    dataset = _dataset_with_empty_generators()
    dataset.add_signal_generator(DummyGenerator(class_name="a"), likelihood=1.0)

    with pytest.raises(ValueError, match="Cannot mix explicit probability"):
        dataset.add_signal_generator(DummyGenerator(class_name="b"), probability=0.5)


def test_add_signal_generator_rejects_probability_sum_greater_than_one():
    dataset = _dataset_with_empty_generators()
    dataset.add_signal_generator(DummyGenerator(class_name="a"), probability=0.75)

    with pytest.raises(ValueError, match="signal probabilities must sum to 1.0 or less"):
        dataset.add_signal_generator(DummyGenerator(class_name="b"), probability=0.50)


@pytest.mark.parametrize("bad_likelihood", [0.0, -1.0])
def test_add_signal_generator_rejects_nonpositive_likelihood(bad_likelihood):
    dataset = _dataset_with_empty_generators()

    with pytest.raises(ValueError, match="likelihood must be > 0"):
        dataset.add_signal_generator(DummyGenerator(class_name="a"), likelihood=bad_likelihood)


@pytest.mark.parametrize("bad_probability", [0.0, -1.0])
def test_add_signal_generator_rejects_nonpositive_probability(bad_probability):
    dataset = _dataset_with_empty_generators()

    with pytest.raises(ValueError, match="probability must be > 0"):
        dataset.add_signal_generator(DummyGenerator(class_name="a"), probability=bad_probability)


def test_validate_metadata_fields_calls_generator_validate_metadata_fields():
    dataset = _dataset_with_empty_generators()

    generator = ValidatingGenerator(class_name="a")
    generator.validate_metadata_fields.reset_mock()

    dataset.add_signal_generator(generator)
    generator.validate_metadata_fields.reset_mock()

    dataset.validate_metadata_fields()

    generator.validate_metadata_fields.assert_called_once_with()


def test_add_signal_generator_skips_validation_when_validate_init_false():
    dataset = _dataset_with_empty_generators()
    dataset.validate_init = False

    generator = ValidatingGenerator(class_name="a")
    generator.validate_metadata_fields.reset_mock()

    dataset.add_signal_generator(generator)

    generator.validate_metadata_fields.assert_not_called()


def test_add_signal_generator_ignores_missing_validate_metadata_fields():
    dataset = _dataset_with_empty_generators()
    dataset.validate_init = True

    generator = NonValidatingGenerator(class_name="a")
    dataset.add_signal_generator(generator)

    assert dataset.signal_generators == [generator]


# =============================================================================
# TorchSigIterableDataset misc behavior
# =============================================================================

def test_iterable_dataset_call_returns_next_sample(monkeypatch):
    dataset = _dataset_with_empty_generators()
    expected = Signal(data=np.ones(8, dtype=np.complex64))

    monkeypatch.setattr(TorchSigIterableDataset, "__next__", lambda self: expected)

    assert dataset() is expected


def test_iterable_dataset_repr_includes_core_fields():
    dataset = _dataset_with_empty_generators()

    result = repr(dataset)

    assert result.startswith("TorchSigIterableDataset(")
    assert "metadata=" in result
    assert "transforms=" in result
    assert "signal_generators=" in result
    assert result.endswith(")")


# =============================================================================
# SafeTorchSigIterableDataset
# =============================================================================

def test_safe_iterable_dataset_set_fallback_policy_defaults_to_original():
    dataset = _safe_dataset()

    dataset.set_fallback_policy()

    assert dataset.pipeline_fallback == "original"


@pytest.mark.parametrize("fallback", ["original", "zero"])
def test_safe_iterable_dataset_set_fallback_policy_without_retries(fallback):
    dataset = _safe_dataset()

    dataset.set_fallback_policy(fallback=fallback)

    assert dataset.pipeline_fallback == fallback


def test_safe_iterable_dataset_set_fallback_policy_retry_sets_max_retries():
    dataset = _safe_dataset()

    dataset.set_fallback_policy(fallback="retry", max_retries=5)

    assert dataset.pipeline_fallback == "retry"
    assert dataset.pipeline_max_retries == 5


def test_safe_iterable_dataset_set_fallback_policy_retry_without_max_retries_preserves_existing_value():
    dataset = _safe_dataset()
    dataset.pipeline_max_retries = 7

    dataset.set_fallback_policy(fallback="retry")

    assert dataset.pipeline_fallback == "retry"
    assert dataset.pipeline_max_retries == 7


@pytest.mark.parametrize("fallback", ["original", "zero"])
def test_safe_iterable_dataset_set_fallback_policy_rejects_retries_without_retry_mode(fallback):
    dataset = _safe_dataset()

    with pytest.raises(ValueError, match="max_retries is only allowed with fallback='retry'"):
        dataset.set_fallback_policy(fallback=fallback, max_retries=3)



# =============================================================================
# StaticTorchSigDataset
# =============================================================================

def test_static_dataset_getitem_raises_index_error_for_out_of_bounds(tmp_path):
    root = tmp_path / "static_dataset"
    root.mkdir()

    dataset = StaticTorchSigDataset.__new__(StaticTorchSigDataset)
    dataset.root = root
    dataset.dataset_length = 3

    with pytest.raises(IndexError, match=r"Index -1 is out of bounds"):
        dataset[-1]

    with pytest.raises(IndexError, match=r"Index 3 is out of bounds"):
        dataset[3]


def test_static_dataset_verify_raises_for_missing_root(tmp_path):
    dataset = StaticTorchSigDataset.__new__(StaticTorchSigDataset)
    dataset.root = tmp_path / "does_not_exist"

    with pytest.raises(ValueError, match=r"root does not exist:"):
        dataset._verify()


def test_static_dataset_verify_accepts_existing_root(tmp_path):
    dataset = StaticTorchSigDataset.__new__(StaticTorchSigDataset)
    dataset.root = tmp_path

    dataset._verify()


def test_static_dataset_str():
    dataset = StaticTorchSigDataset.__new__(StaticTorchSigDataset)
    dataset.root = Path("/tmp/test_dataset")

    assert str(dataset) == "StaticTorchSigDataset: /tmp/test_dataset"


def test_static_dataset_repr():
    dataset = StaticTorchSigDataset.__new__(StaticTorchSigDataset)
    dataset.root = Path("/tmp/test_dataset")
    dataset.reader = "DummyReader"

    assert repr(dataset) == (
        "StaticTorchSigDataset("
        "root=/tmp/test_dataset, "
        "file_handler_class=DummyReader)"
    )


# =============================================================================
# Slow static dataset integration/regression tests
# =============================================================================

def test_static_dataset_preserves_property_backed_target_labels(tmp_path):
    sample = _parent_with_components(num_signals_max=3)

    class FakeReader:
        def __init__(self, root):
            self.root = root

        def __len__(self):
            return 1

        def read(self, idx):
            assert idx == 0
            return sample

    static_dataset = StaticTorchSigDataset(
        root=tmp_path,
        file_handler_class=FakeReader,
        target_labels=["class_name", "start", "stop", "lower_freq", "upper_freq"],
    )

    _, targets = static_dataset[0]
    class_names, starts, stops, lower_freqs, upper_freqs = targets

    assert class_names == ["bpsk", "qpsk"]
    assert starts == [0.1, 0.5]
    assert stops == [0.3, 0.8]
    assert lower_freqs == [80.0, -230.0]
    assert upper_freqs == [120.0, -170.0]


def test_static_dataset_class_index_is_single_label_when_one_signal(tmp_path):
    sample = Signal(
        data=np.zeros(16, dtype=np.complex64),
        component_signals=[
            Signal(
                data=np.ones(8, dtype=np.complex64),
                class_index=7,
                num_iq_samples_dataset=16,
            )
        ],
        num_iq_samples_dataset=16,
        num_signals_max=1,
    )
    sample["class_index"] = 99
    sample.component_signals[0].add_parent(sample, register=False)

    class FakeReader:
        def __init__(self, root):
            self.root = root

        def __len__(self):
            return 1

        def read(self, idx):
            assert idx == 0
            return sample

    static_dataset = StaticTorchSigDataset(
        root=tmp_path,
        file_handler_class=FakeReader,
        target_labels=["class_index"],
    )

    _, target = static_dataset[0]

    assert target == 7
    assert isinstance(target, int)


@pytest.mark.slow
def test_static_iq_dataset_class_index_labels_are_valid(tmp_path):
    seed = 123
    dataset_length = 200
    batch_size = 32
    fft_size = 256

    metadata = TorchSigDefaults().default_dataset_metadata.copy()
    metadata.update(
        {
            "num_iq_samples_dataset": fft_size**2,
            "fft_size": fft_size,
            "fft_stride": fft_size,
            "num_signals_max": 1,
            "num_signals_min": 1,
            "noise_power_db": 1,
            "signal_center_freq_min": 1000,
            "signal_center_freq_max": 2000,
            "sample_rate": 10000,
            "frequency_min": 1000,
            "frequency_max": 2000,
            "cochannel_overlap_probability": 0.2,
            "bandwidth_min": 1000,
            "bandwidth_max": 2000,
        }
    )

    iterable = TorchSigIterableDataset(
        metadata=metadata,
        transforms=[ComplexTo2D()],
        target_labels=None,
        signal_generators="all",
    )

    dataloader = WorkerSeedingDataLoader(
        iterable,
        batch_size=batch_size,
        collate_fn=lambda x: x,
        num_workers=1,
    )
    dataloader.seed(seed)

    root = tmp_path / "static_iq_dataset" / "train"

    DatasetCreator(
        dataloader=dataloader,
        root=root,
        overwrite=True,
        dataset_length=dataset_length,
    ).create()

    static_dataset = StaticTorchSigDataset(root=root, target_labels=["class_index"])
    label_counts = Counter()

    for idx in range(dataset_length):
        _, target = static_dataset[idx]

        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().reshape(-1).tolist()
        elif isinstance(target, np.ndarray):
            target = target.reshape(-1).tolist()

        assert not isinstance(target, (list, tuple))
        assert isinstance(target, (int, np.integer))
        assert 0 <= int(target) < len(iterable.class_names)

        label_counts[int(target)] += 1

    assert sum(label_counts.values()) == dataset_length


@pytest.mark.slow
def test_static_iq_dataset_target_labels_are_parallel_and_valid(tmp_path):
    seed = 123
    dataset_length = 200
    batch_size = 32
    fft_size = 256

    metadata = TorchSigDefaults().default_dataset_metadata.copy()
    metadata.update(
        {
            "num_iq_samples_dataset": fft_size**2,
            "fft_size": fft_size,
            "fft_stride": fft_size,
            "num_signals_max": 5,
            "num_signals_min": 1,
            "noise_power_db": 0,
        }
    )

    iterable = TorchSigIterableDataset(
        metadata=metadata,
        target_labels=None,
        signal_generators="all",
    )

    dataloader = WorkerSeedingDataLoader(
        iterable,
        batch_size=batch_size,
        collate_fn=lambda x: x,
        num_workers=1,
    )
    dataloader.seed(seed)

    root = tmp_path / "static_multilabel_dataset" / "train"

    DatasetCreator(
        dataloader=dataloader,
        root=root,
        overwrite=True,
        dataset_length=dataset_length,
    ).create()

    target_labels = [
        "class_name",
        "class_index",
        "start",
        "stop",
        "lower_freq",
        "upper_freq",
        "snr_db",
    ]

    static_dataset = StaticTorchSigDataset(root=root, target_labels=target_labels)

    valid_class_names = set(iterable.class_names)
    num_classes = len(iterable.class_names)

    for idx in range(dataset_length):
        _, targets = static_dataset[idx]

        assert isinstance(targets, list)
        assert len(targets) == len(target_labels)

        class_names, class_indices, starts, stops, lower_freqs, upper_freqs, snrs = targets

        lengths = [len(values) for values in targets]
        assert len(set(lengths)) == 1

        num_labels = lengths[0]
        assert 1 <= num_labels <= metadata["num_signals_max"]

        assert all(name in valid_class_names for name in class_names)
        assert all(isinstance(label, (int, np.integer)) for label in class_indices)
        assert all(0 <= int(label) < num_classes for label in class_indices)
        assert all(0.0 <= float(start) <= 1.0 for start in starts)
        assert all(float(stop) >= float(start) for start, stop in zip(starts, stops))
        assert all(float(lower) <= float(upper) for lower, upper in zip(lower_freqs, upper_freqs))
        assert all(np.isfinite(float(snr)) for snr in snrs)
