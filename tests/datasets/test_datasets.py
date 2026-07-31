"""Unit Tests for datasets"""

import itertools
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, List

import numpy as np
import pytest
import torch
import yaml
from unittest.mock import MagicMock
import warnings

from torchsig.datasets.datasets import (
    SafeTorchSigIterableDataset,
    StaticTorchSigDataset,
    TorchSigIterableDataset,
    apply_label_to_signal,
    apply_transforms_and_labels_to_signal,
)
from torchsig.signals.builder import BaseSignalGenerator
from torchsig.signals.signal_types import Signal
from torchsig.transforms.impairments import Impairments
from torchsig.transforms.metadata_transforms import MultiHotLabel, YOLOLabel
from torchsig.transforms.transforms import ComplexTo2D, Spectrogram
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from torchsig.utils.defaults import TorchSigDefaults
from torchsig.utils.dsp import TorchSigRealDataType
from torchsig.utils.writer import DatasetCreator


# =============================================================================
# Helpers
# =============================================================================


test_dataset_getitem_params = list(
    itertools.product(
        # num_signals_max
        [1, 2, 3],
        # target transforms to test
        [
            #        None,
            #        [],
            ["class_name"],
            ["yolo_label"],
            ["class_name", "snr_db"],
            ["class_name", "yolo_label"],
            ["class_name", "class_index", "start", "stop", "snr_db"],
        ],
        # num_workers
        [0, 2]
    )
)
num_check = 5


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
# Tests
# =============================================================================

def verify_getitem_targets(num_signals_max: int, target_labels: List[str], sample: Any) -> None:
    """Verfies target labels applied correctly

    Target Labels Table

    | Case      | target_labels                  | num_signals_max = 1          | num_signals_max > 1                                               |
    |-----------|--------------------------------|------------------------------|-------------------------------------------------------------------|
    | Case 1    | None                           | nothing, just Signal object  | nothing, just Signal object                                       |
    | Case 2    | []                             | nothing, just returns data   | nothing, just returns data                                        |
    | Case 3    | ["class_name"]                 | '8msk'                       | ['8msk', 'ofdm-600']                                              |
    | Case 4    | ["class_name", "class_index"]  | ('8msk', 0)                  | [('8msk', 0), ('ofdm-600', 1)]                                    |
    | Case 5    | ["class_name", "yolo_label"]   | ('8msk', (idx, x, y, w, h))  | [('8msk', (idx, x, y, w, h)), ('ofdm-600', (idx, x, y, w, h))]    |
    | Case 6    | ["yolo_label"]                 | (idx, x, y, w, h)            | [(idx, x, y, w, h), (idx, x, y, w, h)]                            |


    """
    # target_labels are None or []
    # just return data
    if target_labels is None:
        # Case 1
        assert isinstance(sample, Signal)
    elif len(target_labels) == 0:
        # Case 2
        assert isinstance(sample, np.ndarray)
    else:
        # Case 3-6
        # target_labels has at least 1 item
        data, targets = sample
        print(targets)

        if num_signals_max == 1:
            # one signal
            assert isinstance(targets, tuple) or isinstance(targets, list) or isinstance(targets, float) or isinstance(targets, int) or isinstance(targets, str)
        else:
            # sample has more than one signal
            # targets should be a list
            assert isinstance(targets, list)
            for t in targets:
                assert isinstance(targets, tuple) or isinstance(targets, list) or isinstance(targets, float) or isinstance(targets, int) or isinstance(targets, str)


def test_IterableDataset_transforms():
    seed = 83843293432
    impairments = Impairments(level=2)
    burst_impairments = impairments.signal_transforms
    whole_signal_impairments = impairments.dataset_transforms

    md = TorchSigDefaults().default_dataset_metadata
    md["fft_size"] = 64
    md["fft_stride"] = 64
    md["num_iq_samples_dataset"] = 64**2

    dataset_unimpaired = TorchSigIterableDataset(
        metadata=md,
        transforms=[
            # whole_signal_impairments,
            Spectrogram(fft_size=md["fft_size"]),
        ],
        target_labels=[],
    )
    dataset_whole_impaired = TorchSigIterableDataset(
        metadata=md,
        transforms=[
            whole_signal_impairments,
            Spectrogram(fft_size=md["fft_size"]),
        ],
        target_labels=[],
    )
    dataset_component_impaired = TorchSigIterableDataset(
        metadata=md,
        transforms=[
            Spectrogram(fft_size=md["fft_size"]),
        ],
        component_transforms=[burst_impairments],
        target_labels=[],
    )
    dataset_impaired = TorchSigIterableDataset(
        metadata=md,
        transforms=[
            whole_signal_impairments,
            Spectrogram(fft_size=md["fft_size"]),
        ],
        component_transforms=[burst_impairments],
        target_labels=[],
    )

    datasets = [
        dataset_whole_impaired,
        dataset_unimpaired,
        dataset_component_impaired,
        dataset_impaired
    ]

    for d in datasets:
        d.seed(seed)

    # check they are all different
    datas = [next(d) for d in datasets]

    for i, j in itertools.combinations(range(len(datas)), 2):
        if np.array_equal(datas[i], datas[j]):
            raise AssertionError(f"Datasets {i} and {j} are identical")


@pytest.mark.parametrize("num_signals_max, target_labels, num_workers", test_dataset_getitem_params)
def test_IterableDataset_getitem(
    num_signals_max: int,
    target_labels: List[str],
    num_workers: int
):
    """Tests targets from target transform are properly returned from dataset's getitem

    >>> pytest test_datasets.py -s

    Args:
        num_signals_max (str): Maximum number of signals.
        target_labels: List[str] (List[TargetTransform]): target labels to test.
        num_workers (int): Number of worker processes for the dataloader.
    """
    print(f"\nnum_signals_max={num_signals_max}, target_labels={target_labels}, num_workers={num_workers}")
    dm = TorchSigDefaults().default_dataset_metadata
    dataset = TorchSigIterableDataset(metadata=dm, transforms=[YOLOLabel()])

    for _ in range(num_check):
        sample = next(dataset)
        # data, targets = sample.data, [x.to_dict() for x in sample.get_full_metadata()]

        verify_getitem_targets(num_signals_max, None, sample)


@pytest.mark.parametrize("num_signals_max, target_labels, num_workers", test_dataset_getitem_params)
def test_StaticDataset_getitem(tmp_path, num_signals_max: int, target_labels: List[str], num_workers: int):
    """Tests targets from target transform are properly returned from dataset's getitem

    >>> pytest test_datasets.py -s

    Args:
        num_signals_max (int): Maximum number of signals.
        target_labels (List[TargetTransform]): target labels to test.
        num_workers (int): Number of worker processes for the dataloader.
    """
    print(f"\nnum_signals_max={num_signals_max}, target_labels={target_labels}, num_workers={num_workers}")
    if target_labels is None or len(target_labels) == 0:
        # skip
        return
    root = tmp_path / "run0"
    num_generate = num_check * 2

    dm = TorchSigDefaults().default_dataset_metadata
    new_dataset = TorchSigIterableDataset(metadata=dm, transforms=[YOLOLabel()])
    new_dataloader = WorkerSeedingDataLoader(new_dataset, num_workers=num_workers)
    dc = DatasetCreator(dataloader=new_dataloader, root=root, overwrite=True, dataset_length=num_generate)

    dc.create()

    static_dataset = StaticTorchSigDataset(root=root)

    for i in range(num_check):
        idx = np.random.randint(len(static_dataset))
        sample = static_dataset[idx]

        # verify_getitem_targets(num_signals_max, target_labels, sample)


@pytest.mark.parametrize("params, is_error",
                        [({"dataset_length": 10, "num_workers": 0}, False),
                         ({"dataset_length": 10, "num_workers": 2}, False)]
    )
def test_datasets(tmp_path, params: dict, is_error: bool) -> None:
    """Test datasets with pytest - TorchSigIterableDataset and StaticTorchSigDataset.

    Args:
        is_error (bool): Is a test error expected.

    Raises:
        AssertionError: If unexpected test outcome.

    """
    root0 = tmp_path / "run0"

    seed = 123456789
    rng = np.random.default_rng(seed)
    dataset_length = params["dataset_length"]
    num_workers = params["num_workers"]
    fft_size = rng.integers(128, 1024, dtype=int)
    transforms = [Spectrogram(fft_size=fft_size)]

    md = TorchSigDefaults().default_dataset_metadata

    if is_error:
        with pytest.raises(Exception, match=r".*"):
            DS = TorchSigIterableDataset(metadata=md, target_labels=["class_index"], transforms=transforms)
            DL = WorkerSeedingDataLoader(DS, num_workers=num_workers, collate_fn=default_collate_fn)
            DL.seed(seed)
            dc = DatasetCreator(dataloader=DL, root=root0, dataset_length=dataset_length, overwrite=True)
            dc.create()
            SDS = StaticTorchSigDataset(
                root=root0,
            )
    else:
        # create the dataset object, derived from the metadata object
        DS0 = TorchSigIterableDataset(metadata=deepcopy(md), target_labels=None, seed=seed, transforms=transforms)

        # save dataset to disk
        DL0 = WorkerSeedingDataLoader(DS0, num_workers=num_workers, collate_fn=lambda x: x)
        DL0.seed(seed)
        dc = DatasetCreator(dataloader=DL0, root=root0, dataset_length=dataset_length, overwrite=True)
        dc.create()

        # load dataset from disk
        SDS0 = StaticTorchSigDataset(root=root0, target_labels=["class_index"])
        SDS1 = StaticTorchSigDataset(root=root0, target_labels=["class_index"])

        # dataset
        assert isinstance(DS0, TorchSigIterableDataset)

        # static dataset
        assert isinstance(SDS0, StaticTorchSigDataset)
        assert len(SDS0) == dataset_length
        for i in range(dataset_length):
            data0, meta0 = SDS0[i]
            data1, meta1 = SDS1[i]  # reproducible copy

            assert type(data0) == np.ndarray
            assert data0.dtype == TorchSigRealDataType
            assert meta0 == meta1
            assert np.allclose(data0, data1, 1e-6)

        ds_yaml = yaml.safe_load(open(root0 / "dataset_info.yaml", "r")) or {}
        assert ds_yaml["dataset_length"] == dataset_length



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


def test_apply_label_to_signal_prefers_direct_sample_level_label():
    sample = _parent_with_components()
    sample["multi_hot_label"] = np.array([0, 1, 0, 1], dtype=np.float32)

    values = apply_label_to_signal(sample, "multi_hot_label")

    assert len(values) == 1
    np.testing.assert_array_equal(values[0], sample.multi_hot_label)


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


def test_apply_transforms_and_labels_returns_wideband_multi_hot_vector():
    sample = _parent_with_components(num_signals_max=2)

    data, target = apply_transforms_and_labels_to_signal(
        sample,
        [MultiHotLabel(num_classes=6)],
        ["multi_hot_label"],
    )

    assert data is sample.data
    np.testing.assert_array_equal(
        target,
        np.array([0, 0, 0, 1, 1, 0], dtype=np.float32),
    )


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


def test_choose_start_sample_allows_last_valid_position(monkeypatch):
    dataset = TorchSigIterableDataset(
        metadata=_small_metadata(),
        signal_generators=[],
        validate_init=False,
    )

    iq_samples = np.zeros(8, dtype=np.complex64)
    signal = Signal(
        data=np.ones(3, dtype=np.complex64),
        center_freq=0,
        bandwidth=1,
    )

    class StubRandomGenerator:
        def integers(self, *, low, high):
            assert low == 0
            assert high == 6
            return high - 1

    monkeypatch.setattr(
        dataset,
        "random_generator",
        StubRandomGenerator(),
    )

    start_sample = dataset._choose_start_sample(
        iq_samples,
        signal,
    )

    assert start_sample == 5


def _insert_component_signal(
    self,
    iq_samples: np.ndarray,
    signal: Signal,
    start_sample: int,
) -> None:
    """Insert a component signal into the dataset IQ buffer."""
    stop_sample = min(
        start_sample + len(signal.data),
        len(iq_samples),
    )
    num_samples_to_add = stop_sample - start_sample

    if num_samples_to_add < len(signal.data):
        signal.data = signal.data[:num_samples_to_add]
        signal["duration_in_samples"] = num_samples_to_add

    iq_samples[start_sample:stop_sample] += signal.data
    signal["start_in_samples"] = start_sample


def test_insert_component_signal_truncates_component_data():
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

    dataset._insert_component_signal(
        iq_samples,
        signal,
        start_sample=5,
    )

    expected_component_data = np.array(
        [0, 1, 2],
        dtype=np.complex64,
    )
    expected_iq_samples = np.zeros(8, dtype=np.complex64)
    expected_iq_samples[5:8] = expected_component_data

    np.testing.assert_array_equal(
        signal.data,
        expected_component_data,
    )
    np.testing.assert_array_equal(
        iq_samples,
        expected_iq_samples,
    )
    assert signal.start_in_samples == 5
    assert signal.duration_in_samples == 3
    assert len(signal.data) == signal.duration_in_samples


def test_iterable_dataset_warns_when_max_signal_duration_exceeds_sample_length():
    metadata = _small_metadata()
    metadata["num_iq_samples_dataset"] = 4096
    metadata["signal_duration_in_samples_max"] = 262144

    with pytest.warns(
        UserWarning,
        match=(
            "signal_duration_in_samples_max exceeds "
            "num_iq_samples_dataset"
        ),
    ):
        TorchSigIterableDataset(
            metadata=metadata,
            signal_generators=[],
            validate_init=True,
        )


def test_iterable_dataset_does_not_warn_when_max_signal_duration_equals_sample_length():
    metadata = _small_metadata()
    metadata["num_iq_samples_dataset"] = 4096
    metadata["signal_duration_in_samples_max"] = 4096

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")

        TorchSigIterableDataset(
            metadata=metadata,
            signal_generators=[],
            validate_init=True,
        )

    matching_warnings = [
        warning
        for warning in caught_warnings
        if (
            "signal_duration_in_samples_max exceeds "
            "num_iq_samples_dataset"
        )
        in str(warning.message)
    ]

    assert matching_warnings == []


def test_validate_signal_duration_limits_warns_when_max_exceeds_sample_length():
    metadata = _small_metadata()
    metadata["num_iq_samples_dataset"] = 4096
    metadata["signal_duration_in_samples_max"] = 262144

    dataset = TorchSigIterableDataset(
        metadata=metadata,
        signal_generators=[],
        validate_init=False,
    )

    with pytest.warns(
        UserWarning,
        match=(
            "signal_duration_in_samples_max exceeds "
            "num_iq_samples_dataset"
        ),
    ):
        dataset._validate_signal_duration_limits()


def test_validate_signal_duration_limits_allows_equal_sample_length():
    metadata = _small_metadata()
    metadata["num_iq_samples_dataset"] = 4096
    metadata["signal_duration_in_samples_max"] = 4096

    dataset = TorchSigIterableDataset(
        metadata=metadata,
        signal_generators=[],
        validate_init=False,
    )

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        dataset._validate_signal_duration_limits()

    assert caught_warnings == []


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


# =============================================================================
# Immutable Defaults
# =============================================================================

def test_iterable_dataset_transform_defaults_are_not_shared():
    first = _dataset_with_empty_generators()
    second = _dataset_with_empty_generators()

    first.transforms.append(object())
    first.component_transforms.append(object())

    assert second.transforms == []
    assert second.component_transforms == []


def test_iterable_dataset_accepts_explicit_transform_lists():
    transform = MagicMock()
    component_transform = MagicMock()

    dataset = TorchSigIterableDataset(
        metadata=TorchSigDefaults().default_dataset_metadata.copy(),
        signal_generators=[],
        transforms=[transform],
        component_transforms=[component_transform],
        validate_init=False,
    )

    assert dataset.transforms == [transform]
    assert dataset.component_transforms == [component_transform]
