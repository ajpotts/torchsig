"""Tests for structured HDF5 PyTorch materialization."""

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from torchsig.datasets import StructuredHDF5Dataset, materialize_structured_dataset


class _TupleDataset(Dataset):
    def __init__(self, length: int = 5) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        return (
            np.full((2, 3), index, dtype=np.float32),
            {"class": np.int64(index), "valid": np.bool_(index % 2)},
        )


class _Iterable(IterableDataset):
    def __init__(self, length: int) -> None:
        self.length = length

    def __iter__(self):
        for index in range(self.length):
            yield {"x": np.array([index], dtype=np.float32), "y": np.int16(index)}


def _stack_tuple_batch(samples):
    return (
        torch.from_numpy(np.stack([sample[0] for sample in samples])),
        {
            "class": torch.tensor([sample[1]["class"] for sample in samples], dtype=torch.int64),
            "valid": torch.tensor([sample[1]["valid"] for sample in samples], dtype=torch.bool),
        },
    )


def _assert_tuple_sample(actual, index: int) -> None:
    assert isinstance(actual, tuple)
    np.testing.assert_array_equal(actual[0], np.full((2, 3), index, dtype=np.float32))
    assert actual[0].dtype == np.dtype(np.float32)
    assert actual[1]["class"] == index
    assert actual[1]["class"].dtype == np.dtype(np.int64)
    assert actual[1]["valid"] == bool(index % 2)


def test_materializes_dataset_and_preserves_sample_structure(tmp_path) -> None:
    result = materialize_structured_dataset(
        _TupleDataset(),
        tmp_path,
        batch_size=2,
        progress=False,
        writer_kwargs={"compression": None},
    )
    try:
        assert isinstance(result, StructuredHDF5Dataset)
        assert len(result) == 5
        for index in range(5):
            _assert_tuple_sample(result[index], index)
    finally:
        result.close()


def test_consumes_already_collated_dataloader_without_using_field_count(tmp_path) -> None:
    loader = DataLoader(_TupleDataset(), batch_size=3, collate_fn=_stack_tuple_batch)

    result = materialize_structured_dataset(loader, tmp_path, progress=False)
    try:
        assert len(result) == 5
        assert isinstance(result[0], tuple)
        _assert_tuple_sample(result[4], 4)
    finally:
        result.close()


def test_consumes_identity_collated_dataloader(tmp_path) -> None:
    loader = DataLoader(_TupleDataset(), batch_size=2, collate_fn=lambda samples: samples)

    result = materialize_structured_dataset(loader, tmp_path, progress=False)
    try:
        assert len(result) == 5
        _assert_tuple_sample(result[3], 3)
    finally:
        result.close()


def test_dataset_source_supports_custom_collate_and_final_partial_batch(tmp_path) -> None:
    result = materialize_structured_dataset(
        _TupleDataset(7),
        tmp_path,
        batch_size=3,
        collate_fn=_stack_tuple_batch,
        progress=False,
    )
    try:
        assert len(result) == 7
        _assert_tuple_sample(result[6], 6)
    finally:
        result.close()


def test_iterable_dataset_requires_and_accepts_expected_length(tmp_path) -> None:
    source = _Iterable(4)
    with pytest.raises(ValueError, match="expected_length is required"):
        materialize_structured_dataset(source, tmp_path / "missing", progress=False)

    result = materialize_structured_dataset(
        source,
        tmp_path / "complete",
        batch_size=3,
        expected_length=4,
        progress=False,
    )
    try:
        assert len(result) == 4
        np.testing.assert_array_equal(result[3]["x"], [3])
    finally:
        result.close()


@pytest.mark.parametrize(
    ("source", "expected_length", "message"),
    [
        (_Iterable(2), 3, "produced 2 samples; expected 3"),
        (_Iterable(4), 3, "more than the expected 3 samples"),
    ],
)
def test_rejects_too_few_or_too_many_samples(tmp_path, source, expected_length, message) -> None:
    with pytest.raises(RuntimeError, match=message):
        materialize_structured_dataset(
            source,
            tmp_path,
            batch_size=2,
            expected_length=expected_length,
            progress=False,
        )
    with h5py.File(tmp_path / "data.h5", "r") as handle:
        assert not bool(handle.attrs["complete"])


def test_progress_can_be_enabled_or_disabled(tmp_path, capsys) -> None:
    disabled = materialize_structured_dataset(_TupleDataset(2), tmp_path / "disabled", progress=False)
    disabled.close()
    assert "Materializing structured dataset" not in capsys.readouterr().err

    enabled = materialize_structured_dataset(_TupleDataset(2), tmp_path / "enabled", progress=True)
    enabled.close()
    assert "Materializing structured dataset" in capsys.readouterr().err


def test_map_dataset_uses_contiguous_batch_reader(tmp_path, monkeypatch) -> None:
    result = materialize_structured_dataset(_TupleDataset(), tmp_path, progress=False)
    calls = []
    original = result.reader.read_batch

    def record(start, stop):
        calls.append((start, stop))
        return original(start, stop)

    monkeypatch.setattr(result.reader, "read_batch", record)
    try:
        samples = result.__getitems__([1, 2, 3])
        assert calls == [(1, 4)]
        assert len(samples) == 3
    finally:
        result.close()
