"""Tests for the experimental packed HDF5 format."""

import h5py
import numpy as np
import pytest

from torchsig.signals.signal_types import Signal
from torchsig.transforms.transforms import ComplexTo2D, Spectrogram
from torchsig.utils.abstractions import HierarchicalMetadataObject
from torchsig.utils.file_handlers.hdf5_batched import (
    BatchedHDF5Reader,
    BatchedHDF5Writer,
)


def test_batched_hdf5_round_trip_components_and_metadata(tmp_path) -> None:
    grandparent = HierarchicalMetadataObject(
        metadata={"sample_rate": 2_000_000.0, "labels": np.array([1, 2])}
    )
    parent = HierarchicalMetadataObject(
        parent=grandparent,
        metadata={"split": "train"},
    )
    component = Signal(
        data=np.arange(7, dtype=np.complex64),
        parent=parent,
        class_name="component",
        bounds=(1, 4),
    )
    signals = [
        Signal(
            data=(np.arange(16 + idx) + 1j * idx).astype(np.complex64),
            component_signals=[component.copy()],
            parent=parent,
            class_name="sample",
            sample_index=np.int64(idx),
        )
        for idx in range(3)
    ]

    with BatchedHDF5Writer(
        tmp_path,
        shuffle=False,
        fletcher32=False,
        max_batches_in_memory=2,
    ) as writer:
        writer.write(1, signals[2:])
        writer.write(0, signals[:2])

    reader = BatchedHDF5Reader(tmp_path)
    try:
        assert len(reader) == len(signals)
        for idx, expected in enumerate(signals):
            actual = reader.read(idx)
            np.testing.assert_array_equal(actual.data, expected.data)
            assert actual.metadata == expected.metadata
            assert actual["split"] == "train"
            np.testing.assert_array_equal(actual["labels"], np.array([1, 2]))
            assert len(actual.component_signals) == 1
            np.testing.assert_array_equal(
                actual.component_signals[0].data,
                expected.component_signals[0].data,
            )
            assert actual.component_signals[0]["bounds"] == (1, 4)
    finally:
        reader.teardown()


def test_batched_hdf5_preserves_mixed_signal_dtypes(tmp_path) -> None:
    signals = [
        Signal(data=np.ones(4, dtype=np.complex64)),
        Signal(data=np.ones((2, 4), dtype=np.float32)),
    ]
    with BatchedHDF5Writer(tmp_path, max_batches_in_memory=1) as writer:
        writer.write(0, signals)

    reader = BatchedHDF5Reader(tmp_path)
    try:
        complex_signal = reader.read(0)
        real_signal = reader.read(1)
        assert complex_signal.data.dtype == np.complex64
        assert real_signal.data.dtype == np.float32
        assert real_signal.data.shape == (2, 4)
    finally:
        reader.teardown()


def test_batched_hdf5_orders_batches_across_flush_boundaries(tmp_path) -> None:
    batches = {
        idx: [Signal(data=np.array([idx], dtype=np.int64))]
        for idx in range(4)
    }
    with BatchedHDF5Writer(tmp_path, max_batches_in_memory=2) as writer:
        writer.write(2, batches[2])
        writer.write(3, batches[3])
        writer.write(0, batches[0])
        writer.write(1, batches[1])

    reader = BatchedHDF5Reader(tmp_path)
    try:
        assert [int(reader.read(idx).data[0]) for idx in range(4)] == [
            0,
            1,
            2,
            3,
        ]
    finally:
        reader.teardown()


def test_batched_hdf5_rejects_duplicate_batch_index(tmp_path) -> None:
    writer = BatchedHDF5Writer(tmp_path, max_batches_in_memory=2)
    writer.setup()
    writer.write(0, [])
    with pytest.raises(ValueError, match="Duplicate"):
        writer.write(0, [])
    writer.teardown()


def test_batched_hdf5_rejects_missing_batch_at_teardown(tmp_path) -> None:
    writer = BatchedHDF5Writer(tmp_path, max_batches_in_memory=1)
    writer.setup()
    writer.write(1, [])

    with pytest.raises(ValueError, match="missing batch index 0"):
        writer.teardown()
    assert writer._file is None  # noqa: SLF001
    with h5py.File(tmp_path / "data.h5", "r") as handle:
        assert not bool(handle.attrs["complete"])


@pytest.mark.parametrize("batch_idx", [-1, 1.5, True])
def test_batched_hdf5_rejects_invalid_batch_index(tmp_path, batch_idx) -> None:
    with (
        BatchedHDF5Writer(tmp_path) as writer,
        pytest.raises((TypeError, ValueError), match="batch index"),
    ):
        writer.write(batch_idx, [])


def test_batched_hdf5_preserves_reserved_metadata_tags(tmp_path) -> None:
    metadata = {
        "__torchsig_type__": "complex",
        "nested": {
            "__torchsig_type__": "ndarray",
            "data": "ordinary user metadata",
        },
    }
    with BatchedHDF5Writer(tmp_path, max_batches_in_memory=1) as writer:
        writer.write(
            0,
            [Signal(data=np.ones(4, dtype=np.complex64), payload=metadata)],
        )

    reader = BatchedHDF5Reader(tmp_path)
    try:
        assert reader.read(0)["payload"] == metadata
    finally:
        reader.teardown()


@pytest.mark.parametrize(
    "value",
    [
        b"\x00\xff",
        1 + 2j,
        (1, "two"),
        np.int16(3),
        np.array([[1, 2], [3, 4]], dtype=np.int32),
    ],
    ids=["bytes", "complex", "tuple", "numpy-scalar", "numpy-array"],
)
def test_batched_hdf5_round_trips_encoded_metadata_types(
    tmp_path, value
) -> None:
    with BatchedHDF5Writer(tmp_path, max_batches_in_memory=1) as writer:
        writer.write(
            0,
            [Signal(data=np.ones(4, dtype=np.complex64), value=value)],
        )

    reader = BatchedHDF5Reader(tmp_path)
    try:
        actual = reader.read(0)["value"]
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(actual, value)
        else:
            assert actual == value
            assert type(actual) is type(value)
    finally:
        reader.teardown()


def test_batched_hdf5_rejects_non_string_metadata_dictionary_key(
    tmp_path,
) -> None:
    signal = Signal(
        data=np.ones(4, dtype=np.complex64),
        payload={1: "not allowed"},
    )
    writer = BatchedHDF5Writer(tmp_path, max_batches_in_memory=1)
    writer.setup()
    with pytest.raises(TypeError, match="keys must be strings"):
        writer.write(0, [signal])
    writer.teardown()


def test_batched_hdf5_invalid_batch_does_not_append_partial_data(
    tmp_path,
) -> None:
    valid = Signal(data=np.ones(4, dtype=np.complex64), label="valid")
    invalid = Signal(
        data=np.ones(4, dtype=np.float32),
        unsupported=object(),
    )
    writer = BatchedHDF5Writer(tmp_path, max_batches_in_memory=1)
    writer.setup()
    writer.write(0, [valid])
    lengths_before = {
        "records": len(writer._records),  # noqa: SLF001
        "metadata": len(writer._metadata),  # noqa: SLF001
        "shapes": len(writer._shapes),  # noqa: SLF001
        "components": len(writer._components),  # noqa: SLF001
        "index": len(writer._index),  # noqa: SLF001
        "dtypes": len(writer._dtypes),  # noqa: SLF001
        "parents": len(writer._parent_records),  # noqa: SLF001
    }
    data_streams_before = set(writer._data_group)  # noqa: SLF001

    with pytest.raises(TypeError, match="Unsupported packed HDF5 metadata"):
        writer.write(1, [valid, invalid])

    lengths_after = {
        "records": len(writer._records),  # noqa: SLF001
        "metadata": len(writer._metadata),  # noqa: SLF001
        "shapes": len(writer._shapes),  # noqa: SLF001
        "components": len(writer._components),  # noqa: SLF001
        "index": len(writer._index),  # noqa: SLF001
        "dtypes": len(writer._dtypes),  # noqa: SLF001
        "parents": len(writer._parent_records),  # noqa: SLF001
    }
    assert lengths_after == lengths_before
    assert set(writer._data_group) == data_streams_before  # noqa: SLF001
    writer.teardown()


def test_batched_hdf5_rejects_component_cycle_before_appending(
    tmp_path,
) -> None:
    signal = Signal(data=np.ones(4, dtype=np.complex64))
    signal.component_signals.append(signal)
    writer = BatchedHDF5Writer(tmp_path, max_batches_in_memory=1)
    writer.setup()

    with pytest.raises(ValueError, match="component signal cycle"):
        writer.write(0, [signal])

    assert len(writer._records) == 0  # noqa: SLF001
    assert len(writer._index) == 0  # noqa: SLF001
    writer.teardown()


def test_batched_hdf5_marks_successful_file_complete(tmp_path) -> None:
    with BatchedHDF5Writer(tmp_path, max_batches_in_memory=1) as writer:
        writer.write(0, [Signal(data=np.ones(4, dtype=np.complex64))])

    with h5py.File(tmp_path / "data.h5", "r") as handle:
        assert bool(handle.attrs["complete"])


def test_batched_hdf5_context_exception_leaves_file_incomplete(tmp_path) -> None:
    def fail_during_generation() -> None:
        with BatchedHDF5Writer(
            tmp_path, max_batches_in_memory=1
        ) as writer:
            writer.write(
                1, [Signal(data=np.ones(4, dtype=np.complex64))]
            )
            raise RuntimeError("generation failed")

    with pytest.raises(RuntimeError, match="generation failed"):
        fail_during_generation()

    with h5py.File(tmp_path / "data.h5", "r") as handle:
        assert not bool(handle.attrs["complete"])


@pytest.mark.parametrize(
    "transform",
    [ComplexTo2D(), Spectrogram(fft_size=8)],
    ids=["complex-to-2d", "spectrogram"],
)
def test_batched_hdf5_preserves_transformed_2d_shape(tmp_path, transform) -> None:
    source = Signal(
        data=np.exp(2j * np.pi * np.arange(64) / 8).astype(np.complex64)
    )
    expected = transform(source)
    assert expected.data.ndim == 2

    with BatchedHDF5Writer(
        tmp_path,
        shuffle=False,
        fletcher32=False,
        max_batches_in_memory=1,
    ) as writer:
        writer.write(0, [expected])

    reader = BatchedHDF5Reader(tmp_path)
    try:
        actual = reader.read(0)
        assert actual.data.shape == expected.data.shape
        assert actual.data.dtype == expected.data.dtype
        np.testing.assert_array_equal(actual.data, expected.data)
    finally:
        reader.teardown()
