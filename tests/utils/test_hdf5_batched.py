"""Tests for the experimental packed HDF5 format."""

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


def test_batched_hdf5_rejects_mixed_signal_dtypes(tmp_path) -> None:
    signals = [
        Signal(data=np.ones(4, dtype=np.complex64)),
        Signal(data=np.ones(4, dtype=np.complex128)),
    ]
    writer = BatchedHDF5Writer(tmp_path, max_batches_in_memory=1)
    writer.setup()
    try:
        with pytest.raises(TypeError, match="All signals in a batch"):
            writer.write(0, signals)
    finally:
        writer.teardown()


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
