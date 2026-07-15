"""Tests for the comparison-only cached HDF5 reader."""

import numpy as np

from torchsig.signals.signal_types import Signal
from torchsig.utils.abstractions import HierarchicalMetadataObject
from torchsig.utils.file_handlers.hdf5 import HDF5Reader, HDF5Writer
from torchsig.utils.file_handlers.hdf5_cached_reader import CachedHDF5Reader


def test_cached_hdf5_reader_matches_current_reader(tmp_path) -> None:
    parent = HierarchicalMetadataObject(
        metadata={"sample_rate": 2_000_000.0, "source": "unit-test"}
    )
    component = Signal(
        data=np.arange(8, dtype=np.float32).astype(np.complex64),
        parent=parent,
        class_name="component",
    )
    expected = Signal(
        data=(np.arange(32) + 1j * np.arange(32)).astype(np.complex64),
        component_signals=[component],
        parent=parent,
        class_name="test",
        sample_index=7,
    )

    with HDF5Writer(
        tmp_path,
        shuffle=False,
        fletcher32=False,
        max_batches_in_memory=1,
    ) as writer:
        writer.write(0, [expected])

    current = HDF5Reader(tmp_path)
    cached = CachedHDF5Reader(tmp_path)
    try:
        current_signal = current.read(0)
        cached_signal = cached.read(0)

        np.testing.assert_array_equal(cached_signal.data, current_signal.data)
        assert cached_signal.metadata == current_signal.metadata
        assert cached_signal.get_full_metadata() == current_signal.get_full_metadata()
        assert len(cached_signal.component_signals) == 1
        np.testing.assert_array_equal(
            cached_signal.component_signals[0].data,
            current_signal.component_signals[0].data,
        )
        assert (
            cached_signal.component_signals[0].get_full_metadata()
            == current_signal.component_signals[0].get_full_metadata()
        )
    finally:
        current.teardown()
        cached.teardown()


def test_cached_hdf5_reader_returns_independent_objects(tmp_path) -> None:
    signal = Signal(data=np.ones(4, dtype=np.complex64), label="original")
    with HDF5Writer(
        tmp_path,
        shuffle=False,
        fletcher32=False,
        max_batches_in_memory=1,
    ) as writer:
        writer.write(0, [signal])

    reader = CachedHDF5Reader(tmp_path)
    try:
        first = reader.read(0)
        first.data[0] = 0
        first["label"] = "changed"
        second = reader.read(0)

        assert second.data[0] == 1
        assert second["label"] == "original"
    finally:
        reader.teardown()


def test_cached_hdf5_reader_len_does_not_load_index_records(tmp_path) -> None:
    signals = [Signal(data=np.ones(4, dtype=np.complex64)) for _ in range(3)]
    with HDF5Writer(
        tmp_path,
        shuffle=False,
        fletcher32=False,
        max_batches_in_memory=1,
    ) as writer:
        writer.write(0, signals)

    reader = CachedHDF5Reader(tmp_path)
    try:
        assert len(reader) == 3
        assert reader._index_ids is None  # noqa: SLF001

        reader.read(1)
        assert reader._index_ids == [None, "1", None]  # noqa: SLF001
    finally:
        reader.teardown()
