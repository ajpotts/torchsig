"""Tests for the comparison-only direct HDF5 writer."""

import h5py
import numpy as np

from torchsig.signals.signal_types import Signal
from torchsig.utils.abstractions import HierarchicalMetadataObject
from torchsig.utils.file_handlers.hdf5 import HDF5Reader
from torchsig.utils.file_handlers.hdf5_direct_writer import DirectHDF5Writer


def test_direct_hdf5_writer_is_readable_by_current_reader(tmp_path) -> None:
    parent = HierarchicalMetadataObject(
        metadata={"sample_rate": 2_000_000.0, "source": "unit-test"}
    )
    component = Signal(
        data=np.arange(8, dtype=np.complex64),
        parent=parent,
        class_name="component",
    )
    signals = [
        Signal(
            data=(np.arange(32) + idx).astype(np.complex64),
            component_signals=[component.copy()],
            parent=parent,
            class_name="test",
            sample_index=idx,
        )
        for idx in range(3)
    ]

    with DirectHDF5Writer(
        tmp_path,
        shuffle=False,
        fletcher32=False,
        max_batches_in_memory=2,
    ) as writer:
        writer.write(1, signals[2:])
        writer.write(0, signals[:2])

    reader = HDF5Reader(tmp_path)
    try:
        assert len(reader) == len(signals)
        for idx, expected in enumerate(signals):
            actual = reader.read(idx)
            np.testing.assert_array_equal(actual.data, expected.data)
            assert actual.metadata == expected.metadata
            assert actual.get_full_metadata() == expected.get_full_metadata()
            assert len(actual.component_signals) == 1
            np.testing.assert_array_equal(
                actual.component_signals[0].data,
                expected.component_signals[0].data,
            )
    finally:
        reader.teardown()


def test_direct_hdf5_writer_preserves_current_layout(tmp_path) -> None:
    signals = [Signal(data=np.ones(4, dtype=np.complex64), index=idx) for idx in range(2)]
    with DirectHDF5Writer(
        tmp_path,
        shuffle=False,
        fletcher32=False,
        max_batches_in_memory=1,
    ) as writer:
        writer.write(0, signals)

    with h5py.File(tmp_path / "data.h5", "r") as handle:
        assert set(handle) == {"component_signals", "data", "index", "metadata"}
        assert set(handle["data"]) == {"0", "1"}
        assert set(handle["index"]) == {"0", "1"}
        assert set(handle["metadata"]) == {"0", "1"}
