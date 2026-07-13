"""Tests for the multiprocessing-safe TorchSig HDF5 file handler."""

from __future__ import annotations

# Built-In
import multiprocessing as mp
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Third Party
import h5py
import numpy as np
import pytest

# TorchSig
from torchsig.signals.signal_types import Signal
from torchsig.utils.abstractions import HierarchicalMetadataObject
from torchsig.utils.file_handlers.hdf5 import (
    HDF5FileHandler,
    HDF5Reader,
    HDF5Writer,
)


def _make_signal(marker: int, *, with_component: bool = False) -> Signal:
    """Create a small, pickleable signal whose value identifies its batch."""
    component_signals: list[Signal] = []
    if with_component:
        component = Signal(
            data=np.asarray(
                [marker + 0.25j, marker + 0.5j],
                dtype=np.complex64,
            )
        )
        component["component_marker"] = marker
        component_signals.append(component)

    signal = Signal(
        data=np.asarray(
            [marker + 1j, marker + 2j],
            dtype=np.complex64,
        ),
        component_signals=component_signals,
    )
    signal["marker"] = marker
    return signal


def _write_batch_from_worker(
    writer: HDF5Writer,
    batch_idx: int,
    marker: int,
) -> None:
    """Generation-worker entry point used by spawn-based tests."""
    writer.write(batch_idx, [_make_signal(marker)])
    # A worker copy must not terminate the dedicated writer process.
    writer.teardown()


def _read_marker_from_worker(root: str, output_queue: Any) -> None:
    """Read one sample in a separate process and return its marker."""
    reader = HDF5Reader(root)
    try:
        output_queue.put(int(reader.read(0)["marker"]))
    finally:
        reader.teardown()


@pytest.fixture
def writer_factory() -> Callable[..., HDF5Writer]:
    """Create writers and ensure their child processes are cleaned up."""
    writers: list[HDF5Writer] = []

    def factory(root: Path, **kwargs: Any) -> HDF5Writer:
        writer = HDF5Writer(
            root,
            compression=None,
            shuffle=False,
            fletcher32=False,
            multiprocessing_context="spawn",
            **kwargs,
        )
        writers.append(writer)
        return writer

    yield factory

    for writer in writers:
        if not writer._closed:
            writer.teardown()


def test_writer_applies_requested_hdf5_filters(
    tmp_path: Path,
) -> None:
    """Requested compression filters should be visible on stored IQ data."""
    writer = HDF5Writer(
        tmp_path,
        compression="gzip",
        compression_opts=3,
        shuffle=True,
        fletcher32=True,
        multiprocessing_context="spawn",
    )
    writer.write(0, [_make_signal(1)])
    writer.teardown()

    with h5py.File(tmp_path / "data.h5", "r") as hdf5_file:
        signal_id = str(hdf5_file["index"]["0"][()])
        if signal_id.startswith("b'"):
            signal_id = signal_id[2:-1]
        dataset = hdf5_file["data"][signal_id]
        assert dataset.compression == "gzip"
        assert dataset.compression_opts == 3
        assert dataset.shuffle is True
        assert dataset.fletcher32 is True
        assert dataset.chunks is not None


def test_writer_can_disable_optional_hdf5_filters(
    tmp_path: Path,
) -> None:
    """Disabling filters should create an uncompressed IQ dataset."""
    writer = HDF5Writer(
        tmp_path,
        compression=None,
        shuffle=False,
        fletcher32=False,
        multiprocessing_context="spawn",
    )
    writer.write(0, [_make_signal(1)])
    writer.teardown()

    with h5py.File(tmp_path / "data.h5", "r") as hdf5_file:
        signal_id = str(hdf5_file["index"]["0"][()])
        if signal_id.startswith("b'"):
            signal_id = signal_id[2:-1]
        dataset = hdf5_file["data"][signal_id]
        assert dataset.compression is None
        assert dataset.shuffle is False
        assert dataset.fletcher32 is False


def test_writer_rejects_invalid_queue_capacity(tmp_path: Path) -> None:
    """The queue and flush interval must both be positive."""
    with pytest.raises(ValueError, match="at least 1"):
        HDF5Writer(tmp_path, max_batches_in_memory=0)


def test_writer_rejects_negative_batch_index(
    tmp_path: Path,
    writer_factory: Callable[..., HDF5Writer],
) -> None:
    """Negative batch indices cannot participate in ordered serialization."""
    writer = writer_factory(tmp_path)

    with pytest.raises(ValueError, match="non-negative"):
        writer.write(-1, [_make_signal(0)])


def test_writer_requires_teardown_before_len(
    tmp_path: Path,
    writer_factory: Callable[..., HDF5Writer],
) -> None:
    """Length is undefined while batches may still be queued."""
    writer = writer_factory(tmp_path)
    writer.write(0, [_make_signal(10)])

    with pytest.raises(RuntimeError, match="only after teardown"):
        len(writer)

    writer.teardown()
    assert len(writer) == 1


def test_writer_serializes_out_of_order_batches_in_batch_index_order(
    tmp_path: Path,
    writer_factory: Callable[..., HDF5Writer],
) -> None:
    """Arrival order must not affect the final dataset index."""
    writer = writer_factory(tmp_path, max_batches_in_memory=2)

    writer.write(2, [_make_signal(30)])
    writer.write(0, [_make_signal(10)])
    writer.write(1, [_make_signal(20)])
    writer.teardown()

    reader = HDF5Reader(tmp_path)
    try:
        assert len(reader) == 3
        assert [int(reader.read(idx)["marker"]) for idx in range(3)] == [10, 20, 30]
    finally:
        reader.teardown()


def test_generation_processes_can_share_writer(
    tmp_path: Path,
    writer_factory: Callable[..., HDF5Writer],
) -> None:
    """Independent generation processes should safely enqueue one file."""
    writer = writer_factory(tmp_path, max_batches_in_memory=2)
    context = mp.get_context("spawn")

    # Start in deliberately scrambled order. The file must still be indexed by
    # batch_idx rather than process start or completion order.
    jobs = [
        context.Process(target=_write_batch_from_worker, args=(writer, 2, 300)),
        context.Process(target=_write_batch_from_worker, args=(writer, 0, 100)),
        context.Process(target=_write_batch_from_worker, args=(writer, 1, 200)),
    ]

    for job in jobs:
        job.start()
    for job in jobs:
        job.join(timeout=30)
        assert not job.is_alive(), "generation worker did not terminate"
        assert job.exitcode == 0

    writer.teardown()

    reader = HDF5Reader(tmp_path)
    try:
        assert [int(reader.read(idx)["marker"]) for idx in range(3)] == [
            100,
            200,
            300,
        ]
    finally:
        reader.teardown()


def test_round_trip_preserves_components_and_parent_metadata(
    tmp_path: Path,
    writer_factory: Callable[..., HDF5Writer],
) -> None:
    """Nested signals and hierarchical metadata should survive serialization."""
    parent = HierarchicalMetadataObject(metadata={"dataset_name": "unit-test"})
    signal = _make_signal(7, with_component=True)
    signal.add_parent(parent)

    writer = writer_factory(tmp_path)
    writer.write(0, [signal])
    writer.teardown()

    reader = HDF5Reader(tmp_path)
    try:
        restored = reader.read(0)
    finally:
        reader.teardown()

    np.testing.assert_array_equal(restored.data, signal.data)
    assert int(restored["marker"]) == 7
    assert restored.parent["dataset_name"] == "unit-test"
    assert len(restored.component_signals) == 1
    np.testing.assert_array_equal(
        restored.component_signals[0].data,
        signal.component_signals[0].data,
    )
    assert int(restored.component_signals[0]["component_marker"]) == 7


def test_writer_assigns_unique_keys_to_all_serialized_objects(
    tmp_path: Path,
    writer_factory: Callable[..., HDF5Writer],
) -> None:
    """Top-level signals, components, and parents must not collide."""
    parent = HierarchicalMetadataObject(metadata={"source": "test"})
    first = _make_signal(1, with_component=True)
    second = _make_signal(2, with_component=True)
    first.add_parent(parent)
    second.add_parent(parent)

    writer = writer_factory(tmp_path)
    writer.write(0, [first, second])
    writer.teardown()

    with h5py.File(tmp_path / "data.h5", "r") as hdf5_file:
        assert len(hdf5_file["index"]) == 2
        # Two top-level signals plus two components.
        assert len(hdf5_file["data"]) == 4
        # Four signals plus the shared metadata parent.
        assert len(hdf5_file["metadata"]) == 5
        assert len(set(hdf5_file["data"].keys())) == 4


def test_reader_is_pickleable_after_opening_file(
    tmp_path: Path,
    writer_factory: Callable[..., HDF5Writer],
) -> None:
    """An open reader must discard its h5py handle when pickled."""
    writer = writer_factory(tmp_path)
    writer.write(0, [_make_signal(42)])
    writer.teardown()

    reader = HDF5Reader(tmp_path)
    try:
        assert int(reader.read(0)["marker"]) == 42
        restored = pickle.loads(pickle.dumps(reader))
    finally:
        reader.teardown()

    try:
        assert restored._file is None
        assert restored._file_pid is None
        assert int(restored.read(0)["marker"]) == 42
    finally:
        restored.teardown()


def test_reader_can_open_file_in_spawned_process(
    tmp_path: Path,
    writer_factory: Callable[..., HDF5Writer],
) -> None:
    """Each spawned reader process should establish its own HDF5 handle."""
    writer = writer_factory(tmp_path)
    writer.write(0, [_make_signal(88)])
    writer.teardown()

    context = mp.get_context("spawn")
    output_queue = context.Queue()
    process = context.Process(
        target=_read_marker_from_worker,
        args=(str(tmp_path), output_queue),
    )
    process.start()
    process.join(timeout=30)

    assert not process.is_alive(), "reader worker did not terminate"
    assert process.exitcode == 0
    assert output_queue.get(timeout=5) == 88
    output_queue.close()
    output_queue.join_thread()


def test_reader_rejects_out_of_range_indices(
    tmp_path: Path,
    writer_factory: Callable[..., HDF5Writer],
) -> None:
    """Reader bounds checking should happen before HDF5 lookup."""
    writer = writer_factory(tmp_path)
    writer.write(0, [_make_signal(5)])
    writer.teardown()

    reader = HDF5Reader(tmp_path)
    try:
        with pytest.raises(IndexError):
            reader.read(-1)
        with pytest.raises(IndexError):
            reader.read(1)
    finally:
        reader.teardown()


def test_write_after_teardown_is_rejected(
    tmp_path: Path,
    writer_factory: Callable[..., HDF5Writer],
) -> None:
    """Closed writers must not accept batches that can never reach disk."""
    writer = writer_factory(tmp_path)
    writer.write(0, [_make_signal(1)])
    writer.teardown()

    with pytest.raises(RuntimeError, match="after HDF5Writer.teardown"):
        writer.write(1, [_make_signal(2)])


def test_file_handler_factory_selects_reader_and_writer(tmp_path: Path) -> None:
    """The public factory should preserve its existing mode interface."""
    writer = HDF5FileHandler.create_handler(
        "w",
        tmp_path,
        compression=None,
        shuffle=False,
        fletcher32=False,
        multiprocessing_context="spawn",
    )
    assert isinstance(writer, HDF5Writer)
    writer.teardown()

    reader = HDF5FileHandler.create_handler("r", tmp_path)
    assert isinstance(reader, HDF5Reader)
    reader.teardown()

    with pytest.raises(ValueError, match="Invalid file-handler mode"):
        HDF5FileHandler.create_handler("invalid", tmp_path)

