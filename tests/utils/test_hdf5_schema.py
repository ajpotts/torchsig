"""Tests for the self-describing packed HDF5 schema."""

import json

import h5py
import numpy as np
import pytest

from torchsig.signals.signal_types import Signal
from torchsig.utils.file_handlers.hdf5_batched import (
    BatchedHDF5Reader,
    BatchedHDF5Writer,
)
from torchsig.utils.file_handlers.hdf5_schema import (
    default_packed_schema,
    read_schema,
)


def _write_file(root) -> None:
    with BatchedHDF5Writer(root, max_batches_in_memory=1) as writer:
        writer.write(0, [Signal(data=np.ones(4, dtype=np.complex64))])


def _update_schema(root, update) -> None:
    with h5py.File(root / "data.h5", "r+") as handle:
        payload = handle["schema"][()]
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        value = json.loads(payload)
        update(value)
        handle["schema"][()] = json.dumps(value, separators=(",", ":"))


def test_batched_writer_embeds_readable_schema(tmp_path) -> None:
    _write_file(tmp_path)
    with h5py.File(tmp_path / "data.h5", "r") as handle:
        schema = read_schema(handle)

    assert schema == default_packed_schema()
    assert schema.schema_major == 0
    assert schema.schema_minor == 1


def test_batched_reader_rejects_unsupported_schema_major(tmp_path) -> None:
    _write_file(tmp_path)
    _update_schema(tmp_path, lambda value: value.update(schema_major=999))

    reader = BatchedHDF5Reader(tmp_path)
    with pytest.raises(ValueError, match="schema major version"):
        reader.read(0)
    assert reader._file is None  # noqa: SLF001


def test_batched_reader_rejects_unknown_required_feature(tmp_path) -> None:
    _write_file(tmp_path)

    def add_feature(value) -> None:
        value["required_features"].append("future_required_feature")

    _update_schema(tmp_path, add_feature)
    reader = BatchedHDF5Reader(tmp_path)
    with pytest.raises(ValueError, match="Unsupported required"):
        reader.read(0)


def test_batched_reader_rejects_missing_declared_path(tmp_path) -> None:
    _write_file(tmp_path)
    with h5py.File(tmp_path / "data.h5", "r+") as handle:
        del handle["shapes"]

    reader = BatchedHDF5Reader(tmp_path)
    with pytest.raises(ValueError, match="missing declared paths"):
        reader.read(0)
