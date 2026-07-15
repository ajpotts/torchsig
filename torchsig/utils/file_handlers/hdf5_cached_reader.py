"""Cached reader for TorchSig's existing HDF5 dataset layout.

This module intentionally leaves :class:`HDF5Reader` unchanged so the two
implementations can be compared against the same files.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import h5py
import numpy as np

from torchsig.signals.signal_types import Signal
from torchsig.utils.abstractions import HierarchicalMetadataObject
from torchsig.utils.file_handlers.base_handler import FileReader
from torchsig.utils.file_handlers.hdf5 import handle_bytes_as_string

__all__ = ["CachedHDF5Reader"]


def _clone_value(value: Any) -> Any:
    """Return a value which callers may mutate without altering the cache."""
    if isinstance(value, np.ndarray):
        return value.copy()
    return deepcopy(value)


class CachedHDF5Reader(FileReader):
    """Read TorchSig HDF5 files while caching their small structural data.

    Signal arrays are still loaded from HDF5 for every call. Index entries,
    metadata records, parent links, and component-signal links are cached in
    memory after their first use. Fresh Signal and metadata objects are built
    for every read so mutations do not leak between returned samples.
    """

    def __init__(self, root) -> None:
        super().__init__(root=root)
        self.datapath = self.root.joinpath("data.h5")
        self._file: h5py.File | None = None
        self._len_cache: int | None = None
        self._index_ids: list[str | None] | None = None
        self._metadata_cache: dict[str, tuple[dict[str, Any], str | None]] = {}
        self._component_id_cache: dict[str, tuple[str, ...]] = {}
        self._locking = False

    @staticmethod
    def _as_id(value: Any) -> str:
        return str(handle_bytes_as_string(value))

    def _ensure_open(self) -> None:
        """Open lazily to remain safe for construction before worker startup."""
        if self._file is None:
            self._file = h5py.File(self.datapath, "r", locking=self._locking)

    def _index_id(self, idx: int) -> str:
        self._ensure_open()
        if self._index_ids is None:
            self._index_ids = [None] * len(self)
        id_str = self._index_ids[idx]
        if id_str is None:
            id_str = self._as_id(self._file["index"][str(idx)][()])
            self._index_ids[idx] = id_str
        return id_str

    def _metadata_record(self, id_str: str) -> tuple[dict[str, Any], str | None]:
        try:
            return self._metadata_cache[id_str]
        except KeyError:
            metadata_group = self._file["metadata"][id_str]
            values = {
                key: handle_bytes_as_string(metadata_group[key][()])
                for key in metadata_group
                if key != "parent_metadata_id"
            }
            parent_id = None
            if "parent_metadata_id" in metadata_group:
                parent_id = self._as_id(metadata_group["parent_metadata_id"][()])
            record = (values, parent_id)
            self._metadata_cache[id_str] = record
            return record

    def _component_ids(self, id_str: str) -> tuple[str, ...]:
        try:
            return self._component_id_cache[id_str]
        except KeyError:
            component_group = self._file["component_signals"]
            ids = (
                ()
                if id_str not in component_group
                else tuple(
                    self._as_id(value) for value in component_group[id_str][()]
                )
            )
            self._component_id_cache[id_str] = ids
            return ids

    def _build_parent(self, id_str: str) -> HierarchicalMetadataObject:
        values, parent_id = self._metadata_record(id_str)
        obj = HierarchicalMetadataObject(
            metadata={key: _clone_value(value) for key, value in values.items()}
        )
        if parent_id is not None:
            obj.add_parent(self._build_parent(parent_id), register=False)
        return obj

    def _read_signal(self, id_str: str) -> Signal:
        values, parent_id = self._metadata_record(id_str)
        components = [
            self._read_signal(component_id)
            for component_id in self._component_ids(id_str)
        ]
        signal = Signal(
            data=self._file["data"][id_str][()],
            component_signals=components,
            metadata={key: _clone_value(value) for key, value in values.items()},
        )
        if parent_id is not None:
            signal.add_parent(self._build_parent(parent_id), register=False)
        return signal

    def __len__(self) -> int:
        """Return the number of indexed signals."""
        self._ensure_open()
        if self._len_cache is None:
            self._len_cache = len(self._file["index"])
        return self._len_cache

    def read(self, idx: int) -> Signal:
        """Read one sample by its zero-based dataset index."""
        if idx < 0 or idx >= len(self):
            raise IndexError(f"HDF5 sample index out of range: {idx}")
        return self._read_signal(self._index_id(idx))

    def teardown(self) -> None:
        """Close the file and discard cached records."""
        if self._file is not None:
            self._file.close()
            self._file = None
        self._len_cache = None
        self._index_ids = None
        self._metadata_cache.clear()
        self._component_id_cache.clear()
