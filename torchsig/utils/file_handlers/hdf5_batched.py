"""Experimental packed HDF5 reader and writer.

Unlike the current one-dataset-per-signal layout, this format stores IQ data,
record descriptors, component links, and metadata in appendable datasets.
It is intentionally separate because its on-disk schema is not backward
compatible with :mod:`torchsig.utils.file_handlers.hdf5`.
"""

from __future__ import annotations

import base64
import json
import threading
from copy import deepcopy
from io import BytesIO
from typing import Any

import h5py
import numpy as np

from torchsig.signals.signal_types import Signal
from torchsig.utils.abstractions import HierarchicalMetadataObject
from torchsig.utils.dsp import torchsig_cache_version
from torchsig.utils.file_handlers.base_handler import FileReader, FileWriter

__all__ = ["BatchedHDF5Reader", "BatchedHDF5Writer"]

_NO_PARENT = np.iinfo(np.int64).max
_RECORD_DTYPE = np.dtype(
    [
        ("data_offset", np.uint64),
        ("data_length", np.uint64),
        ("shape_offset", np.uint64),
        ("shape_count", np.uint16),
        ("component_offset", np.uint64),
        ("component_count", np.uint32),
        ("parent_id", np.uint64),
    ]
)
_PARENT_DTYPE = np.dtype([("parent_id", np.uint64)])


def _pack_value(value: Any) -> Any:  # noqa: PLR0911
    if isinstance(value, np.ndarray):
        buffer = BytesIO()
        np.save(buffer, value, allow_pickle=False)
        return {
            "__torchsig_type__": "ndarray",
            "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
        }
    if isinstance(value, np.generic):
        return _pack_value(np.asarray(value)) | {"scalar": True}
    if isinstance(value, tuple):
        return {
            "__torchsig_type__": "tuple",
            "items": [_pack_value(item) for item in value],
        }
    if isinstance(value, list):
        return [_pack_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _pack_value(item) for key, item in value.items()}
    if isinstance(value, bytes):
        return {
            "__torchsig_type__": "bytes",
            "data": base64.b64encode(value).decode("ascii"),
        }
    if isinstance(value, complex):
        return {
            "__torchsig_type__": "complex",
            "real": value.real,
            "imag": value.imag,
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported packed HDF5 metadata type: {type(value).__name__}")


def _unpack_value(value: Any) -> Any:  # noqa: PLR0911
    if isinstance(value, list):
        return [_unpack_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    value_type = value.get("__torchsig_type__")
    if value_type == "ndarray":
        array = np.load(
            BytesIO(base64.b64decode(value["data"])), allow_pickle=False
        )
        return array[()] if value.get("scalar", False) else array
    if value_type == "tuple":
        return tuple(_unpack_value(item) for item in value["items"])
    if value_type == "bytes":
        return base64.b64decode(value["data"])
    if value_type == "complex":
        return complex(value["real"], value["imag"])
    return {key: _unpack_value(item) for key, item in value.items()}


def _encode_metadata(obj: HierarchicalMetadataObject) -> str:
    # HierarchicalMetadataObject is not iterable; its keys() API is required.
    metadata = {key: _pack_value(obj[key]) for key in obj.keys()}  # noqa: SIM118
    return json.dumps(metadata, separators=(",", ":"), allow_nan=True)


def _decode_metadata(value: str | bytes) -> dict[str, Any]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return _unpack_value(json.loads(value))


def _append(dataset: h5py.Dataset, values: Any) -> int:
    start = len(dataset)
    dataset.resize(start + len(values), axis=0)
    dataset[start:] = values
    return start


class BatchedHDF5Writer(FileWriter):
    """Write signals into a small set of appendable HDF5 datasets."""

    def __init__(
        self,
        root,
        compression: str | None = "lzf",
        compression_opts: int | None = None,
        shuffle: bool = True,
        fletcher32: bool = True,
        chunk_cache_size: int = 10 * 1024 * 1024,
        max_batches_in_memory: int = 4,
    ) -> None:
        super().__init__(root=root)
        self.datapath = self.root.joinpath("data.h5")
        self.compression = compression
        self.compression_opts = compression_opts
        self.shuffle = shuffle
        self.fletcher32 = fletcher32
        self.chunk_cache_size = chunk_cache_size
        self.max_batches_in_memory = max_batches_in_memory
        self._file: h5py.File | None = None
        self._data: h5py.Dataset | None = None
        self._batch_buffer: list[tuple[int, list[Signal]]] = []
        self._parent_ids: dict[tuple[str, int], int] = {}
        self._lock = threading.Lock()

    def _setup(self) -> None:
        self._file = h5py.File(
            self.datapath,
            "w",
            libver="latest",
            rdcc_nbytes=self.chunk_cache_size,
            rdcc_w0=0.75,
        )
        self._file.attrs["torchsig_version"] = torchsig_cache_version()
        self._file.attrs["format"] = "torchsig-packed-v2"
        self._file.attrs["compression"] = self.compression or "none"
        string_dtype = h5py.string_dtype(encoding="utf-8")
        self._records = self._file.create_dataset(
            "records", shape=(0,), maxshape=(None,), dtype=_RECORD_DTYPE, chunks=True
        )
        self._metadata = self._file.create_dataset(
            "metadata", shape=(0,), maxshape=(None,), dtype=string_dtype, chunks=True
        )
        self._components = self._file.create_dataset(
            "components", shape=(0,), maxshape=(None,), dtype=np.uint64, chunks=True
        )
        self._shapes = self._file.create_dataset(
            "shapes", shape=(0,), maxshape=(None,), dtype=np.uint64, chunks=True
        )
        self._index = self._file.create_dataset(
            "index", shape=(0,), maxshape=(None,), dtype=np.uint64, chunks=True
        )
        self._parent_records = self._file.create_dataset(
            "parent_records",
            shape=(0,),
            maxshape=(None,),
            dtype=_PARENT_DTYPE,
            chunks=True,
        )
        self._parent_metadata = self._file.create_dataset(
            "parent_metadata",
            shape=(0,),
            maxshape=(None,),
            dtype=string_dtype,
            chunks=True,
        )
        self._parent_ids.clear()

    def _data_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"chunks": True}
        if self.compression is not None:
            kwargs["compression"] = self.compression
            if self.compression != "lzf" and self.compression_opts is not None:
                kwargs["compression_opts"] = self.compression_opts
        if self.shuffle:
            kwargs["shuffle"] = True
        if self.fletcher32:
            kwargs["fletcher32"] = True
        return kwargs

    def _ensure_data(self, dtype: np.dtype) -> None:
        if self._data is None:
            self._data = self._file.create_dataset(
                "data",
                shape=(0,),
                maxshape=(None,),
                dtype=dtype,
                **self._data_kwargs(),
            )
        elif self._data.dtype != dtype:
            raise TypeError(
                "Packed HDF5 requires one signal dtype per file: "
                f"expected {self._data.dtype}, got {dtype}"
            )

    def _store_parent(self, parent: HierarchicalMetadataObject | None) -> int:
        if parent is None:
            return int(_NO_PARENT)
        parent_parent_id = self._store_parent(parent.parent)
        encoded_metadata = _encode_metadata(parent)
        parent_key = (encoded_metadata, parent_parent_id)
        if parent_key in self._parent_ids:
            return self._parent_ids[parent_key]
        parent_id = len(self._parent_records)
        _append(
            self._parent_records,
            np.array([(parent_parent_id,)], dtype=_PARENT_DTYPE),
        )
        _append(self._parent_metadata, [encoded_metadata])
        self._parent_ids[parent_key] = parent_id
        return parent_id

    def _write_batch(self, signals: list[Signal]) -> None:
        flattened: list[Signal] = []
        component_ids: list[list[int]] = []

        def add_signal(signal: Signal) -> int:
            record_id = len(self._records) + len(flattened)
            flattened.append(signal)
            component_ids.append([])
            component_ids[-1].extend(add_signal(item) for item in signal.component_signals)
            return record_id

        top_ids = [add_signal(signal) for signal in signals]
        if not flattened:
            return

        dtype = np.asarray(flattened[0].data).dtype
        self._ensure_data(dtype)
        arrays = []
        records = np.empty(len(flattened), dtype=_RECORD_DTYPE)
        metadata = []
        links: list[int] = []
        shapes: list[int] = []
        data_offset = len(self._data)
        component_offset = len(self._components)
        shape_offset = len(self._shapes)
        for idx, signal in enumerate(flattened):
            array = np.asarray(signal.data)
            if array.dtype != dtype:
                raise TypeError(
                    f"All signals in a batch must use {dtype}, got {array.dtype}"
                )
            arrays.append(array.reshape(-1))
            children = component_ids[idx]
            records[idx] = (
                data_offset,
                array.size,
                shape_offset + len(shapes),
                array.ndim,
                component_offset + len(links),
                len(children),
                self._store_parent(signal.parent),
            )
            data_offset += array.size
            shapes.extend(array.shape)
            links.extend(children)
            metadata.append(_encode_metadata(signal))

        _append(self._data, np.concatenate(arrays))
        _append(self._records, records)
        _append(self._metadata, metadata)
        if shapes:
            _append(self._shapes, np.asarray(shapes, dtype=np.uint64))
        if links:
            _append(self._components, np.asarray(links, dtype=np.uint64))
        _append(self._index, np.asarray(top_ids, dtype=np.uint64))

    def _flush_buffer(self) -> None:
        if not self._batch_buffer:
            return
        with self._lock:
            self._batch_buffer.sort(key=lambda item: item[0])
            batches = self._batch_buffer
            self._batch_buffer = []
            for _, signals in batches:
                self._write_batch(signals)
            self._file.flush()

    def write(self, batch_idx: int, data: list[Signal]) -> None:
        """Buffer a generated batch for ordered packed writing."""
        with self._lock:
            self._batch_buffer.append((batch_idx, data))
            should_flush = len(self._batch_buffer) >= self.max_batches_in_memory
        if should_flush:
            self._flush_buffer()

    def __len__(self) -> int:
        """Return the number of indexed top-level signals."""
        return len(self._index)

    def teardown(self) -> None:
        """Flush pending batches and close the packed file."""
        if self._file is None:
            return
        self._flush_buffer()
        self._file.close()
        self._file = None
        self._data = None


class BatchedHDF5Reader(FileReader):
    """Read the experimental packed HDF5 schema."""

    def __init__(self, root) -> None:
        super().__init__(root=root)
        self.datapath = self.root.joinpath("data.h5")
        self._file: h5py.File | None = None
        self._len_cache: int | None = None
        self._parent_cache: dict[int, tuple[dict[str, Any], int]] = {}
        self._metadata_cache: dict[int, dict[str, Any]] = {}
        self._locking = False

    def _ensure_open(self) -> None:
        if self._file is None:
            self._file = h5py.File(self.datapath, "r", locking=self._locking)
            self._records = self._file["records"]
            self._metadata = self._file["metadata"]
            self._components = self._file["components"]
            self._shapes = self._file["shapes"]
            self._index = self._file["index"]
            self._data = self._file["data"]
            self._parent_records = self._file["parent_records"]
            self._parent_metadata = self._file["parent_metadata"]

    def __len__(self) -> int:
        """Return the number of indexed top-level signals."""
        self._ensure_open()
        if self._len_cache is None:
            self._len_cache = len(self._index)
        return self._len_cache

    def _build_parent(self, parent_id: int) -> HierarchicalMetadataObject | None:
        if parent_id == _NO_PARENT:
            return None
        try:
            metadata, ancestor_id = self._parent_cache[parent_id]
        except KeyError:
            record = self._parent_records[parent_id]
            ancestor_id = int(record["parent_id"])
            metadata = _decode_metadata(self._parent_metadata[parent_id])
            self._parent_cache[parent_id] = (metadata, ancestor_id)
        parent = HierarchicalMetadataObject(metadata=deepcopy(metadata))
        ancestor = self._build_parent(ancestor_id)
        if ancestor is not None:
            parent.add_parent(ancestor, register=False)
        return parent

    def _read_record(self, record_id: int) -> Signal:
        record = self._records[record_id]
        data_start = int(record["data_offset"])
        data_stop = data_start + int(record["data_length"])
        shape_start = int(record["shape_offset"])
        shape_stop = shape_start + int(record["shape_count"])
        shape = tuple(int(value) for value in self._shapes[shape_start:shape_stop])
        component_start = int(record["component_offset"])
        component_stop = component_start + int(record["component_count"])
        component_ids = self._components[component_start:component_stop]
        try:
            metadata = self._metadata_cache[record_id]
        except KeyError:
            metadata = _decode_metadata(self._metadata[record_id])
            self._metadata_cache[record_id] = metadata
        signal = Signal(
            data=self._data[data_start:data_stop].reshape(shape),
            component_signals=[
                self._read_record(int(component_id)) for component_id in component_ids
            ],
            metadata=deepcopy(metadata),
        )
        parent = self._build_parent(int(record["parent_id"]))
        if parent is not None:
            signal.add_parent(parent, register=False)
        return signal

    def read(self, idx: int) -> Signal:
        """Read a top-level signal by dataset index."""
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Packed HDF5 sample index out of range: {idx}")
        return self._read_record(int(self._index[idx]))

    def teardown(self) -> None:
        """Close the packed file and clear cached parent metadata."""
        if self._file is not None:
            self._file.close()
            self._file = None
        self._len_cache = None
        self._parent_cache.clear()
        self._metadata_cache.clear()
