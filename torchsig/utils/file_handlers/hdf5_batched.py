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
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import h5py
import numpy as np

from torchsig.signals.signal_types import Signal
from torchsig.utils.abstractions import HierarchicalMetadataObject
from torchsig.utils.dsp import torchsig_cache_version
from torchsig.utils.file_handlers.base_handler import FileReader, FileWriter
from torchsig.utils.file_handlers.hdf5_schema import (
    PackedHDF5Schema,
    default_packed_schema,
    read_schema,
    write_schema,
)

__all__ = ["BatchedHDF5Reader", "BatchedHDF5Writer"]

_RECORD_DTYPE = np.dtype(
    [
        ("data_offset", np.uint64),
        ("data_length", np.uint64),
        ("dtype_id", np.uint32),
        ("shape_offset", np.uint64),
        ("shape_count", np.uint16),
        ("component_offset", np.uint64),
        ("component_count", np.uint32),
        ("parent_id", np.uint64),
    ]
)
_PARENT_DTYPE = np.dtype([("parent_id", np.uint64)])


@dataclass
class _PreparedBatch:
    """Validated in-memory representation of one append operation."""

    top_ids: np.ndarray
    records: np.ndarray
    metadata: list[str]
    shapes: np.ndarray
    components: np.ndarray
    arrays_by_dtype: dict[int, np.ndarray]
    new_dtypes: list[tuple[int, np.dtype]]
    new_parents: list[tuple[int, int, str]]
    dtype_ids: dict[str, int]
    parent_ids: dict[tuple[str, int], int]


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
        non_string_keys = [key for key in value if not isinstance(key, str)]
        if non_string_keys:
            raise TypeError(f"Packed HDF5 metadata dictionary keys must be strings; got {type(non_string_keys[0]).__name__}")
        return {
            "__torchsig_type__": "dict",
            "items": {key: _pack_value(item) for key, item in value.items()},
        }
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
        array = np.load(BytesIO(base64.b64decode(value["data"])), allow_pickle=False)
        return array[()] if value.get("scalar", False) else array
    if value_type == "tuple":
        return tuple(_unpack_value(item) for item in value["items"])
    if value_type == "bytes":
        return base64.b64decode(value["data"])
    if value_type == "complex":
        return complex(value["real"], value["imag"])
    if value_type == "dict":
        return {key: _unpack_value(item) for key, item in value["items"].items()}
    return {key: _unpack_value(item) for key, item in value.items()}


def _encode_metadata(obj: HierarchicalMetadataObject) -> str:
    # HierarchicalMetadataObject is not iterable; its keys() API is required.
    metadata = {key: obj[key] for key in obj.keys()}  # noqa: SIM118
    return json.dumps(
        _pack_value(metadata),
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=True,
    )


def _decode_metadata(value: str | bytes) -> dict[str, Any]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return _unpack_value(json.loads(value))


def _append(dataset: h5py.Dataset, values: Any) -> int:
    start = len(dataset)
    dataset.resize(start + len(values), axis=0)
    dataset[start:] = values
    return start


def _validate_declared_datasets(file: h5py.File, schema: PackedHDF5Schema) -> None:
    """Ensure required logical specifications and physical paths exist."""
    required = set(default_packed_schema().datasets)
    missing_specs = required - set(schema.datasets)
    if missing_specs:
        raise ValueError(f"Packed HDF5 schema is missing dataset specifications: {sorted(missing_specs)}")
    missing_paths = [item.path for item in schema.datasets.values() if item.path not in file]
    if missing_paths:
        raise ValueError(f"Packed HDF5 file is missing declared paths: {missing_paths}")


def _validate_complete_file(file: h5py.File) -> None:
    """Reject files which were not finalized by a successful writer."""
    if "complete" not in file.attrs:
        raise ValueError("Invalid packed HDF5 file: missing completeness marker")
    if not bool(file.attrs["complete"]):
        raise ValueError("Packed HDF5 file is incomplete")


def _validate_acyclic_links(links: list[list[int]], *, relationship: str) -> None:
    """Reject cycles in a table of record relationships."""
    unvisited = 0
    visiting = 1
    visited = 2
    states = np.zeros(len(links), dtype=np.uint8)

    for record_id in range(len(links)):
        if states[record_id] != unvisited:
            continue
        stack = [(record_id, False)]
        while stack:
            linked_id, exiting = stack.pop()
            if exiting:
                states[linked_id] = visited
                continue
            if states[linked_id] == visiting:
                raise ValueError(f"Invalid packed HDF5 file: {relationship} cycle at record {linked_id}")
            if states[linked_id] == visited:
                continue
            states[linked_id] = visiting
            stack.append((linked_id, True))
            stack.extend((child_id, False) for child_id in reversed(links[linked_id]))


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
        self._data: dict[int, h5py.Dataset] = {}
        self._dtype_ids: dict[str, int] = {}
        self._batch_buffer: dict[int, list[Signal]] = {}
        self._next_batch_idx = 0
        self._parent_ids: dict[tuple[str, int], int] = {}
        self._lock = threading.Lock()
        self._write_failed = False
        self.schema = default_packed_schema()

    def _setup(self) -> None:
        self._file = h5py.File(
            self.datapath,
            "w",
            libver="latest",
            rdcc_nbytes=self.chunk_cache_size,
            rdcc_w0=0.75,
        )
        self._file.attrs["torchsig_version"] = torchsig_cache_version()
        self._file.attrs["format"] = self.schema.format
        self._file.attrs["compression"] = self.compression or "none"
        self._file.attrs["complete"] = False
        write_schema(self._file, self.schema)
        string_dtype = h5py.string_dtype(encoding="utf-8")
        spec = self.schema.datasets
        self._data_group = self._file.create_group(spec["data"].path)
        self._dtypes = self._file.create_dataset(
            spec["dtypes"].path,
            shape=(0,),
            maxshape=(None,),
            dtype=string_dtype,
            chunks=True,
        )
        self._records = self._file.create_dataset(
            spec["records"].path,
            shape=(0,),
            maxshape=(None,),
            dtype=_RECORD_DTYPE,
            chunks=True,
        )
        self._metadata = self._file.create_dataset(
            spec["metadata"].path,
            shape=(0,),
            maxshape=(None,),
            dtype=string_dtype,
            chunks=True,
        )
        self._components = self._file.create_dataset(
            spec["components"].path,
            shape=(0,),
            maxshape=(None,),
            dtype=np.uint64,
            chunks=True,
        )
        self._shapes = self._file.create_dataset(
            spec["shapes"].path,
            shape=(0,),
            maxshape=(None,),
            dtype=np.uint64,
            chunks=True,
        )
        self._index = self._file.create_dataset(
            spec["index"].path,
            shape=(0,),
            maxshape=(None,),
            dtype=np.uint64,
            chunks=True,
        )
        self._parent_records = self._file.create_dataset(
            spec["parent_records"].path,
            shape=(0,),
            maxshape=(None,),
            dtype=_PARENT_DTYPE,
            chunks=True,
        )
        self._parent_metadata = self._file.create_dataset(
            spec["parent_metadata"].path,
            shape=(0,),
            maxshape=(None,),
            dtype=string_dtype,
            chunks=True,
        )
        self._parent_ids.clear()
        self._dtype_ids.clear()
        self._data.clear()
        self._batch_buffer.clear()
        self._next_batch_idx = 0
        self._write_failed = False

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

    def _prepare_batch(self, signals: list[Signal]) -> _PreparedBatch | None:
        """Validate and encode a batch without modifying the HDF5 file."""
        flattened: list[Signal] = []
        component_ids: list[list[int]] = []
        active_components: set[int] = set()

        def add_signal(signal: Signal) -> int:
            if not isinstance(signal, Signal):
                raise TypeError("Packed HDF5 batches must contain Signal instances")
            signal_identity = id(signal)
            if signal_identity in active_components:
                raise ValueError("Packed HDF5 component signal cycle detected")
            active_components.add(signal_identity)
            record_id = len(self._records) + len(flattened)
            flattened.append(signal)
            component_ids.append([])
            children = component_ids[-1]
            children.extend(add_signal(item) for item in signal.component_signals)
            active_components.remove(signal_identity)
            return record_id

        top_ids = [add_signal(signal) for signal in signals]
        if not flattened:
            return None

        dtype_ids = self._dtype_ids.copy()
        parent_ids = self._parent_ids.copy()
        new_dtypes: list[tuple[int, np.dtype]] = []
        new_parents: list[tuple[int, int, str]] = []
        active_parents: set[int] = set()

        def prepare_dtype(dtype: np.dtype) -> int:
            if dtype.hasobject:
                raise TypeError("Packed HDF5 does not support object signal dtypes")
            dtype_string = dtype.str
            if dtype_string not in dtype_ids:
                dtype_id = len(dtype_ids)
                if dtype_id > np.iinfo(np.uint32).max:
                    raise OverflowError("Packed HDF5 has too many signal dtypes")
                dtype_ids[dtype_string] = dtype_id
                new_dtypes.append((dtype_id, dtype))
            return dtype_ids[dtype_string]

        def prepare_parent(
            parent: HierarchicalMetadataObject | None,
        ) -> int:
            if parent is None:
                return self.schema.sentinels["no_parent"]
            parent_identity = id(parent)
            if parent_identity in active_parents:
                raise ValueError("Packed HDF5 parent metadata cycle detected")
            active_parents.add(parent_identity)
            ancestor_id = prepare_parent(parent.parent)
            encoded_metadata = _encode_metadata(parent)
            parent_key = (encoded_metadata, ancestor_id)
            if parent_key not in parent_ids:
                parent_id = len(parent_ids)
                parent_ids[parent_key] = parent_id
                new_parents.append((parent_id, ancestor_id, encoded_metadata))
            active_parents.remove(parent_identity)
            return parent_ids[parent_key]

        arrays_by_dtype: dict[int, list[np.ndarray]] = {}
        offsets_by_dtype: dict[int, int] = {}
        records = np.empty(len(flattened), dtype=_RECORD_DTYPE)
        metadata: list[str] = []
        links: list[int] = []
        shapes: list[int] = []
        component_offset = len(self._components)
        shape_offset = len(self._shapes)
        for idx, signal in enumerate(flattened):
            array = np.asarray(signal.data)
            if array.ndim > np.iinfo(np.uint16).max:
                raise OverflowError("Packed HDF5 signal has too many dimensions")
            children = component_ids[idx]
            if len(children) > np.iinfo(np.uint32).max:
                raise OverflowError("Packed HDF5 signal has too many component signals")
            encoded_metadata = _encode_metadata(signal)
            parent_id = prepare_parent(signal.parent)
            dtype_id = prepare_dtype(array.dtype)
            if dtype_id not in offsets_by_dtype:
                offsets_by_dtype[dtype_id] = len(self._data[dtype_id]) if dtype_id in self._data else 0
                arrays_by_dtype[dtype_id] = []
            data_offset = offsets_by_dtype[dtype_id]
            arrays_by_dtype[dtype_id].append(array.reshape(-1))
            records[idx] = (
                data_offset,
                array.size,
                dtype_id,
                shape_offset + len(shapes),
                array.ndim,
                component_offset + len(links),
                len(children),
                parent_id,
            )
            offsets_by_dtype[dtype_id] += array.size
            shapes.extend(array.shape)
            links.extend(children)
            metadata.append(encoded_metadata)

        return _PreparedBatch(
            top_ids=np.asarray(top_ids, dtype=np.uint64),
            records=records,
            metadata=metadata,
            shapes=np.asarray(shapes, dtype=np.uint64),
            components=np.asarray(links, dtype=np.uint64),
            arrays_by_dtype={dtype_id: np.concatenate(arrays) for dtype_id, arrays in arrays_by_dtype.items()},
            new_dtypes=new_dtypes,
            new_parents=new_parents,
            dtype_ids=dtype_ids,
            parent_ids=parent_ids,
        )

    def _commit_batch(self, batch: _PreparedBatch) -> None:
        """Append an already validated batch to the open HDF5 file."""
        for dtype_id, dtype in batch.new_dtypes:
            _append(self._dtypes, [dtype.str])
            self._data[dtype_id] = self._data_group.create_dataset(
                str(dtype_id),
                shape=(0,),
                maxshape=(None,),
                dtype=dtype,
                **self._data_kwargs(),
            )
        self._dtype_ids = batch.dtype_ids

        for _, ancestor_id, encoded_metadata in batch.new_parents:
            _append(
                self._parent_records,
                np.array([(ancestor_id,)], dtype=_PARENT_DTYPE),
            )
            _append(self._parent_metadata, [encoded_metadata])
        self._parent_ids = batch.parent_ids

        for dtype_id, array in batch.arrays_by_dtype.items():
            _append(self._data[dtype_id], array)
        _append(self._records, batch.records)
        _append(self._metadata, batch.metadata)
        if len(batch.shapes):
            _append(self._shapes, batch.shapes)
        if len(batch.components):
            _append(self._components, batch.components)
        _append(self._index, batch.top_ids)

    def _write_batch(self, signals: list[Signal]) -> None:
        batch = self._prepare_batch(signals)
        if batch is not None:
            self._commit_batch(batch)

    def _flush_buffer(self, *, final: bool = False) -> None:
        with self._lock:
            while self._next_batch_idx in self._batch_buffer:
                signals = self._batch_buffer.pop(self._next_batch_idx)
                self._write_batch(signals)
                self._next_batch_idx += 1
            if final and self._batch_buffer:
                pending = sorted(self._batch_buffer)
                raise ValueError(f"Cannot finalize packed HDF5 file: missing batch index {self._next_batch_idx}; pending batch indices: {pending}")
            if self._file is not None:
                self._file.flush()

    def write(self, batch_idx: int, data: list[Signal]) -> None:
        """Buffer a uniquely indexed batch and write each contiguous prefix.

        Batch indices must be non-negative and form a contiguous sequence
        beginning at zero. Batches may arrive out of order, but a batch is not
        committed until every preceding batch has arrived.
        """
        if not isinstance(batch_idx, int) or isinstance(batch_idx, bool):
            raise TypeError("Packed HDF5 batch index must be an integer")
        if batch_idx < 0:
            raise ValueError("Packed HDF5 batch index must be non-negative")
        with self._lock:
            if batch_idx < self._next_batch_idx or batch_idx in self._batch_buffer:
                raise ValueError(f"Duplicate packed HDF5 batch index: {batch_idx}")
            self._batch_buffer[batch_idx] = data
            should_flush = len(self._batch_buffer) >= self.max_batches_in_memory
        if should_flush:
            try:
                self._flush_buffer()
            except Exception:
                self._write_failed = True
                raise

    def __len__(self) -> int:
        """Return the number of indexed top-level signals."""
        return len(self._index)

    def teardown(self) -> None:
        """Flush pending batches and close the packed file."""
        if self._file is None:
            return
        try:
            self._flush_buffer(final=True)
            if not self._write_failed:
                self._file.attrs["complete"] = True
                self._file.flush()
        except Exception:
            self._write_failed = True
            raise
        finally:
            self._file.close()
            self._file = None
            self._data.clear()

    def __exit__(self, exc_type, exc_value, traceback):
        """Close the file while preserving an incomplete marker on failure."""
        if exc_type is not None:
            self._write_failed = True
            if self._file is not None:
                self._file.close()
                self._file = None
                self._data.clear()
            return False
        self.teardown()
        return False


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
        self.schema: PackedHDF5Schema | None = None

    def _ensure_open(self) -> None:
        if self._file is None:
            self._file = h5py.File(self.datapath, "r", locking=self._locking)
            try:
                self.schema = read_schema(self._file)
                _validate_complete_file(self._file)
                spec = self.schema.datasets
                _validate_declared_datasets(self._file, self.schema)
                self._records = self._file[spec["records"].path]
                self._metadata = self._file[spec["metadata"].path]
                self._components = self._file[spec["components"].path]
                self._shapes = self._file[spec["shapes"].path]
                self._index = self._file[spec["index"].path]
                self._data = self._file[spec["data"].path]
                self._data_streams = {int(dtype_id): self._data[dtype_id] for dtype_id in self._data}
                self._dtypes = self._file[spec["dtypes"].path]
                self._parent_records = self._file[spec["parent_records"].path]
                self._parent_metadata = self._file[spec["parent_metadata"].path]
                self._record_fields = spec["records"].fields
                self._parent_fields = spec["parent_records"].fields
                self._no_parent = self.schema.sentinels["no_parent"]
                self._validate_integrity()
            except Exception:
                self._file.close()
                self._file = None
                raise

    def _validate_integrity(self) -> None:
        """Validate cross-dataset references before serving any records."""
        if len(self._records) != len(self._metadata):
            raise ValueError("Invalid packed HDF5 file: records and metadata lengths differ")
        if len(self._parent_records) != len(self._parent_metadata):
            raise ValueError("Invalid packed HDF5 file: parent record and metadata lengths differ")

        required_record_fields = set(_RECORD_DTYPE.names or ())
        if self._record_fields is None or required_record_fields - set(self._record_fields):
            raise ValueError("Invalid packed HDF5 schema: missing record field mappings")
        required_parent_fields = set(_PARENT_DTYPE.names or ())
        if self._parent_fields is None or required_parent_fields - set(self._parent_fields):
            raise ValueError("Invalid packed HDF5 schema: missing parent field mappings")
        missing_record_fields = set(self._record_fields.values()) - set(self._records.dtype.names or ())
        if missing_record_fields:
            raise ValueError(f"Invalid packed HDF5 file: record dataset is missing fields {sorted(missing_record_fields)}")
        missing_parent_fields = set(self._parent_fields.values()) - set(self._parent_records.dtype.names or ())
        if missing_parent_fields:
            raise ValueError(f"Invalid packed HDF5 file: parent record dataset is missing fields {sorted(missing_parent_fields)}")

        dtype_count = len(self._dtypes)
        expected_dtype_ids = set(range(dtype_count))
        if set(self._data_streams) != expected_dtype_ids:
            raise ValueError("Invalid packed HDF5 file: data stream IDs do not match the dtype table")
        for dtype_id, stream in self._data_streams.items():
            dtype_value = self._dtypes[dtype_id]
            if isinstance(dtype_value, bytes):
                dtype_value = dtype_value.decode("utf-8")
            try:
                declared_dtype = np.dtype(dtype_value)
            except TypeError as error:
                raise ValueError(f"Invalid packed HDF5 file: invalid dtype descriptor at index {dtype_id}") from error
            if stream.dtype != declared_dtype:
                raise ValueError(f"Invalid packed HDF5 file: data stream dtype mismatch at index {dtype_id}")

        record_count = len(self._records)
        component_links: list[list[int]] = []
        fields = self._record_fields
        for record_id in range(record_count):
            record = self._records[record_id]
            dtype_id = int(record[fields["dtype_id"]])
            if dtype_id not in self._data_streams:
                raise ValueError(f"Invalid packed HDF5 file: invalid dtype ID {dtype_id} in record {record_id}")
            data_start = int(record[fields["data_offset"]])
            data_length = int(record[fields["data_length"]])
            if data_start + data_length > len(self._data_streams[dtype_id]):
                raise ValueError(f"Invalid packed HDF5 file: data slice out of bounds in record {record_id}")
            shape_start = int(record[fields["shape_offset"]])
            shape_count = int(record[fields["shape_count"]])
            if shape_start + shape_count > len(self._shapes):
                raise ValueError(f"Invalid packed HDF5 file: shape slice out of bounds in record {record_id}")
            shape = tuple(int(value) for value in self._shapes[shape_start : shape_start + shape_count])
            if int(np.prod(shape, dtype=np.uint64)) != data_length:
                raise ValueError(f"Invalid packed HDF5 file: shape does not match data length in record {record_id}")
            component_start = int(record[fields["component_offset"]])
            component_count = int(record[fields["component_count"]])
            if component_start + component_count > len(self._components):
                raise ValueError(f"Invalid packed HDF5 file: component slice out of bounds in record {record_id}")
            children = [int(value) for value in self._components[component_start : component_start + component_count]]
            if any(child_id >= record_count for child_id in children):
                raise ValueError(f"Invalid packed HDF5 file: invalid component record ID in record {record_id}")
            component_links.append(children)
            parent_id = int(record[fields["parent_id"]])
            if parent_id != self._no_parent and parent_id >= len(self._parent_records):
                raise ValueError(f"Invalid packed HDF5 file: invalid parent ID {parent_id} in record {record_id}")

        top_ids = [int(value) for value in self._index]
        if any(record_id >= record_count for record_id in top_ids):
            raise ValueError("Invalid packed HDF5 file: invalid top-level record ID")
        _validate_acyclic_links(component_links, relationship="component relationship")

        parent_links: list[list[int]] = []
        parent_field = self._parent_fields["parent_id"]
        parent_count = len(self._parent_records)
        for parent_id in range(parent_count):
            ancestor_id = int(self._parent_records[parent_id][parent_field])
            if ancestor_id == self._no_parent:
                parent_links.append([])
            elif ancestor_id >= parent_count:
                raise ValueError(f"Invalid packed HDF5 file: invalid ancestor ID {ancestor_id} in parent record {parent_id}")
            else:
                parent_links.append([ancestor_id])
        _validate_acyclic_links(parent_links, relationship="parent relationship")

    def __len__(self) -> int:
        """Return the number of indexed top-level signals."""
        self._ensure_open()
        if self._len_cache is None:
            self._len_cache = len(self._index)
        return self._len_cache

    def _build_parent(self, parent_id: int) -> HierarchicalMetadataObject | None:
        if parent_id == self._no_parent:
            return None
        try:
            metadata, ancestor_id = self._parent_cache[parent_id]
        except KeyError:
            record = self._parent_records[parent_id]
            ancestor_id = int(record[self._parent_fields["parent_id"]])
            metadata = _decode_metadata(self._parent_metadata[parent_id])
            self._parent_cache[parent_id] = (metadata, ancestor_id)
        parent = HierarchicalMetadataObject(metadata=deepcopy(metadata))
        ancestor = self._build_parent(ancestor_id)
        if ancestor is not None:
            parent.add_parent(ancestor, register=False)
        return parent

    def _read_record(self, record_id: int) -> Signal:
        record = self._records[record_id]
        fields = self._record_fields
        data_start = int(record[fields["data_offset"]])
        data_stop = data_start + int(record[fields["data_length"]])
        dtype_id = int(record[fields["dtype_id"]])
        shape_start = int(record[fields["shape_offset"]])
        shape_stop = shape_start + int(record[fields["shape_count"]])
        shape = tuple(int(value) for value in self._shapes[shape_start:shape_stop])
        component_start = int(record[fields["component_offset"]])
        component_stop = component_start + int(record[fields["component_count"]])
        component_ids = self._components[component_start:component_stop]
        try:
            metadata = self._metadata_cache[record_id]
        except KeyError:
            metadata = _decode_metadata(self._metadata[record_id])
            self._metadata_cache[record_id] = metadata
        signal = Signal(
            data=self._data_streams[dtype_id][data_start:data_stop].reshape(shape),
            component_signals=[self._read_record(int(component_id)) for component_id in component_ids],
            metadata=deepcopy(metadata),
        )
        parent = self._build_parent(int(record[fields["parent_id"]]))
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
        self.schema = None
        self._parent_cache.clear()
        self._metadata_cache.clear()
