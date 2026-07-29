"""Prototype HDF5 format with homogeneous top-level signal arrays.

Top-level arrays share one fixed dtype and shape and are stored in a native
``(num_signals, *signal_shape)`` dataset. Component signals remain ragged:
each sample indexes a variable-length range of component records whose sample
arrays are stored in flattened dtype streams.

This prototype intentionally rejects parent metadata hierarchies and nested
component signals. It is separate from the versioned packed HDF5 schema and
is intended for layout and performance evaluation.
"""

from __future__ import annotations

from typing import Any

import h5py
import numpy as np

from torchsig.signals.signal_types import Signal
from torchsig.utils.file_handlers.base_handler import FileReader, FileWriter
from torchsig.utils.file_handlers.hdf5_batched import (
    _decode_metadata,
    _encode_metadata,
)

__all__ = ["HomogeneousHDF5Reader", "HomogeneousHDF5Writer"]

_FORMAT = "torchsig-homogeneous-prototype"
_COMPONENT_DTYPE = np.dtype(
    [
        ("data_offset", np.uint64),
        ("data_length", np.uint64),
        ("dtype_id", np.uint32),
        ("shape_offset", np.uint64),
        ("shape_count", np.uint16),
    ]
)


def _append(dataset: h5py.Dataset, values: Any) -> int:
    """Append values to a one-dimensional extensible dataset."""
    start = len(dataset)
    dataset.resize(start + len(values), axis=0)
    dataset[start:] = values
    return start


class HomogeneousHDF5Writer(FileWriter):
    """Write fixed-shape top-level arrays with ragged component signals.

    Every top-level signal must have the same NumPy dtype and shape. Component
    counts, component shapes, and component dtypes may vary by sample.
    """

    def __init__(
        self,
        root,
        compression: str | None = "lzf",
        compression_opts: int | None = None,
        shuffle: bool = True,
        fletcher32: bool = True,
        chunk_samples: int = 32,
    ) -> None:
        super().__init__(root=root)
        if chunk_samples < 1:
            raise ValueError("chunk_samples must be positive")
        self.datapath = self.root / "data.h5"
        self.compression = compression
        self.compression_opts = compression_opts
        self.shuffle = shuffle
        self.fletcher32 = fletcher32
        self.chunk_samples = chunk_samples
        self._file: h5py.File | None = None
        self._data: h5py.Dataset | None = None
        self._shape: tuple[int, ...] | None = None
        self._dtype: np.dtype | None = None
        self._next_batch_idx = 0
        self._component_data: dict[int, h5py.Dataset] = {}
        self._component_dtype_ids: dict[str, int] = {}
        self._failed = False

    def _filter_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.compression is not None:
            kwargs["compression"] = self.compression
            if self.compression != "lzf" and self.compression_opts is not None:
                kwargs["compression_opts"] = self.compression_opts
        if self.shuffle:
            kwargs["shuffle"] = True
        if self.fletcher32:
            kwargs["fletcher32"] = True
        return kwargs

    def _setup(self) -> None:
        self._file = h5py.File(self.datapath, "w", libver="latest")
        self._file.attrs["format"] = _FORMAT
        self._file.attrs["complete"] = False
        self._file.attrs["compression"] = self.compression or "none"
        string_dtype = h5py.string_dtype(encoding="utf-8")
        self._metadata = self._file.create_dataset(
            "metadata",
            shape=(0,),
            maxshape=(None,),
            dtype=string_dtype,
            chunks=True,
        )
        self._component_offsets = self._file.create_dataset(
            "component_offsets",
            data=np.array([0], dtype=np.uint64),
            maxshape=(None,),
            chunks=True,
        )
        self._components = self._file.create_dataset(
            "components",
            shape=(0,),
            maxshape=(None,),
            dtype=_COMPONENT_DTYPE,
            chunks=True,
        )
        self._component_metadata = self._file.create_dataset(
            "component_metadata",
            shape=(0,),
            maxshape=(None,),
            dtype=string_dtype,
            chunks=True,
        )
        self._component_shapes = self._file.create_dataset(
            "component_shapes",
            shape=(0,),
            maxshape=(None,),
            dtype=np.uint64,
            chunks=True,
        )
        self._component_dtypes = self._file.create_dataset(
            "component_dtypes",
            shape=(0,),
            maxshape=(None,),
            dtype=string_dtype,
            chunks=True,
        )
        self._component_data_group = self._file.create_group("component_data")

    def _validate_signal(self, signal: Signal) -> np.ndarray:
        if not isinstance(signal, Signal):
            raise TypeError("Homogeneous HDF5 batches must contain Signal instances")
        if signal.parent is not None:
            raise ValueError("Homogeneous HDF5 prototype does not support parent metadata")
        array = np.asarray(signal.data)
        if array.dtype.hasobject:
            raise TypeError("Homogeneous HDF5 does not support object arrays")
        for component in signal.component_signals:
            if component.parent is not None:
                raise ValueError("Homogeneous HDF5 prototype does not support parent metadata")
            if component.component_signals:
                raise ValueError("Homogeneous HDF5 prototype does not support nested components")
            component_array = np.asarray(component.data)
            if component_array.dtype.hasobject:
                raise TypeError("Homogeneous HDF5 does not support object arrays")
        return array

    def _create_data(self, array: np.ndarray) -> None:
        self._shape = array.shape
        self._dtype = array.dtype
        chunk_shape = (self.chunk_samples, *array.shape)
        self._data = self._file.create_dataset(
            "data",
            shape=(0, *array.shape),
            maxshape=(None, *array.shape),
            dtype=array.dtype,
            chunks=chunk_shape,
            **self._filter_kwargs(),
        )

    def _validate_homogeneity(self, arrays: list[np.ndarray]) -> None:
        for array in arrays:
            if array.shape != self._shape or array.dtype != self._dtype:
                raise ValueError("Homogeneous HDF5 top-level arrays must share one dtype and shape")

    def _component_dtype_id(self, dtype: np.dtype) -> int:
        key = dtype.str
        try:
            return self._component_dtype_ids[key]
        except KeyError:
            dtype_id = len(self._component_dtype_ids)
            self._component_dtype_ids[key] = dtype_id
            _append(self._component_dtypes, [key])
            self._component_data[dtype_id] = self._component_data_group.create_dataset(
                str(dtype_id),
                shape=(0,),
                maxshape=(None,),
                dtype=dtype,
                chunks=True,
                **self._filter_kwargs(),
            )
            return dtype_id

    def write(self, batch_idx: int, data: list[Signal]) -> None:
        """Append one sequentially indexed batch."""
        if self._file is None:
            raise RuntimeError("Homogeneous HDF5 writer is not open")
        if self._failed:
            raise RuntimeError("Homogeneous HDF5 writer cannot continue after failure")
        if batch_idx != self._next_batch_idx:
            raise ValueError(f"Homogeneous HDF5 prototype requires sequential batch indices; expected {self._next_batch_idx}, got {batch_idx}")
        try:
            arrays = [self._validate_signal(signal) for signal in data]
            if arrays and self._data is None:
                self._create_data(arrays[0])
            self._validate_homogeneity(arrays)

            if arrays:
                start = len(self._data)
                self._data.resize(start + len(arrays), axis=0)
                self._data[start:] = np.stack(arrays)
            _append(self._metadata, [_encode_metadata(signal) for signal in data])

            component_total = len(self._components)
            new_offsets = []
            for signal in data:
                for component in signal.component_signals:
                    array = np.asarray(component.data)
                    dtype_id = self._component_dtype_id(array.dtype)
                    data_offset = _append(
                        self._component_data[dtype_id],
                        array.reshape(-1),
                    )
                    shape_offset = _append(
                        self._component_shapes,
                        np.asarray(array.shape, dtype=np.uint64),
                    )
                    _append(
                        self._components,
                        np.array(
                            [
                                (
                                    data_offset,
                                    array.size,
                                    dtype_id,
                                    shape_offset,
                                    array.ndim,
                                )
                            ],
                            dtype=_COMPONENT_DTYPE,
                        ),
                    )
                    _append(self._component_metadata, [_encode_metadata(component)])
                    component_total += 1
                new_offsets.append(component_total)
            _append(
                self._component_offsets,
                np.asarray(new_offsets, dtype=np.uint64),
            )
            self._next_batch_idx += 1
        except Exception:
            self._failed = True
            raise

    def __len__(self) -> int:
        """Return the number of stored top-level signals."""
        return len(self._metadata)

    def teardown(self) -> None:
        """Finalize and close the prototype file."""
        if self._file is None:
            return
        try:
            if not self._failed:
                self._file.attrs["complete"] = True
                self._file.flush()
        finally:
            self._file.close()
            self._file = None

    def __exit__(self, exc_type, exc_value, traceback):
        """Close while leaving failed files incomplete."""
        if exc_type is not None:
            self._failed = True
        self.teardown()
        return False


class HomogeneousHDF5Reader(FileReader):
    """Read homogeneous top-level arrays and ragged component signals."""

    def __init__(self, root) -> None:
        super().__init__(root=root)
        self.datapath = self.root / "data.h5"
        self._file: h5py.File | None = None

    def _ensure_open(self) -> None:
        if self._file is not None:
            return
        self._file = h5py.File(self.datapath, "r", locking=False)
        try:
            self._validate_file()
            self._data = self._file["data"]
            self._metadata = self._file["metadata"]
            self._component_offsets = self._file["component_offsets"]
            self._components = self._file["components"]
            self._component_metadata = self._file["component_metadata"]
            self._component_shapes = self._file["component_shapes"]
            self._component_data = self._file["component_data"]
        except Exception:
            self._file.close()
            self._file = None
            raise

    def _validate_file(self) -> None:
        if self._file.attrs.get("format") != _FORMAT:
            raise ValueError("Not a homogeneous HDF5 prototype file")
        if not bool(self._file.attrs.get("complete", False)):
            raise ValueError("Homogeneous HDF5 prototype file is incomplete")

    def __len__(self) -> int:
        """Return the number of stored top-level signals."""
        self._ensure_open()
        return len(self._metadata)

    def _read_component(self, component_id: int) -> Signal:
        record = self._components[component_id]
        data_offset = int(record["data_offset"])
        data_stop = data_offset + int(record["data_length"])
        shape_offset = int(record["shape_offset"])
        shape_stop = shape_offset + int(record["shape_count"])
        shape = tuple(int(value) for value in self._component_shapes[shape_offset:shape_stop])
        dtype_id = int(record["dtype_id"])
        return Signal(
            data=self._component_data[str(dtype_id)][data_offset:data_stop].reshape(shape),
            metadata=_decode_metadata(self._component_metadata[component_id]),
        )

    def read(self, idx: int) -> Signal:
        """Read one signal and its variable-length component list."""
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Homogeneous HDF5 sample index out of range: {idx}")
        component_start = int(self._component_offsets[idx])
        component_stop = int(self._component_offsets[idx + 1])
        return Signal(
            data=self._data[idx],
            component_signals=[self._read_component(component_id) for component_id in range(component_start, component_stop)],
            metadata=_decode_metadata(self._metadata[idx]),
        )

    def read_batch(self, start: int, stop: int) -> np.ndarray:
        """Read a contiguous batch of top-level arrays without components."""
        if start < 0 or stop < start or stop > len(self):
            raise IndexError(f"Homogeneous HDF5 batch range out of bounds: [{start}, {stop})")
        return self._data[start:stop]

    def teardown(self) -> None:
        """Close the prototype file."""
        if self._file is not None:
            self._file.close()
            self._file = None
