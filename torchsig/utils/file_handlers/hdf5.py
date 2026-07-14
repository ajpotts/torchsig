"""Sharded HDF5 file handling for TorchSig datasets.

Batches are distributed deterministically across independent shard-writer
processes. Each process owns one HDF5 file, and a manifest exposes the shards
as one ordered logical dataset.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import pickle
import queue
import traceback
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from torchsig.signals.signal_types import Signal
from torchsig.utils.abstractions import HierarchicalMetadataObject
from torchsig.utils.file_handlers.base_handler import BaseFileHandler, FileReader, FileWriter

__all__ = [
    "populate_hdf5_group_with_metadata",
    "populate_hdf5_group_with_signal_data",
    "populate_hdf5_group_with_component_signals",
    "populate_hdf5_group_with_signal",
    "populate_hdf5_group_with_signals",
    "HDF5Writer",
    "handle_bytes_as_string",
    "load_value_from_group",
    "fill_object_metadata_from_group_and_id",
    "load_signal_from_group_by_id",
    "load_signal_from_group_by_index",
    "HDF5Reader",
    "HDF5FileHandler",
]

_STOP = "stop"
_BATCH = "batch"
_MANIFEST_VERSION = 1


@dataclass(frozen=True)
class _ShardConfig:
    shard_id: int
    num_shards: int
    datapath: str
    compression: str | None
    compression_opts: int | None
    shuffle: bool
    fletcher32: bool
    chunk_cache_size: int
    flush_every_n_batches: int


def _torchsig_version() -> str:
    try:
        return version("torchsig")
    except PackageNotFoundError:
        return "unknown"


def _hdf5_key(obj: Any) -> str:
    try:
        return str(obj._hdf5_key)
    except AttributeError as exc:
        raise RuntimeError(f"{type(obj).__name__} has no assigned HDF5 key") from exc


def populate_hdf5_group_with_metadata(group: h5py.Group, metadata_obj: Any) -> bool:
    key = _hdf5_key(metadata_obj)
    if key in group:
        return False

    metadata_group = group.create_group(key)
    for metadata_key in metadata_obj.keys():
        value = metadata_obj[metadata_key]
        if value is not None:
            metadata_group.create_dataset(metadata_key, data=value)

    parent = getattr(metadata_obj, "parent", None)
    if parent is not None:
        metadata_group.create_dataset("parent_metadata_id", data=_hdf5_key(parent))
        populate_hdf5_group_with_metadata(group, parent)
    return True


def populate_hdf5_group_with_signal_data(
    group: h5py.Group,
    signal: Signal,
    dataset_kwargs: dict[str, Any] | None = None,
) -> bool:
    key = _hdf5_key(signal)
    if key in group:
        return False
    group.create_dataset(key, data=signal.data, **(dataset_kwargs or {}))
    return True


def populate_hdf5_group_with_component_signals(group: h5py.Group, signal: Signal) -> bool:
    if not signal.component_signals:
        return False
    group.create_dataset(
        _hdf5_key(signal),
        data=[_hdf5_key(component) for component in signal.component_signals],
    )
    return True


def _populate_hdf5_group_with_signal(
    group: h5py.Group,
    signal: Signal,
    data_dataset_kwargs: dict[str, Any] | None = None,
) -> None:
    populate_hdf5_group_with_metadata(group["metadata"], signal)
    populate_hdf5_group_with_signal_data(
        group["data"], signal, dataset_kwargs=data_dataset_kwargs
    )
    populate_hdf5_group_with_component_signals(group["component_signals"], signal)
    for component in signal.component_signals:
        _populate_hdf5_group_with_signal(group, component, data_dataset_kwargs)


def populate_hdf5_group_with_signal(
    group: h5py.Group,
    signal: Signal,
    index: bool = True,
    data_dataset_kwargs: dict[str, Any] | None = None,
) -> None:
    _populate_hdf5_group_with_signal(group, signal, data_dataset_kwargs)
    if index:
        group["index"].create_dataset(str(len(group["index"])), data=_hdf5_key(signal))


def populate_hdf5_group_with_signals(
    group: h5py.Group,
    signals: Iterable[Signal],
    index: bool = True,
    data_dataset_kwargs: dict[str, Any] | None = None,
) -> None:
    for signal in signals:
        populate_hdf5_group_with_signal(group, signal, index, data_dataset_kwargs)


def _dataset_kwargs(config: _ShardConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if config.compression is not None:
        kwargs["compression"] = config.compression
        if config.compression != "lzf" and config.compression_opts is not None:
            kwargs["compression_opts"] = config.compression_opts
    if config.shuffle:
        kwargs["shuffle"] = True
    if config.fletcher32:
        kwargs["fletcher32"] = True
    if kwargs:
        kwargs["chunks"] = True
    return kwargs


def _assign_keys(signals: Sequence[Signal], next_key: int) -> int:
    assigned: dict[int, str] = {}

    def assign(obj: Any) -> None:
        nonlocal next_key
        obj_id = id(obj)
        if obj_id in assigned:
            obj._hdf5_key = assigned[obj_id]
            return
        key = str(next_key)
        next_key += 1
        assigned[obj_id] = key
        obj._hdf5_key = key
        parent = getattr(obj, "parent", None)
        if parent is not None:
            assign(parent)

    def assign_signal(signal: Signal) -> None:
        assign(signal)
        for component in signal.component_signals:
            assign_signal(component)

    for signal in signals:
        assign_signal(signal)
    return next_key


def _create_shard(config: _ShardConfig) -> h5py.File:
    path = Path(config.datapath)
    path.parent.mkdir(parents=True, exist_ok=True)
    file = h5py.File(
        path,
        "w",
        libver="latest",
        rdcc_nbytes=config.chunk_cache_size,
        rdcc_w0=0.75,
    )
    file.attrs["torchsig_version"] = _torchsig_version()
    file.attrs["compression"] = config.compression or "none"
    file.attrs["created_by"] = "TorchSig Sharded HDF5Writer"
    file.attrs["shard_id"] = config.shard_id
    file.attrs["num_shards"] = config.num_shards
    for name in ("data", "metadata", "index", "component_signals", "batches"):
        file.create_group(name)
    return file


def _shard_process(config: _ShardConfig, work_queue: Any, error_queue: Any) -> None:
    pending: dict[int, Sequence[Signal]] = {}
    next_batch = config.shard_id
    next_key = 0
    local_start = 0
    batches_since_flush = 0

    try:
        with _create_shard(config) as file:
            dataset_kwargs = _dataset_kwargs(config)

            def write_batch(batch_idx: int, batch: Sequence[Signal]) -> None:
                nonlocal next_key, local_start, batches_since_flush
                next_key = _assign_keys(batch, next_key)
                populate_hdf5_group_with_signals(
                    file, batch, data_dataset_kwargs=dataset_kwargs
                )
                batch_group = file["batches"].create_group(str(batch_idx))
                batch_group.create_dataset("local_start", data=local_start)
                batch_group.create_dataset("length", data=len(batch))
                local_start += len(batch)
                batches_since_flush += 1
                if batches_since_flush >= config.flush_every_n_batches:
                    file.flush()
                    batches_since_flush = 0

            while True:
                message = work_queue.get()
                kind = message[0]
                if kind == _BATCH:
                    _, batch_idx, payload = message
                    if batch_idx % config.num_shards != config.shard_id:
                        raise ValueError(f"Batch {batch_idx} routed to wrong shard")
                    if batch_idx in pending or batch_idx < next_batch:
                        raise ValueError(f"Duplicate or late batch {batch_idx}")
                    pending[batch_idx] = pickle.loads(payload)
                    while next_batch in pending:
                        write_batch(next_batch, pending.pop(next_batch))
                        next_batch += config.num_shards
                elif kind == _STOP:
                    for batch_idx in sorted(pending):
                        write_batch(batch_idx, pending[batch_idx])
                    file.flush()
                    break
                else:
                    raise ValueError(f"Unknown message type: {kind!r}")
    except BaseException:
        error_queue.put((config.shard_id, traceback.format_exc()))
        raise


def _build_manifest(manifest_path: Path, shard_paths: Sequence[Path]) -> int:
    batches: list[dict[str, int]] = []
    for shard_id, shard_path in enumerate(shard_paths):
        with h5py.File(shard_path, "r") as file:
            for batch_idx, group in file["batches"].items():
                batches.append(
                    {
                        "batch_idx": int(batch_idx),
                        "shard_id": shard_id,
                        "local_start": int(group["local_start"][()]),
                        "length": int(group["length"][()]),
                    }
                )

    batches.sort(key=lambda item: item["batch_idx"])
    actual = [item["batch_idx"] for item in batches]
    if actual != list(range(len(actual))):
        raise RuntimeError(f"Batch indices are not contiguous: {actual}")

    total_samples = sum(item["length"] for item in batches)
    manifest = {
        "version": _MANIFEST_VERSION,
        "total_samples": total_samples,
        "num_shards": len(shard_paths),
        "shards": [
            {"shard_id": i, "filename": path.name}
            for i, path in enumerate(shard_paths)
        ],
        "batches": batches,
    }
    temp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(temp, manifest_path)
    return total_samples


class HDF5Writer(FileWriter):
    """Write batches concurrently into multiple HDF5 shards."""

    filename = "hdf5_manifest.json"
    manifest_filename = filename

    def __init__(
        self,
        root: str | Path,
        compression: str | None = "lzf",
        compression_opts: int | None = None,
        shuffle: bool = True,
        fletcher32: bool = True,
        chunk_cache_size: int = 10 * 1024 * 1024,
        max_batches_in_memory: int = 4,
        num_shards: int = 4,
        multiprocessing_context: str | None = None,
    ) -> None:
        if max_batches_in_memory < 1:
            raise ValueError("max_batches_in_memory must be at least 1")
        if num_shards < 1:
            raise ValueError("num_shards must be at least 1")

        super().__init__(root=root)

        self.datapath = self.root / self.filename
        self.manifest_path = self.datapath

        self.compression = compression
        self.compression_opts = compression_opts
        self.shuffle = shuffle
        self.fletcher32 = fletcher32
        self.chunk_cache_size = chunk_cache_size
        self.max_batches_in_memory = max_batches_in_memory
        self.num_shards = num_shards
        self.multiprocessing_context = multiprocessing_context

        self._owner_pid = os.getpid()
        self._closed = False

        self._context = None
        self._error_queue = None
        self._queues = []
        self._processes = []

        self._shard_paths = [
            self.root / f"data-{i:05d}.h5"
            for i in range(num_shards)
        ]


    def _config(self, shard_id: int) -> _ShardConfig:
        return _ShardConfig(
            shard_id=shard_id,
            num_shards=self.num_shards,
            datapath=str(self._shard_paths[shard_id]),
            compression=self.compression,
            compression_opts=self.compression_opts,
            shuffle=self.shuffle,
            fletcher32=self.fletcher32,
            chunk_cache_size=self.chunk_cache_size,
            flush_every_n_batches=self.max_batches_in_memory,
        )

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_processes"] = []
        state["_context"] = None
        return state

    def _raise_writer_error(self) -> None:
        """Raise a shard-writer failure in the calling process."""
        if self._error_queue is None:
            return

        errors = []
        while True:
            try:
                errors.append(self._error_queue.get_nowait())
            except queue.Empty:
                break

        if errors:
            details = "\n".join(
                f"Shard {error['shard_id']} failed:\n{error['traceback']}"
                for error in errors
            )
            raise RuntimeError(
                f"HDF5 shard writer process failed:\n{details}"
            )

        for shard_id, process in enumerate(self._processes):
            if process.exitcode not in (None, 0):
                raise RuntimeError(
                    f"HDF5 shard writer {shard_id} exited with "
                    f"code {process.exitcode}"
                )

    def _prepare(self, signal: Signal) -> Signal:
        data = signal.data
        if hasattr(data, "detach"):
            signal.data = data.detach().cpu().numpy().copy()
        elif isinstance(data, np.ndarray):
            signal.data = data.copy()
        for component in signal.component_signals:
            self._prepare(component)
        return signal

    def setup(self) -> None:
        """Prepare the directory and start shard writer processes."""
        super().setup()

        self._context = mp.get_context(self.multiprocessing_context)
        self._error_queue = self._context.Queue()

        self._queues = [
            self._context.Queue(
                maxsize=self.max_batches_in_memory,
            )
            for _ in range(self.num_shards)
        ]

        self._processes = [
            self._context.Process(
                target=_shard_process,
                args=(
                    self._config(shard_id),
                    self._queues[shard_id],
                    self._error_queue,
                ),
                name=f"torchsig-hdf5-shard-{shard_id}",
                daemon=False,
            )
            for shard_id in range(self.num_shards)
        ]

        for process in self._processes:
            process.start()

    def write(self, batch_idx: int, data: Sequence[Signal]) -> None:
        if len(self._queues) != self.num_shards:
            raise RuntimeError(
                "HDF5Writer.setup() has not initialized all shard queues"
            )
        if self._closed:
            raise RuntimeError("Cannot write after teardown")
        if batch_idx < 0:
            raise ValueError("batch_idx must be non-negative")
        self._raise_writer_error()
        prepared = [self._prepare(signal) for signal in data]
        payload = pickle.dumps(prepared, protocol=pickle.HIGHEST_PROTOCOL)
        shard_id = int(batch_idx) % self.num_shards
        self._queues[shard_id].put((_BATCH, int(batch_idx), payload))
        self._raise_writer_error()

    def teardown(self) -> None:
        if self._closed or os.getpid() != self._owner_pid:
            return
        self._closed = True
        self._raise_writer_error()
        for work_queue in self._queues:
            work_queue.put((_STOP,))
        for process in self._processes:
            process.join()
        self._raise_writer_error()
        _build_manifest(self.manifest_path, self._shard_paths)
        for work_queue in self._queues:
            work_queue.close()
            work_queue.join_thread()
        self._error_queue.close()
        self._error_queue.join_thread()

    def __len__(self) -> int:
        if not self._closed:
            raise RuntimeError("Dataset length is available only after teardown")
        return int(json.loads(self.manifest_path.read_text())["total_samples"])


def handle_bytes_as_string(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray) and value.dtype.kind in {"O", "S"}:
        return value.astype(np.str_)
    return value


def load_value_from_group(group: h5py.Group, key: str) -> Any:
    return handle_bytes_as_string(group[key][()])


def fill_object_metadata_from_group_and_id(
    obj: Any, group: h5py.Group, id_str: str
) -> Any:
    metadata_group = group["metadata"][id_str]
    for key in metadata_group.keys():
        if key != "parent_metadata_id":
            obj[key] = load_value_from_group(metadata_group, key)
    if "parent_metadata_id" in metadata_group:
        parent_id = str(load_value_from_group(metadata_group, "parent_metadata_id"))
        parent = fill_object_metadata_from_group_and_id(
            HierarchicalMetadataObject(), group, parent_id
        )
        obj.add_parent(parent)
    return obj


def load_signal_from_group_by_id(group: h5py.Group, id_str: str) -> Signal:
    components: list[Signal] = []
    if id_str in group["component_signals"]:
        component_ids = np.atleast_1d(
            load_value_from_group(group["component_signals"], id_str)
        )
        components = [
            load_signal_from_group_by_id(group, str(component_id))
            for component_id in component_ids
        ]
    signal = Signal(
        data=load_value_from_group(group["data"], id_str),
        component_signals=components,
    )
    return fill_object_metadata_from_group_and_id(signal, group, id_str)


def load_signal_from_group_by_index(group: h5py.Group, index: int) -> Signal:
    id_str = str(load_value_from_group(group["index"], str(index)))
    return load_signal_from_group_by_id(group, id_str)


class HDF5Reader(FileReader):
    """Read a manifest-backed collection of HDF5 shards as one dataset."""

    def __init__(self, root: str | Path) -> None:
        super().__init__(root=root)
        self.datapath = self.root / HDF5Writer.filename
        manifest = json.loads(self.datapath.read_text())
        if int(manifest.get("version", -1)) != _MANIFEST_VERSION:
            raise ValueError(f"Unsupported manifest version: {manifest.get('version')}")
        self._length = int(manifest["total_samples"])
        self._shard_paths = {
            int(item["shard_id"]): self.root / item["filename"]
            for item in manifest["shards"]
        }
        self._locations: list[tuple[int, int]] = []
        for batch in sorted(manifest["batches"], key=lambda item: item["batch_idx"]):
            shard_id = int(batch["shard_id"])
            start = int(batch["local_start"])
            length = int(batch["length"])
            self._locations.extend((shard_id, start + i) for i in range(length))
        if len(self._locations) != self._length:
            raise RuntimeError("Manifest length does not match batch locations")
        self._files: dict[int, h5py.File] = {}
        self._file_pid: int | None = None
        self._locking = False

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_files"] = {}
        state["_file_pid"] = None
        return state

    def __len__(self) -> int:
        return self._length

    def read(self, idx: int) -> Signal:
        if idx < 0 or idx >= self._length:
            raise IndexError(idx)
        self._ensure_process_state()
        shard_id, local_idx = self._locations[idx]
        if shard_id not in self._files:
            self._files[shard_id] = h5py.File(
                self._shard_paths[shard_id], "r", locking=self._locking
            )
        return load_signal_from_group_by_index(self._files[shard_id], local_idx)

    def teardown(self) -> None:
        for file in self._files.values():
            file.close()
        self._files.clear()
        self._file_pid = None

    def _ensure_process_state(self) -> None:
        pid = os.getpid()
        if self._file_pid is not None and self._file_pid != pid:
            self.teardown()
        self._file_pid = pid


class HDF5FileHandler(BaseFileHandler):
    reader_class: type[FileReader] = HDF5Reader
    writer_class: type[FileWriter] = HDF5Writer

    @staticmethod
    def create_handler(
        mode: str, root: str | Path, **kwargs: Any
    ) -> HDF5Writer | HDF5Reader:
        if mode == "r":
            return HDF5FileHandler.reader_class(root, **kwargs)
        if mode == "w":
            return HDF5FileHandler.writer_class(root, **kwargs)
        raise ValueError(f"Invalid file-handler mode: {mode!r}")

