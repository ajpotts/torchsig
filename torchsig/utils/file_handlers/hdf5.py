"""HDF5 file handling for TorchSig datasets.

The writer supports multiprocessing dataset generation by routing generated
batches through a process-safe queue to one dedicated HDF5 writer process.
HDF5 objects are never shared between processes.
"""

from __future__ import annotations

# Built-In
import multiprocessing as mp
import os
import queue
import traceback
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from importlib.metadata import PackageNotFoundError, version

# Third Party
import h5py
import pickle
import numpy as np

# TorchSig
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


def _torchsig_version() -> str:
    try:
        return version("torchsig")
    except PackageNotFoundError:
        return "unknown"

@dataclass(frozen=True)
class _WriterConfig:
    """Serializable configuration consumed by the writer process."""

    datapath: str
    compression: str | None
    compression_opts: int | None
    shuffle: bool
    fletcher32: bool
    chunk_cache_size: int
    flush_every_n_batches: int


def _hdf5_key(obj: Any) -> str:
    """Return the HDF5 key assigned to an object before it is written."""
    try:
        return str(obj._hdf5_key)
    except AttributeError as exc:
        raise RuntimeError(
            f"{type(obj).__name__} has no assigned HDF5 key. "
            "Objects must be keyed by the HDF5 writer before serialization."
        ) from exc


def _metadata_value_is_none(value: Any) -> bool:
    """Return whether a metadata value is scalar ``None``."""
    return value is None


def populate_hdf5_group_with_metadata(group: h5py.Group, metadata_obj: Any) -> bool:
    """Write a metadata object and its parent chain to an HDF5 group.

    Args:
        group: HDF5 metadata group.
        metadata_obj: Metadata-bearing object to serialize.

    Returns:
        True when a new metadata group was created; otherwise False.
    """
    key = _hdf5_key(metadata_obj)
    if key in group:
        return False

    metadata_group = group.create_group(key)
    for metadata_key in metadata_obj.keys():
        value = metadata_obj[metadata_key]
        if not _metadata_value_is_none(value):
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
    """Write one signal's sample array to an HDF5 group."""
    key = _hdf5_key(signal)
    if key in group:
        return False

    group.create_dataset(key, data=signal.data, **(dataset_kwargs or {}))
    return True


def populate_hdf5_group_with_component_signals(
    group: h5py.Group,
    signal: Signal,
) -> bool:
    """Write the component-signal references for one signal."""
    component_signals = signal.component_signals
    if not component_signals:
        return False

    group.create_dataset(
        _hdf5_key(signal),
        data=[_hdf5_key(component) for component in component_signals],
    )
    return True


def _populate_hdf5_group_with_signal(
    group: h5py.Group,
    signal: Signal,
    data_dataset_kwargs: dict[str, Any] | None = None,
) -> None:
    """Write a signal, its metadata, and all recursively nested components."""
    populate_hdf5_group_with_metadata(group["metadata"], signal)
    populate_hdf5_group_with_signal_data(
        group["data"],
        signal,
        dataset_kwargs=data_dataset_kwargs,
    )
    populate_hdf5_group_with_component_signals(group["component_signals"], signal)

    for component_signal in signal.component_signals:
        _populate_hdf5_group_with_signal(
            group,
            component_signal,
            data_dataset_kwargs=data_dataset_kwargs,
        )


def populate_hdf5_group_with_signal(
    group: h5py.Group,
    signal: Signal,
    index: bool = True,
    data_dataset_kwargs: dict[str, Any] | None = None,
) -> None:
    """Write one signal and optionally add it to the top-level sample index."""
    _populate_hdf5_group_with_signal(
        group,
        signal,
        data_dataset_kwargs=data_dataset_kwargs,
    )
    if index:
        group["index"].create_dataset(str(len(group["index"])), data=_hdf5_key(signal))


def populate_hdf5_group_with_signals(
    group: h5py.Group,
    signals: Iterable[Signal],
    index: bool = True,
    data_dataset_kwargs: dict[str, Any] | None = None,
) -> None:
    """Write multiple signals and optionally add each to the sample index."""
    for signal in signals:
        populate_hdf5_group_with_signal(
            group,
            signal,
            index=index,
            data_dataset_kwargs=data_dataset_kwargs,
        )


def _data_dataset_kwargs(config: _WriterConfig) -> dict[str, Any]:
    """Build the HDF5 dataset options used for IQ arrays."""
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


def _assign_hdf5_keys(
    signals: Sequence[Signal],
    next_key: int,
) -> int:
    """Assign collision-free keys to signals and metadata parents.

    Generated objects may originate in different worker processes, where
    ``id(obj)`` values are neither globally unique nor stable after pickling.
    Keys are therefore assigned only inside the single writer process.
    """
    assigned: dict[int, str] = {}

    def assign_metadata_chain(obj: Any) -> None:
        nonlocal next_key
        object_id = id(obj)
        if object_id in assigned:
            obj._hdf5_key = assigned[object_id]
            return

        key = str(next_key)
        next_key += 1
        assigned[object_id] = key
        obj._hdf5_key = key

        parent = getattr(obj, "parent", None)
        if parent is not None:
            assign_metadata_chain(parent)

    def assign_signal(signal: Signal) -> None:
        assign_metadata_chain(signal)
        for component in signal.component_signals:
            assign_signal(component)

    for signal in signals:
        assign_signal(signal)

    return next_key


def _create_hdf5_file(config: _WriterConfig) -> h5py.File:
    """Create the output file and its fixed group structure."""
    datapath = Path(config.datapath)
    datapath.parent.mkdir(parents=True, exist_ok=True)

    hdf5_file = h5py.File(
        datapath,
        "w",
        libver="latest",
        rdcc_nbytes=config.chunk_cache_size,
        rdcc_w0=0.75,
    )
    hdf5_file.attrs["torchsig_version"] = _torchsig_version()
    hdf5_file.attrs["compression"] = config.compression or "none"
    hdf5_file.attrs["created_by"] = "TorchSig HDF5FileHandler"
    hdf5_file.create_group("data")
    hdf5_file.create_group("metadata")
    hdf5_file.create_group("index")
    hdf5_file.create_group("component_signals")
    return hdf5_file


def _writer_process_main(
    config: _WriterConfig,
    work_queue: Any,
    error_queue: Any,
) -> None:
    """Consume generated batches and perform all HDF5 writes in one process."""
    pending: dict[int, Sequence[Signal]] = {}
    next_batch_idx = 0
    next_key = 0
    batches_since_flush = 0

    try:
        with _create_hdf5_file(config) as hdf5_file:
            dataset_kwargs = _data_dataset_kwargs(config)

            def write_batch(batch: Sequence[Signal]) -> None:
                nonlocal next_key, batches_since_flush
                next_key = _assign_hdf5_keys(batch, next_key)
                populate_hdf5_group_with_signals(
                    hdf5_file,
                    batch,
                    data_dataset_kwargs=dataset_kwargs,
                )
                batches_since_flush += 1
                if batches_since_flush >= config.flush_every_n_batches:
                    hdf5_file.flush()
                    batches_since_flush = 0

            while True:
                message = work_queue.get()
                message_type = message[0]

                if message_type == _BATCH:
                    _, batch_idx, payload = message
                    batch = pickle.loads(payload)

                    if batch_idx in pending or batch_idx < next_batch_idx:
                        raise ValueError(f"Duplicate batch index received: {batch_idx}")

                    pending[batch_idx] = batch

                    while next_batch_idx in pending:
                        write_batch(pending.pop(next_batch_idx))
                        next_batch_idx += 1
                    continue

                if message_type == _STOP:
                    # A failed generator may leave a gap. At shutdown, preserve
                    # deterministic ordering among all batches that did arrive.
                    for batch_idx in sorted(pending):
                        write_batch(pending[batch_idx])
                    hdf5_file.flush()
                    break

                raise ValueError(f"Unknown writer message type: {message_type!r}")
    except BaseException:  # propagate the complete child-process failure
        error_queue.put(traceback.format_exc())
        raise


class HDF5Writer(FileWriter):
    """Write generated TorchSig batches through a dedicated HDF5 process.

    Calls to :meth:`write` are safe from multiple generation processes. Each
    call serializes its batch into a multiprocessing queue; only the dedicated
    writer process opens or mutates ``data.h5``.
    """

    def __init__(
        self,
        root: str | Path,
        compression: str | None = "lzf",
        compression_opts: int | None = None,
        shuffle: bool = True,
        fletcher32: bool = True,
        chunk_cache_size: int = 10 * 1024 * 1024,
        max_batches_in_memory: int = 4,
        multiprocessing_context: str | None = None,
    ) -> None:
        """Initialize the process-safe HDF5 writer.

        Args:
            root: Directory in which ``data.h5`` is created.
            compression: HDF5 compression filter, or None.
            compression_opts: Compression-specific options.
            shuffle: Enable the HDF5 shuffle filter.
            fletcher32: Enable Fletcher32 checksums.
            chunk_cache_size: HDF5 raw chunk cache size in bytes.
            max_batches_in_memory: Queue capacity and file-flush interval.
            multiprocessing_context: Optional start method, such as ``spawn``
                or ``forkserver``. The platform default is used when omitted.
        """
        if max_batches_in_memory < 1:
            raise ValueError("max_batches_in_memory must be at least 1")

        super().__init__(root=root)
        self.datapath = self.root.joinpath("data.h5")
        self.compression = compression
        self.compression_opts = compression_opts
        self.shuffle = shuffle
        self.fletcher32 = fletcher32
        self.chunk_cache_size = chunk_cache_size
        self.max_batches_in_memory = max_batches_in_memory

        self._owner_pid = os.getpid()
        self._closed = False
        self._context = mp.get_context(multiprocessing_context)
        self._work_queue = self._context.Queue(maxsize=max_batches_in_memory)
        self._error_queue = self._context.Queue()
        self._writer_process = self._context.Process(
            target=_writer_process_main,
            args=(self._config(), self._work_queue, self._error_queue),
            name="torchsig-hdf5-writer",
            daemon=False,
        )
        self._writer_process.start()

    def _config(self) -> _WriterConfig:
        """Return the serializable writer-process configuration."""
        return _WriterConfig(
            datapath=str(self.datapath),
            compression=self.compression,
            compression_opts=self.compression_opts,
            shuffle=self.shuffle,
            fletcher32=self.fletcher32,
            chunk_cache_size=self.chunk_cache_size,
            flush_every_n_batches=self.max_batches_in_memory,
        )

    def __getstate__(self) -> dict[str, Any]:
        """Make worker copies lightweight while retaining the shared queues."""
        state = self.__dict__.copy()
        state["_writer_process"] = None
        state["_context"] = None
        return state

    def _raise_writer_error(self) -> None:
        """Raise a child writer failure in the calling process, if present."""
        try:
            error = self._error_queue.get_nowait()
        except queue.Empty:
            error = None

        if error is not None:
            raise RuntimeError(f"HDF5 writer process failed:\n{error}")

        process = self._writer_process
        if process is not None and process.exitcode not in (None, 0):
            raise RuntimeError(
                f"HDF5 writer process exited with code {process.exitcode}"
            )

    def _prepare_signal_for_multiprocessing(self, signal: Signal) -> Signal:
        """Convert tensor-backed signal data to independently pickled NumPy arrays."""
        data = signal.data

        if hasattr(data, "detach"):
            signal.data = data.detach().cpu().numpy().copy()
        elif isinstance(data, np.ndarray):
            signal.data = data.copy()

        for component_signal in signal.component_signals:
            self._prepare_signal_for_multiprocessing(component_signal)

        return signal

    def write(self, batch_idx: int, data: Sequence[Signal]) -> None:
        """Enqueue one generated batch for ordered HDF5 serialization."""
        if self._closed:
            raise RuntimeError("Cannot write after HDF5Writer.teardown()")
        if batch_idx < 0:
            raise ValueError("batch_idx must be non-negative")

        self._raise_writer_error()

        prepared_data = [
            self._prepare_signal_for_multiprocessing(signal)
            for signal in data
        ]

        payload = pickle.dumps(
            prepared_data,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        self._work_queue.put((_BATCH, int(batch_idx), payload))

        self._raise_writer_error()

    def teardown(self) -> None:
        """Drain queued work, close the HDF5 file, and join the writer process."""
        if self._closed:
            return
        if os.getpid() != self._owner_pid:
            # Worker copies share the queues but must not stop the owner process.
            return

        self._closed = True
        process = self._writer_process
        if process is None:
            return

        self._raise_writer_error()
        self._work_queue.put((_STOP,))
        process.join()
        self._raise_writer_error()

        self._work_queue.close()
        self._work_queue.join_thread()
        self._error_queue.close()
        self._error_queue.join_thread()

    def __len__(self) -> int:
        """Return the number of indexed samples after writing has completed."""
        if not self._closed:
            raise RuntimeError("Dataset length is available only after teardown()")
        with h5py.File(self.datapath, "r") as hdf5_file:
            return len(hdf5_file["index"])


def handle_bytes_as_string(value: Any) -> Any:
    """Decode byte strings returned by h5py."""
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray) and value.dtype.kind in {"O", "S"}:
        return value.astype(np.str_)
    return value


def load_value_from_group(group: h5py.Group, key: str) -> Any:
    """Load and decode a value from an HDF5 group."""
    return handle_bytes_as_string(group[key][()])


def fill_object_metadata_from_group_and_id(
    obj: Any,
    group: h5py.Group,
    id_str: str,
) -> Any:
    """Populate an object's metadata and recursively reconstruct its parents."""
    metadata_group = group["metadata"][id_str]
    for key in metadata_group.keys():
        if key != "parent_metadata_id":
            obj[key] = load_value_from_group(metadata_group, key)

    if "parent_metadata_id" in metadata_group:
        parent_id = str(load_value_from_group(metadata_group, "parent_metadata_id"))
        parent = fill_object_metadata_from_group_and_id(
            HierarchicalMetadataObject(),
            group,
            parent_id,
        )
        obj.add_parent(parent)

    return obj


def load_signal_from_group_by_id(group: h5py.Group, id_str: str) -> Signal:
    """Load a signal and its recursively nested components by HDF5 ID."""
    component_signals: list[Signal] = []
    if id_str in group["component_signals"]:
        component_ids = np.atleast_1d(
            load_value_from_group(group["component_signals"], id_str)
        )
        component_signals = [
            load_signal_from_group_by_id(group, str(component_id))
            for component_id in component_ids
        ]

    signal = Signal(
        data=load_value_from_group(group["data"], id_str),
        component_signals=component_signals,
    )
    return fill_object_metadata_from_group_and_id(signal, group, id_str)


def load_signal_from_group_by_index(group: h5py.Group, index: int) -> Signal:
    """Load a signal by its integer dataset index."""
    id_str = str(load_value_from_group(group["index"], str(index)))
    return load_signal_from_group_by_id(group, id_str)


class HDF5Reader(FileReader):
    """Read TorchSig HDF5 data safely from DataLoader worker processes."""

    def __init__(self, root: str | Path) -> None:
        """Initialize a reader whose HDF5 handle is opened lazily per process."""
        super().__init__(root=root)
        self.datapath = self.root.joinpath("data.h5")
        self._file: h5py.File | None = None
        self._file_pid: int | None = None
        self._len_cache: int | None = None
        self._locking = False

    def __getstate__(self) -> dict[str, Any]:
        """Exclude non-pickleable HDF5 state when spawning workers."""
        state = self.__dict__.copy()
        state["_file"] = None
        state["_file_pid"] = None
        return state

    def __len__(self) -> int:
        """Return the number of indexed top-level signals."""
        if self._len_cache is None:
            with h5py.File(self.datapath, "r", locking=self._locking) as hdf5_file:
                self._len_cache = len(hdf5_file["index"])
        return self._len_cache

    def read(self, idx: int) -> Signal:
        """Read one top-level signal by index."""
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        self._ensure_open()
        assert self._file is not None
        return load_signal_from_group_by_index(self._file, idx)

    def teardown(self) -> None:
        """Close this process's HDF5 file handle."""
        if self._file is not None:
            self._file.close()
            self._file = None
            self._file_pid = None

    def _ensure_open(self) -> None:
        """Open a fresh HDF5 handle when the current process changes."""
        current_pid = os.getpid()
        if self._file is not None and self._file_pid != current_pid:
            self._file.close()
            self._file = None

        if self._file is None:
            self._file = h5py.File(
                self.datapath,
                "r",
                locking=self._locking,
            )
            self._file_pid = current_pid


class HDF5FileHandler(BaseFileHandler):
    """Create TorchSig HDF5 readers and process-safe writers."""

    reader_class: type[FileReader] = HDF5Reader
    writer_class: type[FileWriter] = HDF5Writer

    @staticmethod
    def create_handler(
        mode: str,
        root: str | Path,
        **kwargs: Any,
    ) -> HDF5Writer | HDF5Reader:
        """Create an HDF5 reader for ``r`` or writer for ``w`` mode."""
        if mode == "r":
            return HDF5FileHandler.reader_class(root, **kwargs)
        if mode == "w":
            return HDF5FileHandler.writer_class(root, **kwargs)
        raise ValueError(f"Invalid file-handler mode: {mode!r}")

