"""Direct writer for TorchSig's existing HDF5 dataset layout.

The implementation is kept separate from :mod:`hdf5` so it can be measured
against the current writer without changing that production path.
"""

from __future__ import annotations

from typing import Any

from torchsig.utils.file_handlers.hdf5 import HDF5Writer

__all__ = ["DirectHDF5Writer"]


def _object_key(obj: Any) -> str:
    """Return the key assigned by the writer or a stable persistent key."""
    return getattr(obj, "_hdf5_key", str(id(obj)))


class DirectHDF5Writer(HDF5Writer):
    """Write the current HDF5 schema with less per-signal bookkeeping.

    This writer remains compatible with :class:`HDF5Reader`. It binds HDF5
    groups once per file and uses an in-memory set for metadata deduplication,
    avoiding repeated HDF5 membership queries for shared parent metadata.
    """

    def __init__(self, root, **kwargs: Any) -> None:
        super().__init__(root=root, **kwargs)
        self._written_metadata_keys: set[str] = set()
        self._index_count = 0

    def _setup(self) -> None:
        """Create the standard layout and bind its groups once."""
        super()._setup()
        self._data_group = self._file["data"]
        self._metadata_group = self._file["metadata"]
        self._index_group = self._file["index"]
        self._component_group = self._file["component_signals"]
        self._written_metadata_keys.clear()
        self._index_count = 0

    def _write_metadata(self, metadata_obj: Any) -> None:
        key = _object_key(metadata_obj)
        if key in self._written_metadata_keys:
            return

        metadata_group = self._metadata_group.create_group(key)
        self._written_metadata_keys.add(key)
        # HierarchicalMetadataObject is not iterable; its keys() API is required.
        for name in metadata_obj.keys():  # noqa: SIM118
            value = metadata_obj[name]
            if value is not None:
                metadata_group.create_dataset(name, data=value)

        parent = metadata_obj.parent
        if parent is not None:
            metadata_group.create_dataset(
                "parent_metadata_id", data=_object_key(parent)
            )
            self._write_metadata(parent)

    def _write_signal(self, signal: Any, dataset_kwargs: dict[str, Any]) -> None:
        key = _object_key(signal)
        self._write_metadata(signal)
        self._data_group.create_dataset(key, data=signal.data, **dataset_kwargs)

        if signal.component_signals:
            self._component_group.create_dataset(
                key,
                data=[_object_key(component) for component in signal.component_signals],
            )
            for component in signal.component_signals:
                self._write_signal(component, dataset_kwargs)

    def _write_batch_to_hdf5(self, data) -> None:
        """Write a batch directly through cached group bindings."""
        dataset_kwargs = self._data_dataset_kwargs()
        for signal in data:
            self._assign_hdf5_keys(signal)
            self._write_signal(signal, dataset_kwargs)
            self._index_group.create_dataset(
                str(self._index_count), data=_object_key(signal)
            )
            self._index_count += 1
