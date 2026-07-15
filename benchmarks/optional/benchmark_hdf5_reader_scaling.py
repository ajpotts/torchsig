"""Scaling benchmarks for deciding whether to enable HDF5 reader caching.

The default synthetic dataset contains 10,000 records. For a 100,000-record
run, or to use an existing dataset instead of generating a fixture:

    TORCHSIG_HDF5_BENCHMARK_RECORDS=100000 \
      pytest benchmarks/optional/benchmark_hdf5_reader_scaling.py \
      --benchmark-only

    TORCHSIG_HDF5_BENCHMARK_ROOT=/path/to/dataset \
      pytest benchmarks/optional/benchmark_hdf5_reader_scaling.py \
      --benchmark-only

Fixture generation is outside every timed region. The root must be the folder
containing ``data.h5``, rather than the HDF5 file itself.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from torchsig.signals.signal_types import Signal
from torchsig.utils.abstractions import HierarchicalMetadataObject
from torchsig.utils.file_handlers.hdf5 import HDF5Reader
from torchsig.utils.file_handlers.hdf5_cached_reader import CachedHDF5Reader
from torchsig.utils.file_handlers.hdf5_direct_writer import DirectHDF5Writer

if TYPE_CHECKING:
    from collections.abc import Callable

DEFAULT_RECORDS = 10_000
DEFAULT_SAMPLES = 256
WRITE_BATCH_SIZE = 256
HOT_READS = 64

READER_IMPLEMENTATIONS: dict[str, Callable] = {
    "current": HDF5Reader,
    "cached": CachedHDF5Reader,
}


def _configured_positive_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _write_synthetic_dataset(root: Path, records: int, samples: int) -> None:
    rng = np.random.default_rng(0)
    parent = HierarchicalMetadataObject(
        metadata={"sample_rate": 1_000_000.0, "dataset_name": "scaling-benchmark"}
    )
    with DirectHDF5Writer(
        root,
        shuffle=False,
        fletcher32=False,
        max_batches_in_memory=1,
    ) as writer:
        for batch_idx, start in enumerate(range(0, records, WRITE_BATCH_SIZE)):
            count = min(WRITE_BATCH_SIZE, records - start)
            matrix = (
                rng.standard_normal((count, samples))
                + 1j * rng.standard_normal((count, samples))
            ).astype(np.complex64)
            batch = []
            for offset, row in enumerate(matrix):
                signal = Signal(
                    data=row,
                    class_name="benchmark",
                    sample_index=start + offset,
                    snr_db=float((start + offset) % 30),
                )
                signal.add_parent(parent, register=False)
                batch.append(signal)
            writer.write(batch_idx, batch)


@pytest.fixture(scope="module")
def scaling_dataset(tmp_path_factory) -> tuple[Path, int]:
    """Return an existing dataset or generate the configured synthetic one."""
    configured_root = os.environ.get("TORCHSIG_HDF5_BENCHMARK_ROOT")
    if configured_root:
        root = Path(configured_root).expanduser().resolve()
        if not (root / "data.h5").is_file():
            raise FileNotFoundError(f"No data.h5 found under {root}")
        reader = HDF5Reader(root)
        try:
            return root, len(reader)
        finally:
            reader.teardown()

    records = _configured_positive_int(
        "TORCHSIG_HDF5_BENCHMARK_RECORDS", DEFAULT_RECORDS
    )
    samples = _configured_positive_int(
        "TORCHSIG_HDF5_BENCHMARK_SAMPLES", DEFAULT_SAMPLES
    )
    root = tmp_path_factory.mktemp("hdf5-reader-scaling")
    _write_synthetic_dataset(root, records, samples)
    return root, records


def _open_count_close(reader_class: Callable, root: Path) -> int:
    reader = reader_class(root)
    try:
        return len(reader)
    finally:
        reader.teardown()


def _scan(reader, indices: range | tuple[int, ...]) -> float:
    checksum = 0.0
    for idx in indices:
        signal = reader.read(idx)
        checksum += float(signal.data[0].real)
    return checksum


def _deep_size(value: Any, seen: set[int] | None = None) -> int:
    """Estimate recursively owned Python memory for reader cache containers."""
    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return 0
    seen.add(value_id)
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(
            _deep_size(key, seen) + _deep_size(item, seen)
            for key, item in value.items()
        )
    elif isinstance(value, (list, tuple, set, frozenset)):
        size += sum(_deep_size(item, seen) for item in value)
    elif isinstance(value, np.ndarray):
        size += value.nbytes
    return size


def _cache_size(reader) -> tuple[int, int]:
    if not isinstance(reader, CachedHDF5Reader):
        return 0, 0
    values = (
        reader._index_ids,  # noqa: SLF001
        reader._metadata_cache,  # noqa: SLF001
        reader._component_id_cache,  # noqa: SLF001
    )
    entries = len(reader._metadata_cache) + len(  # noqa: SLF001
        reader._component_id_cache  # noqa: SLF001
    )
    return _deep_size(values), entries


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("implementation_name", "reader_class"),
    READER_IMPLEMENTATIONS.items(),
    ids=READER_IMPLEMENTATIONS,
)
def test_benchmark_hdf5_reader_open_and_len(
    benchmark,
    scaling_dataset: tuple[Path, int],
    implementation_name: str,
    reader_class: Callable,
) -> None:
    """Expose the cached reader's eager whole-index cost at large scale."""
    del implementation_name
    root, records = scaling_dataset
    result = benchmark(_open_count_close, reader_class, root)
    assert result == records


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("implementation_name", "reader_class"),
    READER_IMPLEMENTATIONS.items(),
    ids=READER_IMPLEMENTATIONS,
)
def test_benchmark_hdf5_reader_cold_full_scan(
    benchmark,
    scaling_dataset: tuple[Path, int],
    implementation_name: str,
    reader_class: Callable,
) -> None:
    """Measure one cold sequential epoch and report resulting cache size."""
    del implementation_name
    root, records = scaling_dataset
    reader = reader_class(root)
    try:
        checksum = benchmark.pedantic(
            _scan,
            args=(reader, range(records)),
            rounds=1,
            iterations=1,
            warmup_rounds=0,
        )
        cache_bytes, cache_entries = _cache_size(reader)
        benchmark.extra_info["records"] = records
        benchmark.extra_info["cache_mib_estimate"] = cache_bytes / (1024**2)
        benchmark.extra_info["cache_entries"] = cache_entries
        assert np.isfinite(checksum)
    finally:
        reader.teardown()


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("implementation_name", "reader_class"),
    READER_IMPLEMENTATIONS.items(),
    ids=READER_IMPLEMENTATIONS,
)
def test_benchmark_hdf5_reader_warm_random_working_set(
    benchmark,
    scaling_dataset: tuple[Path, int],
    implementation_name: str,
    reader_class: Callable,
) -> None:
    """Measure repeated random reads after the selected records are cached."""
    del implementation_name
    root, records = scaling_dataset
    indices = tuple(
        int(idx)
        for idx in np.random.default_rng(1).integers(
            0, records, size=min(HOT_READS, records)
        )
    )
    reader = reader_class(root)
    try:
        assert np.isfinite(_scan(reader, indices))
        checksum = benchmark(_scan, reader, indices)
        assert np.isfinite(checksum)
    finally:
        reader.teardown()
