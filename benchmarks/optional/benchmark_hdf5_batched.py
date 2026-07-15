"""Optional comparison of object-per-record and packed HDF5 formats.

Run with:
    pytest benchmarks/optional/benchmark_hdf5_batched.py --benchmark-only
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from torchsig.signals.signal_types import Signal
from torchsig.utils.abstractions import HierarchicalMetadataObject
from torchsig.utils.file_handlers.hdf5 import HDF5Reader, HDF5Writer
from torchsig.utils.file_handlers.hdf5_batched import (
    BatchedHDF5Reader,
    BatchedHDF5Writer,
)
from torchsig.utils.file_handlers.hdf5_cached_reader import CachedHDF5Reader

NUM_SIGNALS = 256
NUM_SAMPLES = 2_048
BATCH_SIZE = 32
READS_PER_ROUND = 64

WRITERS: dict[str, Callable] = {
    "current": HDF5Writer,
    "packed": BatchedHDF5Writer,
}

READ_FORMATS: dict[str, tuple[Callable, Callable]] = {
    "current": (HDF5Writer, HDF5Reader),
    "cached": (HDF5Writer, CachedHDF5Reader),
    "packed": (BatchedHDF5Writer, BatchedHDF5Reader),
}


@pytest.fixture(scope="module")
def signals() -> list[Signal]:
    """Create representative top-level and component signals."""
    rng = np.random.default_rng(0)
    parent = HierarchicalMetadataObject(
        metadata={"sample_rate": 1_000_000.0, "dataset_name": "packed-benchmark"}
    )
    result = []
    for idx in range(NUM_SIGNALS):
        data = (
            rng.standard_normal(NUM_SAMPLES)
            + 1j * rng.standard_normal(NUM_SAMPLES)
        ).astype(np.complex64)
        component = Signal(
            data=data[:256],
            class_name="component",
            center_freq=float(idx),
        )
        component.add_parent(parent, register=False)
        signal = Signal(
            data=data,
            component_signals=[component],
            class_name="benchmark",
            sample_index=idx,
        )
        signal.add_parent(parent, register=False)
        result.append(signal)
    return result


def _write(writer_class: Callable, root: Path, signals: list[Signal]) -> int:
    writer = writer_class(
        root,
        shuffle=False,
        fletcher32=False,
        max_batches_in_memory=4,
    )
    writer.setup()
    try:
        for batch_idx, start in enumerate(range(0, len(signals), BATCH_SIZE)):
            writer.write(batch_idx, signals[start : start + BATCH_SIZE])
    finally:
        writer.teardown()
    return (root / "data.h5").stat().st_size


def _read(reader, indices: tuple[int, ...]) -> float:
    return sum(float(reader.read(idx).data[0].real) for idx in indices)


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("format_name", "writer_class"),
    WRITERS.items(),
    ids=WRITERS,
)
def test_benchmark_hdf5_format_write(
    benchmark,
    tmp_path,
    signals: list[Signal],
    format_name: str,
    writer_class: Callable,
) -> None:
    """Measure complete file creation, writing, flushing, and closing."""
    del format_name
    root = tmp_path / "dataset"
    file_size = benchmark(_write, writer_class, root, signals)
    benchmark.extra_info["file_size_mib"] = file_size / (1024**2)
    reader_class = (
        BatchedHDF5Reader
        if writer_class is BatchedHDF5Writer
        else HDF5Reader
    )
    reader = reader_class(root)
    try:
        assert len(reader) == NUM_SIGNALS
        assert reader.read(NUM_SIGNALS - 1).data.shape == (NUM_SAMPLES,)
    finally:
        reader.teardown()


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("format_name", "writer_class", "reader_class"),
    [(name, *classes) for name, classes in READ_FORMATS.items()],
    ids=READ_FORMATS,
)
def test_benchmark_hdf5_format_warm_random_read(
    benchmark,
    tmp_path_factory,
    signals: list[Signal],
    format_name: str,
    writer_class: Callable,
    reader_class: Callable,
) -> None:
    """Measure repeated random access after warming the selected records."""
    root = tmp_path_factory.mktemp(f"packed-reader-{format_name}")
    _write(writer_class, root, signals)
    indices = tuple(
        int(idx)
        for idx in np.random.default_rng(1).integers(
            0, NUM_SIGNALS, size=READS_PER_ROUND
        )
    )
    reader = reader_class(root)
    try:
        assert np.isfinite(_read(reader, indices))
        result = benchmark(_read, reader, indices)
        assert np.isfinite(result)
    finally:
        reader.teardown()
