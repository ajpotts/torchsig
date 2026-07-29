"""Compare packed and homogeneous-top-level HDF5 layouts.

Run with:
    pytest benchmarks/optional/benchmark_hdf5_homogeneous.py --benchmark-only
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from torchsig.signals.signal_types import Signal
from torchsig.utils.file_handlers.hdf5_batched import (
    BatchedHDF5Reader,
    BatchedHDF5Writer,
)
from torchsig.utils.file_handlers.hdf5_homogeneous import (
    HomogeneousHDF5Reader,
    HomogeneousHDF5Writer,
)

NUM_SIGNALS = 512
NUM_SAMPLES = 2_048
BATCH_SIZE = 32
READ_BATCH_SIZE = 64
RANDOM_READS = 64

FORMATS: dict[str, tuple[Callable, Callable]] = {
    "packed": (BatchedHDF5Writer, BatchedHDF5Reader),
    "homogeneous": (HomogeneousHDF5Writer, HomogeneousHDF5Reader),
}


@pytest.fixture(scope="module")
def signals() -> list[Signal]:
    """Create fixed-shape observations with variable component counts."""
    rng = np.random.default_rng(0)
    result = []
    for idx in range(NUM_SIGNALS):
        data = (rng.standard_normal(NUM_SAMPLES) + 1j * rng.standard_normal(NUM_SAMPLES)).astype(np.complex64)
        components = [
            Signal(
                data=data[component_idx * 64 : component_idx * 64 + 64 + 16 * component_idx],
                component_index=component_idx,
            )
            for component_idx in range(idx % 4)
        ]
        result.append(
            Signal(
                data=data,
                component_signals=components,
                sample_index=idx,
            )
        )
    return result


def _write(
    writer_class: Callable,
    root: Path,
    signals: list[Signal],
) -> int:
    writer = writer_class(
        root,
        compression=None,
        shuffle=False,
        fletcher32=False,
    )
    writer.setup()
    try:
        for batch_idx, start in enumerate(range(0, len(signals), BATCH_SIZE)):
            writer.write(batch_idx, signals[start : start + BATCH_SIZE])
    finally:
        writer.teardown()
    return (root / "data.h5").stat().st_size


def _random_read(reader, indices: tuple[int, ...]) -> float:
    return sum(float(reader.read(idx).data[0].real) for idx in indices)


def _packed_contiguous_read(reader: BatchedHDF5Reader, start: int, stop: int) -> np.ndarray:
    return np.stack([reader.read(idx).data for idx in range(start, stop)])


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("format_name", "writer_class"),
    [(name, classes[0]) for name, classes in FORMATS.items()],
    ids=FORMATS,
)
def test_benchmark_homogeneous_format_write(
    benchmark,
    tmp_path,
    signals,
    format_name,
    writer_class,
) -> None:
    """Measure full writes with variable component counts."""
    del format_name
    root = tmp_path / "dataset"
    file_size = benchmark(_write, writer_class, root, signals)
    benchmark.extra_info["file_size_mib"] = file_size / (1024**2)
    benchmark.extra_info["signals"] = len(signals)


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("format_name", "writer_class", "reader_class"),
    [(name, *classes) for name, classes in FORMATS.items()],
    ids=FORMATS,
)
def test_benchmark_homogeneous_format_random_read(
    benchmark,
    tmp_path,
    signals,
    format_name,
    writer_class,
    reader_class,
) -> None:
    """Measure warm random reads including component reconstruction."""
    root = tmp_path / f"dataset-{format_name}"
    _write(writer_class, root, signals)
    indices = tuple(int(value) for value in np.random.default_rng(1).integers(0, len(signals), size=RANDOM_READS))
    reader = reader_class(root)
    try:
        assert np.isfinite(_random_read(reader, indices))
        assert np.isfinite(benchmark(_random_read, reader, indices))
    finally:
        reader.teardown()


@pytest.mark.benchmark
@pytest.mark.parametrize("format_name", FORMATS, ids=FORMATS)
def test_benchmark_homogeneous_format_contiguous_batch_read(
    benchmark,
    tmp_path,
    signals,
    format_name,
) -> None:
    """Measure top-level-only contiguous batch reads."""
    writer_class, reader_class = FORMATS[format_name]
    root = tmp_path / f"dataset-{format_name}"
    _write(writer_class, root, signals)
    start = NUM_SIGNALS // 2
    stop = start + READ_BATCH_SIZE
    reader = reader_class(root)
    try:
        actual = (
            benchmark(reader.read_batch, start, stop)
            if isinstance(reader, HomogeneousHDF5Reader)
            else benchmark(
                _packed_contiguous_read,
                reader,
                start,
                stop,
            )
        )
        assert actual.shape == (READ_BATCH_SIZE, NUM_SAMPLES)
    finally:
        reader.teardown()
