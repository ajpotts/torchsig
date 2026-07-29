"""Compare packed and homogeneous HDF5 layouts across signal workloads.

The matrix covers narrowband IQ, wideband IQ, and spectrogram observations,
each with variable component counts and with compression enabled and disabled.

Run the complete matrix with:
    pytest benchmarks/optional/benchmark_hdf5_homogeneous.py --benchmark-only

Run one workload with, for example:
    pytest benchmarks/optional/benchmark_hdf5_homogeneous.py \
        --benchmark-only -k wideband
"""

from collections.abc import Callable
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Workload:
    """Configuration for one representative signal workload."""

    name: str
    signal_count: int
    shape: tuple[int, ...]
    dtype: np.dtype
    max_components: int
    batch_size: int
    read_count: int


WORKLOADS = (
    Workload(
        name="narrowband-iq",
        signal_count=512,
        shape=(2_048,),
        dtype=np.dtype(np.complex64),
        max_components=3,
        batch_size=32,
        read_count=64,
    ),
    Workload(
        name="wideband-iq",
        signal_count=96,
        shape=(65_536,),
        dtype=np.dtype(np.complex64),
        max_components=12,
        batch_size=8,
        read_count=16,
    ),
    Workload(
        name="spectrogram",
        signal_count=128,
        shape=(128, 256),
        dtype=np.dtype(np.float32),
        max_components=8,
        batch_size=16,
        read_count=16,
    ),
)

FORMATS: dict[str, tuple[Callable, Callable]] = {
    "packed": (BatchedHDF5Writer, BatchedHDF5Reader),
    "homogeneous": (HomogeneousHDF5Writer, HomogeneousHDF5Reader),
}
COMPRESSIONS = {
    "none": None,
    "lzf": "lzf",
}
CASES = tuple(
    (workload, compression_name, compression, format_name, *classes) for workload in WORKLOADS for compression_name, compression in COMPRESSIONS.items() for format_name, classes in FORMATS.items()
)


def _case_id(case: tuple) -> str:
    workload, compression_name, _, format_name, *_ = case
    return f"{workload.name}-{compression_name}-{format_name}"


def _random_array(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    dtype: np.dtype,
) -> np.ndarray:
    if np.issubdtype(dtype, np.complexfloating):
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dtype)
    return rng.standard_normal(shape).astype(dtype)


def _iq_components(data: np.ndarray, count: int) -> list[Signal]:
    components = []
    for component_idx in range(count):
        length = 256 + 32 * component_idx
        start = component_idx * 1_024
        components.append(
            Signal(
                data=data[start : start + length],
                component_index=component_idx,
            )
        )
    return components


def _spectrogram_components(data: np.ndarray, count: int) -> list[Signal]:
    components = []
    for component_idx in range(count):
        height = 8 + component_idx
        width = 16 + 2 * component_idx
        row = (component_idx * 11) % (data.shape[0] - height + 1)
        column = (component_idx * 17) % (data.shape[1] - width + 1)
        components.append(
            Signal(
                data=data[row : row + height, column : column + width],
                component_index=component_idx,
            )
        )
    return components


def _make_signals(workload: Workload) -> list[Signal]:
    rng = np.random.default_rng(0)
    signals = []
    for idx in range(workload.signal_count):
        data = _random_array(rng, workload.shape, workload.dtype)
        component_count = idx % (workload.max_components + 1)
        components = _spectrogram_components(data, component_count) if workload.name == "spectrogram" else _iq_components(data, component_count)
        signals.append(
            Signal(
                data=data,
                component_signals=components,
                sample_index=idx,
            )
        )
    return signals


def _write(
    writer_class: Callable,
    root: Path,
    signals: list[Signal],
    workload: Workload,
    compression: str | None,
) -> int:
    writer = writer_class(
        root,
        compression=compression,
        shuffle=compression is not None,
        fletcher32=False,
    )
    writer.setup()
    try:
        for batch_idx, start in enumerate(range(0, len(signals), workload.batch_size)):
            writer.write(
                batch_idx,
                signals[start : start + workload.batch_size],
            )
    finally:
        writer.teardown()
    return (root / "data.h5").stat().st_size


def _random_read(reader, indices: tuple[int, ...]) -> float:
    return sum(float(reader.read(idx).data.reshape(-1)[0].real) for idx in indices)


def _packed_contiguous_read(
    reader: BatchedHDF5Reader,
    start: int,
    stop: int,
) -> np.ndarray:
    return np.stack([reader.read(idx).data for idx in range(start, stop)])


def _set_extra_info(
    benchmark,
    workload: Workload,
    compression_name: str,
    file_size: int,
) -> None:
    benchmark.extra_info["workload"] = workload.name
    benchmark.extra_info["compression"] = compression_name
    benchmark.extra_info["signals"] = workload.signal_count
    benchmark.extra_info["shape"] = workload.shape
    benchmark.extra_info["max_components"] = workload.max_components
    benchmark.extra_info["file_size_mib"] = file_size / (1024**2)


@pytest.mark.benchmark
@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_benchmark_homogeneous_format_write(
    benchmark,
    tmp_path,
    case,
) -> None:
    """Measure full writes and record the resulting file size."""
    (
        workload,
        compression_name,
        compression,
        _,
        writer_class,
        _,
    ) = case
    signals = _make_signals(workload)
    root = tmp_path / "dataset"
    file_size = benchmark(
        _write,
        writer_class,
        root,
        signals,
        workload,
        compression,
    )
    _set_extra_info(benchmark, workload, compression_name, file_size)


@pytest.mark.benchmark
@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_benchmark_homogeneous_format_random_read(
    benchmark,
    tmp_path,
    case,
) -> None:
    """Measure warm random reads including component reconstruction."""
    (
        workload,
        compression_name,
        compression,
        format_name,
        writer_class,
        reader_class,
    ) = case
    signals = _make_signals(workload)
    root = tmp_path / f"dataset-{format_name}"
    file_size = _write(
        writer_class,
        root,
        signals,
        workload,
        compression,
    )
    indices = tuple(
        int(value)
        for value in np.random.default_rng(1).integers(
            0,
            len(signals),
            size=workload.read_count,
        )
    )
    reader = reader_class(root)
    try:
        assert np.isfinite(_random_read(reader, indices))
        assert np.isfinite(benchmark(_random_read, reader, indices))
    finally:
        reader.teardown()
    _set_extra_info(benchmark, workload, compression_name, file_size)


@pytest.mark.benchmark
@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_benchmark_homogeneous_format_contiguous_batch_read(
    benchmark,
    tmp_path,
    case,
) -> None:
    """Measure top-level-only contiguous batch reads."""
    (
        workload,
        compression_name,
        compression,
        format_name,
        writer_class,
        reader_class,
    ) = case
    signals = _make_signals(workload)
    root = tmp_path / f"dataset-{format_name}"
    file_size = _write(
        writer_class,
        root,
        signals,
        workload,
        compression,
    )
    start = (workload.signal_count - workload.read_count) // 2
    stop = start + workload.read_count
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
        assert actual.shape == (workload.read_count, *workload.shape)
    finally:
        reader.teardown()
    _set_extra_info(benchmark, workload, compression_name, file_size)
