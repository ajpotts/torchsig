"""Optional A/B benchmark for the current and cached HDF5 readers.

Run with:
    pytest benchmarks/optional/benchmark_hdf5_readers.py --benchmark-only
"""

from collections.abc import Callable

import numpy as np
import pytest

from torchsig.signals.signal_types import Signal
from torchsig.utils.abstractions import HierarchicalMetadataObject
from torchsig.utils.file_handlers.hdf5 import HDF5Reader, HDF5Writer
from torchsig.utils.file_handlers.hdf5_cached_reader import CachedHDF5Reader

NUM_SIGNALS = 128
NUM_SAMPLES = 4_096
READS_PER_ROUND = 32

READER_IMPLEMENTATIONS: dict[str, Callable] = {
    "current": HDF5Reader,
    "cached": CachedHDF5Reader,
}


@pytest.fixture(scope="module")
def hdf5_dataset(tmp_path_factory):
    """Create one representative file shared by every benchmark case."""
    root = tmp_path_factory.mktemp("hdf5-reader-benchmark")
    rng = np.random.default_rng(0)
    parent = HierarchicalMetadataObject(
        metadata={"sample_rate": 1_000_000.0, "dataset_name": "reader-benchmark"}
    )
    signals = []
    for idx in range(NUM_SIGNALS):
        data = (
            rng.standard_normal(NUM_SAMPLES)
            + 1j * rng.standard_normal(NUM_SAMPLES)
        ).astype(np.complex64)
        component = Signal(
            data=data[:512],
            parent=parent,
            class_name="component",
            center_freq=float(idx),
        )
        signals.append(
            Signal(
                data=data,
                component_signals=[component],
                parent=parent,
                class_name="benchmark",
                sample_index=idx,
            )
        )

    with HDF5Writer(
        root,
        shuffle=False,
        fletcher32=False,
        max_batches_in_memory=1,
    ) as writer:
        writer.write(0, signals)
    return root


def _read_indices(reader, indices: tuple[int, ...]) -> list[Signal]:
    return [reader.read(idx) for idx in indices]


def _benchmark_indices(access_pattern: str) -> tuple[int, ...]:
    if access_pattern == "sequential":
        return tuple(range(READS_PER_ROUND))
    return tuple(
        int(idx)
        for idx in np.random.default_rng(1).integers(
            0, NUM_SIGNALS, size=READS_PER_ROUND
        )
    )


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "access_pattern",
    ["sequential", "random"],
)
@pytest.mark.parametrize(
    ("implementation_name", "reader_class"),
    READER_IMPLEMENTATIONS.items(),
    ids=READER_IMPLEMENTATIONS,
)
def test_benchmark_hdf5_reader(
    benchmark,
    hdf5_dataset,
    implementation_name: str,
    reader_class: Callable,
    access_pattern: str,
) -> None:
    """Compare repeated-epoch sequential reads and a random working set."""
    del implementation_name
    indices = _benchmark_indices(access_pattern)

    reader = reader_class(hdf5_dataset)
    try:
        # Exclude lazy file opening and populate the alternative reader's
        # structural caches. This represents reads after the first epoch.
        assert len(reader) == NUM_SIGNALS
        warm = _read_indices(reader, indices)
        assert all(signal.data.shape == (NUM_SAMPLES,) for signal in warm)

        result = benchmark(_read_indices, reader, indices)
        assert len(result) == READS_PER_ROUND
        assert all(signal.data.dtype == np.complex64 for signal in result)
    finally:
        reader.teardown()
