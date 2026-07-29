"""Tests for the homogeneous-top-level HDF5 prototype."""

import h5py
import numpy as np
import pytest

from torchsig.signals.signal_types import Signal
from torchsig.utils.abstractions import HierarchicalMetadataObject
from torchsig.utils.file_handlers.hdf5_homogeneous import (
    HomogeneousHDF5Reader,
    HomogeneousHDF5Writer,
)


def _signals() -> list[Signal]:
    result = []
    for idx, component_count in enumerate((0, 1, 3)):
        components = [
            Signal(
                data=np.arange(
                    2 + component_idx,
                    dtype=np.float32 if component_idx % 2 == 0 else np.int16,
                ).reshape((2, 2) if component_idx == 2 else (-1,)),
                component_index=component_idx,
            )
            for component_idx in range(component_count)
        ]
        result.append(
            Signal(
                data=np.full((2, 8), idx, dtype=np.complex64),
                component_signals=components,
                sample_index=idx,
            )
        )
    return result


def _write_signals(root) -> None:
    with HomogeneousHDF5Writer(root) as writer:
        writer.write(0, _signals())


def test_homogeneous_hdf5_round_trip_variable_components(tmp_path) -> None:
    signals = _signals()
    with HomogeneousHDF5Writer(
        tmp_path,
        compression=None,
        shuffle=False,
        fletcher32=False,
    ) as writer:
        writer.write(0, signals[:2])
        writer.write(1, signals[2:])

    reader = HomogeneousHDF5Reader(tmp_path)
    try:
        assert len(reader) == len(signals)
        for idx, expected in enumerate(signals):
            actual = reader.read(idx)
            np.testing.assert_array_equal(actual.data, expected.data)
            assert actual.data.dtype == expected.data.dtype
            assert actual["sample_index"] == idx
            assert len(actual.component_signals) == len(expected.component_signals)
            for actual_component, expected_component in zip(
                actual.component_signals,
                expected.component_signals,
                strict=True,
            ):
                np.testing.assert_array_equal(
                    actual_component.data,
                    expected_component.data,
                )
                assert actual_component["component_index"] == expected_component["component_index"]
    finally:
        reader.teardown()


def test_homogeneous_hdf5_writes_schema_version(tmp_path) -> None:
    _write_signals(tmp_path)

    with h5py.File(tmp_path / "data.h5", "r") as file:
        assert file.attrs["schema_version"] == 1


def test_homogeneous_hdf5_rejects_unsupported_schema_version(tmp_path) -> None:
    _write_signals(tmp_path)
    with h5py.File(tmp_path / "data.h5", "r+") as file:
        file.attrs["schema_version"] = 2

    with pytest.raises(ValueError, match=r"Unsupported.*schema version"):
        len(HomogeneousHDF5Reader(tmp_path))


def test_homogeneous_hdf5_rejects_incomplete_file(tmp_path) -> None:
    _write_signals(tmp_path)
    with h5py.File(tmp_path / "data.h5", "r+") as file:
        file.attrs["complete"] = False

    with pytest.raises(ValueError, match="file is incomplete"):
        len(HomogeneousHDF5Reader(tmp_path))


@pytest.mark.parametrize(
    "name",
    ["data", "component_offsets", "component_data"],
)
def test_homogeneous_hdf5_rejects_missing_required_storage(
    tmp_path,
    name,
) -> None:
    _write_signals(tmp_path)
    with h5py.File(tmp_path / "data.h5", "r+") as file:
        del file[name]

    with pytest.raises(ValueError, match="missing required"):
        len(HomogeneousHDF5Reader(tmp_path))


def test_homogeneous_hdf5_rejects_invalid_component_offsets(tmp_path) -> None:
    _write_signals(tmp_path)
    with h5py.File(tmp_path / "data.h5", "r+") as file:
        file["component_offsets"][-1] += 1

    with pytest.raises(ValueError, match="component offsets are invalid"):
        len(HomogeneousHDF5Reader(tmp_path))


def test_homogeneous_hdf5_rejects_top_level_length_mismatch(tmp_path) -> None:
    _write_signals(tmp_path)
    with h5py.File(tmp_path / "data.h5", "r+") as file:
        file["metadata"].resize(len(file["metadata"]) - 1, axis=0)

    with pytest.raises(ValueError, match="data and metadata lengths differ"):
        len(HomogeneousHDF5Reader(tmp_path))


def test_homogeneous_hdf5_rejects_missing_component_stream(tmp_path) -> None:
    _write_signals(tmp_path)
    with h5py.File(tmp_path / "data.h5", "r+") as file:
        del file["component_data"]["0"]

    with pytest.raises(ValueError, match="data stream is missing"):
        len(HomogeneousHDF5Reader(tmp_path))


def test_homogeneous_hdf5_rejects_invalid_component_data_range(tmp_path) -> None:
    _write_signals(tmp_path)
    with h5py.File(tmp_path / "data.h5", "r+") as file:
        record = file["components"][0]
        record["data_offset"] = len(file["component_data"]["0"])
        file["components"][0] = record

    with pytest.raises(ValueError, match="data range is out of bounds"):
        len(HomogeneousHDF5Reader(tmp_path))


def test_homogeneous_hdf5_rejects_component_shape_mismatch(tmp_path) -> None:
    _write_signals(tmp_path)
    with h5py.File(tmp_path / "data.h5", "r+") as file:
        record = file["components"][0]
        record["data_length"] = 1
        file["components"][0] = record

    with pytest.raises(ValueError, match="shape does not match"):
        len(HomogeneousHDF5Reader(tmp_path))


def test_homogeneous_hdf5_reads_native_contiguous_batch(tmp_path) -> None:
    signals = _signals()
    with HomogeneousHDF5Writer(tmp_path) as writer:
        writer.write(0, signals)

    reader = HomogeneousHDF5Reader(tmp_path)
    try:
        actual = reader.read_batch(0, len(signals))
        expected = np.stack([signal.data for signal in signals])
        np.testing.assert_array_equal(actual, expected)
    finally:
        reader.teardown()


@pytest.mark.parametrize(
    ("shape", "dtype", "expected_chunks"),
    [
        ((65_536,), np.complex64, (1, 65_536)),
        ((128, 256), np.float32, (1, 128, 256)),
        ((2_048,), np.complex64, (32, 2_048)),
    ],
    ids=["wideband", "spectrogram", "narrowband"],
)
def test_homogeneous_hdf5_selects_top_level_chunks(
    tmp_path,
    shape,
    dtype,
    expected_chunks,
) -> None:
    signal = Signal(data=np.ones(shape, dtype=dtype))
    with HomogeneousHDF5Writer(
        tmp_path,
        compression="lzf",
        chunk_samples=32,
    ) as writer:
        writer.write(0, [signal])

    with h5py.File(tmp_path / "data.h5", "r") as file:
        assert file["data"].chunks == expected_chunks


@pytest.mark.parametrize(
    "signal",
    [
        Signal(data=np.ones((2, 7), dtype=np.complex64)),
        Signal(data=np.ones((2, 8), dtype=np.float32)),
    ],
    ids=["shape", "dtype"],
)
def test_homogeneous_hdf5_rejects_heterogeneous_top_level_data(tmp_path, signal) -> None:
    writer = HomogeneousHDF5Writer(tmp_path)
    writer.setup()
    writer.write(0, [_signals()[0]])
    with pytest.raises(ValueError, match="share one dtype and shape"):
        writer.write(1, [signal])
    writer.teardown()


def test_homogeneous_hdf5_rejects_parent_metadata(tmp_path) -> None:
    parent = HierarchicalMetadataObject(metadata={"sample_rate": 1.0})
    signal = Signal(
        data=np.ones((2, 8), dtype=np.complex64),
        parent=parent,
    )
    with (
        HomogeneousHDF5Writer(tmp_path) as writer,
        pytest.raises(ValueError, match="does not support parent metadata"),
    ):
        writer.write(0, [signal])


def test_homogeneous_hdf5_rejects_nested_components(tmp_path) -> None:
    nested = Signal(data=np.ones(2, dtype=np.complex64))
    component = Signal(
        data=np.ones(3, dtype=np.complex64),
        component_signals=[nested],
    )
    signal = Signal(
        data=np.ones((2, 8), dtype=np.complex64),
        component_signals=[component],
    )
    with (
        HomogeneousHDF5Writer(tmp_path) as writer,
        pytest.raises(ValueError, match="does not support nested components"),
    ):
        writer.write(0, [signal])
