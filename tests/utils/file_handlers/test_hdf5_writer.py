"""Tests for standard HDF5 writer options."""

import h5py
import numpy as np

from torchsig.signals.signal_types import Signal
from torchsig.transforms.transforms import Spectrogram
from torchsig.utils.file_handlers.hdf5 import HDF5Writer


def test_standard_hdf5_writer_supports_no_compression(tmp_path) -> None:
    """Record disabled compression without passing None to HDF5 attributes."""
    signals = [Signal(data=np.ones(4, dtype=np.complex64))]

    with HDF5Writer(
        tmp_path,
        compression=None,
        shuffle=False,
        fletcher32=False,
    ) as writer:
        writer.write(0, signals)

    with h5py.File(tmp_path / "data.h5", "r") as handle:
        assert handle.attrs["compression"] == "none"
        assert handle["data"]["0"].compression is None


def test_standard_hdf5_writer_persists_spectrogram_support_vectors(tmp_path) -> None:
    """Physical spectrogram axes must survive an HDF5 round trip."""
    signal = Spectrogram(fft_size=4, fft_stride=2)(Signal(data=np.ones(16, dtype=np.complex64), sample_rate=8.0))

    with HDF5Writer(tmp_path, max_batches_in_memory=1) as writer:
        writer.write(0, [signal])

    with h5py.File(tmp_path / "data.h5", "r") as handle:
        metadata = handle["metadata"]["0"]
        np.testing.assert_array_equal(metadata["spectrogram_frequency"][()], signal.spectrogram_frequency)
        np.testing.assert_array_equal(metadata["spectrogram_time"][()], signal.spectrogram_time)
        assert metadata["spectrogram_frequency_units"].asstr()[()] == "Hz"
        assert metadata["spectrogram_time_units"].asstr()[()] == "seconds"
