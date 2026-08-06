"""Tests for the DSP utilities."""

import numpy as np
import torch
import pytest

from torchsig.utils.dsp import (
    TorchSigComplexDataType,
    compute_spectrogram,
)

FFT_SIZE = 64
ZEROS = np.zeros(512, dtype=TorchSigComplexDataType)

def _tone(bin_index, num_samples=512, fft_size=FFT_SIZE, amplitude=1.0):
    """Complex exponential centered exactly on FFT bin ``bin_index``."""
    n = np.arange(num_samples)
    tone = amplitude * np.exp(2j * np.pi * bin_index * n / fft_size)
    return tone.astype(TorchSigComplexDataType)
 

# compute_spectrogram tests
@pytest.mark.parametrize(
    "samples,fft_size,fft_stride",
    [
        (ZEROS, 0, FFT_SIZE),
        (ZEROS, -FFT_SIZE, FFT_SIZE),
        (ZEROS, FFT_SIZE, 0),
        (ZEROS, FFT_SIZE, -1),
        (ZEROS.reshape(2, -1), FFT_SIZE, FFT_SIZE),
    ],
)
def test_compute_spectrogram_invalid_arguments_raise(samples, fft_size, fft_stride):
    """Non-positive sizes and non-1D input raise ValueError."""
    with pytest.raises(ValueError):
        compute_spectrogram(samples, fft_size, fft_stride)
 
@pytest.mark.parametrize(
    "num_samples,fft_stride,expected_frames",
    [
        (1024, FFT_SIZE, 16),          # no overlap
        (1024, FFT_SIZE // 2, 31),     # 50% overlap
        (1024, 2 * FFT_SIZE, 8),       # stride > fft_size, subset sampling
        (100, FFT_SIZE, 1),            # trailing partial frame discarded
        (FFT_SIZE // 4, FFT_SIZE, 1),  # short input, zero padded
    ],
)
def test_compute_spectrogram_output_geometry(num_samples, fft_stride, expected_frames):
    """Shape is (fft_size, 1 + (num_samples - fft_size) // fft_stride), float32."""
    spec = compute_spectrogram(_tone(3, num_samples), FFT_SIZE, fft_stride)
    assert spec.shape == (FFT_SIZE, expected_frames)
    assert spec.dtype == np.float32
 
 
@pytest.mark.parametrize(
    "bin_index", [0, 1, -1, 7, FFT_SIZE // 2 - 1, -FFT_SIZE // 2]
)
def test_compute_spectrogram_frequency_maps_to_descending_rows(bin_index):
    """A tone on bin k peaks at row fft_size // 2 - 1 - k, in every frame."""
    spec = compute_spectrogram(_tone(bin_index), FFT_SIZE, FFT_SIZE)
    expected_row = FFT_SIZE // 2 - 1 - bin_index
    np.testing.assert_array_equal(
        np.argmax(spec, axis=0), np.full(spec.shape[1], expected_row)
    )
 
 
def test_compute_spectrogram_output_is_contiguous_and_torch_safe():
    """No negative strides, so torch.from_numpy accepts the result."""
    spec = compute_spectrogram(_tone(5), FFT_SIZE, FFT_SIZE)
    assert spec.flags["C_CONTIGUOUS"]
    assert all(stride > 0 for stride in spec.strides)
    torch.from_numpy(spec)  # raises on negative strides
 
 
def test_compute_spectrogram_zero_power_bins_are_floored():
    """Exactly-zero bins clamp to peak - 100 dB rather than -inf."""
    silent = np.zeros(256, dtype=TorchSigComplexDataType)
    spec = compute_spectrogram(
        np.concatenate([_tone(4, 256), silent]), FFT_SIZE, FFT_SIZE
    )
    assert np.all(np.isfinite(spec))
    np.testing.assert_allclose(spec.min(), spec.max() - 100.0, atol=1e-3)
 
 
def test_compute_spectrogram_all_zero_input_is_finite():
    """Degenerate all-zero input yields a finite constant, no divide-by-zero."""
    with np.errstate(divide="raise", invalid="raise"):
        spec = compute_spectrogram(ZEROS, FFT_SIZE, FFT_SIZE)
    assert np.all(np.isfinite(spec))
    assert np.all(spec == spec.flat[0])
 
 
def test_compute_spectrogram_values_are_power_db():
    """Doubling amplitude raises the peak by 10*log10(4), not 20*log10(4)."""
    quiet = compute_spectrogram(_tone(5), FFT_SIZE, FFT_SIZE)
    loud = compute_spectrogram(_tone(5, amplitude=2.0), FFT_SIZE, FFT_SIZE)
    np.testing.assert_allclose(
        loud.max() - quiet.max(), 10.0 * np.log10(4.0), atol=1e-3
    )
 
 
def test_compute_spectrogram_input_not_modified():
    """The caller's array is never written to, despite the copy-free asarray."""
    samples = _tone(3)
    original = samples.copy()
    compute_spectrogram(samples, FFT_SIZE, FFT_SIZE)
    np.testing.assert_array_equal(samples, original)
 
 
