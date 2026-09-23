"""Basic frequency-hopping spread-spectrum signal generation."""

from __future__ import annotations

import numpy as np

from torchsig.signals.builder import BaseSignalGenerator
from torchsig.signals.signal_types import Signal
from torchsig.utils.dsp import TorchSigComplexDataType

__all__ = ["FrequencyHoppingSignalGenerator", "frequency_hopping_modulator"]


def frequency_hopping_modulator(
    num_samples: int,
    sample_rate: float,
    bandwidth: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a simple, phase-continuous frequency-hopping waveform.

    The number of equal-duration hops and every hop frequency are selected at
    random. This intentionally provides a toy FHSS waveform rather than a
    configurable protocol implementation.

    Args:
        num_samples: Number of IQ samples to generate.
        sample_rate: IQ sample rate in Hz.
        bandwidth: Full frequency span in which hop frequencies are selected.
        rng: Random number generator used for reproducible generation.

    Returns:
        Complex unit-magnitude IQ samples.

    Raises:
        ValueError: If a numeric input is outside its valid range.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    if bandwidth > sample_rate:
        raise ValueError("bandwidth must not exceed sample_rate")

    if rng is None:
        rng = np.random.default_rng()

    max_hops = min(16, num_samples)
    min_hops = min(2, max_hops)
    num_hops = int(rng.integers(min_hops, max_hops + 1))
    hop_frequencies = rng.uniform(-bandwidth / 2, bandwidth / 2, size=num_hops)

    hop_index = np.minimum(
        np.arange(num_samples) * num_hops // num_samples,
        num_hops - 1,
    )
    instantaneous_frequency = hop_frequencies[hop_index]
    phase = 2 * np.pi * np.cumsum(instantaneous_frequency) / sample_rate
    phase -= phase[0]

    return np.exp(1j * phase).astype(TorchSigComplexDataType)


class FrequencyHoppingSignalGenerator(BaseSignalGenerator):
    """Generate a toy FHSS signal with randomly selected hopping behavior."""

    def __init__(self, **kwargs: dict[str, str | float | int]) -> None:
        """Initialize the generator using standard dataset metadata only.

        Args:
            **kwargs: Metadata containing ``sample_rate``, bandwidth bounds,
                and signal-duration bounds.
        """
        super().__init__(**kwargs)
        self.required_metadata_fields = [
            "sample_rate",
            "bandwidth_min",
            "bandwidth_max",
            "signal_duration_in_samples_min",
            "signal_duration_in_samples_max",
        ]
        self.set_default_class_name("fhss")

    def generate(self) -> Signal:
        """Generate an FHSS waveform with random hop count and frequencies."""
        num_samples = int(
            self.random_generator.integers(
                self["signal_duration_in_samples_min"],
                self["signal_duration_in_samples_max"] + 1,
            )
        )
        bandwidth = float(
            self.random_generator.uniform(
                self["bandwidth_min"],
                self["bandwidth_max"],
            )
        )
        data = frequency_hopping_modulator(
            num_samples,
            self["sample_rate"],
            bandwidth,
            self.random_generator,
        )
        return Signal(data=data, center_freq=0, bandwidth=bandwidth)
