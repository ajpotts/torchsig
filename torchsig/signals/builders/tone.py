"""Tone Signal Builder and Modulator Module"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from torchsig.signals.builder import BaseSignalGenerator
from torchsig.signals.signal_types import Signal
from torchsig.utils.dsp import TorchSigComplexDataType

__all__ = ["ToneGenerationParameters", "ToneSignalGenerator", "tone_modulator"]


@dataclass(frozen=True)
class ToneGenerationParameters:
    """Parameters sampled independently of tone waveform generation.

    Attributes:
        num_samples: Total duration of the conceptual tone burst.
    """

    num_samples: int


def tone_modulator(num_samples: int) -> np.ndarray:
    """Implements a tone modulator.

    Generates a constant tone signal at baseband (all ones).

    Args:
        num_samples: Number of samples to generate.

    Returns:
        np.ndarray: Tone signal (array of ones) with shape (num_samples,).

    Raises:
        ValueError: If num_samples is not positive.
    """
    # Input validation
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    # Generate tone signal (all ones at baseband)
    return np.ones(num_samples, dtype=TorchSigComplexDataType)


class ToneSignalGenerator(BaseSignalGenerator):
    """Tone Signal Generator.

    Implements tone waveforms with configurable parameters.
    """

    def __init__(self, **kwargs: dict[str, str | float | int]) -> None:
        """Initializes Tone Signal Generator.

        Args:
            **kwargs: Metadata parameters including:
                - signal_duration_in_samples_min: Minimum signal duration (samples)
                - signal_duration_in_samples_max: Maximum signal duration (samples)

        Raises:
            ValueError: If required metadata fields are missing or invalid.
        """
        super().__init__(**kwargs)
        self.required_metadata_fields = [
            "signal_duration_in_samples_min",
            "signal_duration_in_samples_max",
        ]
        self.set_default_class_name("tone")

    def generate(self) -> Signal:
        """Generates a tone signal based on the configured parameters.

        Returns:
            Signal: Generated tone signal with metadata.

        Raises:
            ValueError: If required metadata fields are missing or invalid.
        """
        num_samples = self.random_generator.integers(
            low=self["signal_duration_in_samples_min"],
            high=self["signal_duration_in_samples_max"] + 1,
        )
        return Signal(
            data=tone_modulator(num_samples),
            center_freq=0,
            bandwidth=1,
        )

    def sample_parameters(self) -> ToneGenerationParameters:
        """Sample tone metadata without generating its IQ samples.

        Returns:
            Parameters describing the complete conceptual tone burst.
        """
        num_samples = int(
            self.random_generator.integers(
                low=self["signal_duration_in_samples_min"],
                high=self["signal_duration_in_samples_max"] + 1,
            )
        )
        return ToneGenerationParameters(num_samples=num_samples)

    def generate_from_parameters(
        self,
        parameters: ToneGenerationParameters,
    ) -> Signal:
        """Generate a complete tone from previously sampled parameters.

        Args:
            parameters: Parameters describing the conceptual tone burst.

        Returns:
            Complete baseband tone signal.
        """
        return self.generate_segment(
            parameters,
            start_sample=0,
            num_samples=parameters.num_samples,
        )

    def generate_segment(
        self,
        parameters: ToneGenerationParameters,
        *,
        start_sample: int,
        num_samples: int,
    ) -> Signal:
        """Generate only one visible interval of a conceptual tone burst.

        At baseband every tone sample is one, so an arbitrary interval can be
        generated exactly without constructing or discarding the hidden prefix.
        Dataset-level frequency placement can subsequently apply the carrier
        phase appropriate for the interval's position.

        Args:
            parameters: Parameters describing the complete tone burst.
            start_sample: Offset of the interval within the complete burst.
            num_samples: Number of interval samples to generate.

        Returns:
            Baseband tone containing only the requested interval.

        Raises:
            ValueError: If the interval is empty or outside the conceptual burst.
        """
        if start_sample < 0:
            raise ValueError("start_sample must be nonnegative")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if start_sample + num_samples > parameters.num_samples:
            raise ValueError("requested segment exceeds the tone duration")

        return Signal(
            data=tone_modulator(num_samples),
            center_freq=0,
            bandwidth=1,  # Tone has 1Hz bandwidth
            segment_start_in_samples=start_sample,
            original_duration_in_samples=parameters.num_samples,
        )
