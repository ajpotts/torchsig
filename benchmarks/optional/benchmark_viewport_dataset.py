"""Benchmark generation-time viewport component insertion."""

from __future__ import annotations

import numpy as np
import pytest

from torchsig.datasets.prototype_viewport_dataset import PrototypeViewportDataset
from torchsig.signals.signal_types import Signal

SAMPLE_RATE = 16_000_000.0
NUM_SAMPLES = 131_072


class BenchmarkDataset(PrototypeViewportDataset):
    """Supply deterministic generation inputs for stable measurements."""

    def _build_noise_floor(self) -> np.ndarray:
        """Return deterministic canvas noise."""
        return np.zeros(NUM_SAMPLES, dtype=np.complex64)

    def _generate_placed_components(self) -> list[Signal]:
        """Return one full-canvas tone component."""
        samples = np.arange(NUM_SAMPLES)
        data = np.exp(2j * np.pi * 2_000_000 * samples / SAMPLE_RATE).astype(
            np.complex64
        )
        return [
            Signal(
                data=data,
                start_in_samples=0,
                duration_in_samples=NUM_SAMPLES,
                center_freq=2_000_000,
                bandwidth=1_000_000,
            )
        ]


@pytest.mark.benchmark(group="viewport-generation")
def test_generation_time_viewport(benchmark) -> None:
    """Benchmark constructing output noise and component IQ directly."""
    dataset = BenchmarkDataset(
        viewport_num_iq_samples=32_768,
        viewport_sample_rate=8_000_000,
        viewport_time_start=32_768,
        viewport_center_freq=2_000_000,
        signal_generators=[],
        validate_init=False,
        metadata={
            "sample_rate": SAMPLE_RATE,
            "num_iq_samples_dataset": NUM_SAMPLES,
            "frequency_min": -SAMPLE_RATE / 2,
            "frequency_max": SAMPLE_RATE / 2,
        },
    )
    benchmark(lambda: next(dataset))
