"""Benchmark experimental generator-integrated viewport selection."""

from __future__ import annotations

import numpy as np
import pytest

from torchsig.datasets.prototype_viewport_dataset import PrototypeViewportDataset
from torchsig.signals.signal_types import Signal

SAMPLE_RATE = 16_000_000.0
NUM_SAMPLES = 131_072
OUTPUT_SAMPLE_RATE = 8_000_000.0
OUTPUT_NUM_SAMPLES = 32_768


def canvas() -> Signal:
    """Construct a deterministic benchmark canvas."""
    samples = np.arange(NUM_SAMPLES)
    data = np.exp(2j * np.pi * 2_000_000 * samples / SAMPLE_RATE).astype(
        np.complex64
    )
    component = Signal(
        data=data.copy(),
        class_name="tone",
        start_in_samples=0,
        duration_in_samples=NUM_SAMPLES,
        center_freq=2_000_000,
        bandwidth=1_000_000,
    )
    return Signal(
        data=data,
        component_signals=[component],
        sample_rate=SAMPLE_RATE,
        num_iq_samples_dataset=NUM_SAMPLES,
        frequency_min=-SAMPLE_RATE / 2,
        frequency_max=SAMPLE_RATE / 2,
        center_freq=0,
        bandwidth=SAMPLE_RATE,
    )


class FixedCanvasViewportDataset(PrototypeViewportDataset):
    """Provide a fixed canvas to isolate integrated lifecycle overhead."""

    def __generate_new_signal__(self) -> Signal:
        """Return a fresh deterministic canvas."""
        return canvas()


@pytest.mark.benchmark(group="viewport-selection")
def test_generator_integrated(benchmark) -> None:
    """Benchmark viewport extraction inside dataset iteration."""
    dataset = FixedCanvasViewportDataset(
        viewport_num_iq_samples=OUTPUT_NUM_SAMPLES,
        viewport_sample_rate=OUTPUT_SAMPLE_RATE,
        viewport_time_start=32_768,
        viewport_center_freq=2_000_000,
        signal_generators=[],
        validate_init=False,
        metadata={
            "sample_rate": SAMPLE_RATE,
            "num_iq_samples_dataset": NUM_SAMPLES,
            "frequency_min": -SAMPLE_RATE / 2,
            "frequency_max": SAMPLE_RATE / 2,
            "signal_center_freq_min": -SAMPLE_RATE / 2,
            "signal_center_freq_max": SAMPLE_RATE / 2,
        },
    )
    benchmark(lambda: next(dataset))
