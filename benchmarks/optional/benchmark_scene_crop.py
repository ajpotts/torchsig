"""Optional benchmarks for the experimental scene-crop transform."""

from __future__ import annotations

import numpy as np
import pytest

from torchsig.signals.signal_types import Signal
from torchsig.transforms.prototype_scene_crop import PrototypeSceneCrop


def _tone(frequency: float, sample_rate: float, length: int) -> np.ndarray:
    """Return a complex64 benchmark tone."""
    samples = np.arange(length)
    return np.exp(2j * np.pi * frequency * samples / sample_rate).astype(
        np.complex64
    )


@pytest.fixture
def expanded_scene() -> Signal:
    """Return a deterministic expanded scene with three components."""
    sample_rate = 20_000_000.0
    length = 524_288
    components = []
    data = np.zeros(length, dtype=np.complex64)
    for index, (start, duration, center) in enumerate(
        [
            (32_768, 131_072, -4_000_000.0),
            (180_000, 196_608, 1_000_000.0),
            (360_000, 131_072, 6_000_000.0),
        ]
    ):
        component_data = _tone(center, sample_rate, duration)
        component = Signal(
            data=component_data,
            class_name=f"tone-{index}",
            start_in_samples=start,
            duration_in_samples=duration,
            center_freq=center,
            bandwidth=2_000_000.0,
        )
        components.append(component)
        data[start : start + duration] += component_data

    return Signal(
        data=data,
        component_signals=components,
        sample_rate=sample_rate,
        num_iq_samples_dataset=length,
        frequency_min=-sample_rate / 2,
        frequency_max=sample_rate / 2,
        center_freq=0.0,
        bandwidth=sample_rate,
    )


@pytest.mark.benchmark(group="prototype-scene-crop")
def test_benchmark_time_only_scene_crop(benchmark, expanded_scene) -> None:
    """Benchmark time cropping without sample-rate reduction."""
    transform = PrototypeSceneCrop(
        num_iq_samples=262_144,
        sample_rate=20_000_000,
        time_start=131_072,
        center_freq=0.0,
    )

    result = benchmark(transform, expanded_scene)

    assert result.data.shape == (262_144,)


@pytest.mark.benchmark(group="prototype-scene-crop")
def test_benchmark_time_frequency_scene_crop(benchmark, expanded_scene) -> None:
    """Benchmark time cropping with subband extraction and decimation."""
    transform = PrototypeSceneCrop(
        num_iq_samples=131_072,
        sample_rate=10_000_000,
        time_start=131_072,
        center_freq=2_000_000.0,
    )

    result = benchmark(transform, expanded_scene)

    assert result.data.shape == (131_072,)
    assert result.data.dtype == np.complex64
