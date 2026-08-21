"""Benchmark experimental generator-integrated viewport selection."""

from __future__ import annotations

import numpy as np
import pytest

from torchsig.datasets.datasets import TorchSigIterableDataset
from torchsig.datasets.prototype_viewport_dataset import PrototypeViewportDataset
from torchsig.signals.builders.tone import (
    ToneGenerationParameters,
    ToneSignalGenerator,
)
from torchsig.signals.signal_types import Signal
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from torchsig.utils.writer import identity_collate_fn

SAMPLE_RATE = 16_000_000.0
NUM_SAMPLES = 131_072
OUTPUT_SAMPLE_RATE = 8_000_000.0
OUTPUT_NUM_SAMPLES = 32_768
BATCH_SIZE = 8


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


def output_signal() -> Signal:
    """Construct the output-sized signal used by the regular loader."""
    data = np.ones(OUTPUT_NUM_SAMPLES, dtype=np.complex64)
    component = Signal(
        data=data.copy(),
        class_name="tone",
        start_in_samples=0,
        duration_in_samples=OUTPUT_NUM_SAMPLES,
        center_freq=0,
        bandwidth=1_000_000,
    )
    return Signal(
        data=data,
        component_signals=[component],
        sample_rate=OUTPUT_SAMPLE_RATE,
        num_iq_samples_dataset=OUTPUT_NUM_SAMPLES,
        frequency_min=-OUTPUT_SAMPLE_RATE / 2,
        frequency_max=OUTPUT_SAMPLE_RATE / 2,
        center_freq=0,
        bandwidth=OUTPUT_SAMPLE_RATE,
    )


class FixedOutputDataset(TorchSigIterableDataset):
    """Provide fixed output-sized signals to a regular DataLoader."""

    def __generate_new_signal__(self) -> Signal:
        """Return a fresh deterministic output-sized signal."""
        return output_signal()


class DirectToneViewportDataset(TorchSigIterableDataset):
    """Generate only the visible interval of a conceptual tone burst."""

    def __generate_new_signal__(self) -> Signal:
        """Return a viewport-sized tone without constructing its hidden samples."""
        generator = self.signal_generators[0]
        parameters = ToneGenerationParameters(num_samples=NUM_SAMPLES)
        component = generator.generate_segment(
            parameters,
            start_sample=32_768,
            num_samples=OUTPUT_NUM_SAMPLES,
        )
        component["class_name"] = generator.class_name
        component["start_in_samples"] = 0
        component["duration_in_samples"] = OUTPUT_NUM_SAMPLES
        sample = Signal(
            data=component.data.copy(),
            component_signals=[component],
            center_freq=0,
            bandwidth=OUTPUT_SAMPLE_RATE,
        )
        sample.add_parent(self, register=False)
        component.add_parent(sample, register=False)
        return sample


def viewport_dataset(
    *,
    include_component_data: bool = False,
) -> FixedCanvasViewportDataset:
    """Build the fixed-canvas viewport dataset used by the benchmarks."""
    return FixedCanvasViewportDataset(
        viewport_num_iq_samples=OUTPUT_NUM_SAMPLES,
        viewport_sample_rate=OUTPUT_SAMPLE_RATE,
        viewport_time_start=32_768,
        viewport_center_freq=2_000_000,
        include_component_data=include_component_data,
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


def regular_dataset() -> FixedOutputDataset:
    """Build the fixed output-sized dataset used by the regular loader."""
    return FixedOutputDataset(
        signal_generators=[],
        validate_init=False,
        metadata={
            "sample_rate": OUTPUT_SAMPLE_RATE,
            "num_iq_samples_dataset": OUTPUT_NUM_SAMPLES,
            "frequency_min": -OUTPUT_SAMPLE_RATE / 2,
            "frequency_max": OUTPUT_SAMPLE_RATE / 2,
            "signal_center_freq_min": -OUTPUT_SAMPLE_RATE / 2,
            "signal_center_freq_max": OUTPUT_SAMPLE_RATE / 2,
        },
    )


def direct_tone_viewport_dataset() -> DirectToneViewportDataset:
    """Build a segment-aware tone dataset with viewport-sized output."""
    return DirectToneViewportDataset(
        signal_generators=[
            ToneSignalGenerator(
                signal_duration_in_samples_min=NUM_SAMPLES,
                signal_duration_in_samples_max=NUM_SAMPLES,
            )
        ],
        validate_init=False,
        metadata={
            "sample_rate": OUTPUT_SAMPLE_RATE,
            "num_iq_samples_dataset": OUTPUT_NUM_SAMPLES,
            "frequency_min": -OUTPUT_SAMPLE_RATE / 2,
            "frequency_max": OUTPUT_SAMPLE_RATE / 2,
            "signal_center_freq_min": -OUTPUT_SAMPLE_RATE / 2,
            "signal_center_freq_max": OUTPUT_SAMPLE_RATE / 2,
        },
    )


def loader(dataset: TorchSigIterableDataset) -> WorkerSeedingDataLoader:
    """Wrap a dataset in the same single-process loader configuration."""
    return WorkerSeedingDataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        collate_fn=identity_collate_fn,
        seed=42,
    )


def full_tone_slice(
    generator: ToneSignalGenerator,
    parameters: ToneGenerationParameters,
) -> Signal:
    """Generate a complete conceptual burst before selecting the viewport."""
    signal = generator.generate_from_parameters(parameters)
    signal.data = signal.data[32_768 : 32_768 + OUTPUT_NUM_SAMPLES]
    return signal


def direct_tone_segment(
    generator: ToneSignalGenerator,
    parameters: ToneGenerationParameters,
) -> Signal:
    """Generate only the conceptual burst interval visible in the viewport."""
    return generator.generate_segment(
        parameters,
        start_sample=32_768,
        num_samples=OUTPUT_NUM_SAMPLES,
    )


@pytest.mark.benchmark(group="viewport-selection")
def test_generator_integrated(benchmark) -> None:
    """Benchmark viewport extraction inside dataset iteration."""
    dataset = viewport_dataset()
    benchmark(lambda: next(dataset))


@pytest.mark.benchmark(group="viewport-vs-regular-loader")
@pytest.mark.parametrize(
    ("name", "dataset_factory"),
    [
        ("viewport", viewport_dataset),
        (
            "viewport-with-component-iq",
            lambda: viewport_dataset(include_component_data=True),
        ),
        ("direct-tone-viewport", direct_tone_viewport_dataset),
        ("regular", regular_dataset),
    ],
    ids=[
        "viewport",
        "viewport-with-component-iq",
        "direct-tone-viewport",
        "regular",
    ],
)
def test_loader_output_size_comparison(benchmark, name, dataset_factory) -> None:
    """Compare loaders that return the same batch count and IQ shape."""
    del name
    data_loader = loader(dataset_factory())
    data_iterator = iter(data_loader)

    result = benchmark(lambda: next(data_iterator))

    assert len(result) == BATCH_SIZE
    assert all(sample.data.shape == (OUTPUT_NUM_SAMPLES,) for sample in result)
    assert all(sample.data.dtype == np.complex64 for sample in result)


@pytest.mark.benchmark(group="tone-segment-generation")
@pytest.mark.parametrize(
    ("name", "implementation"),
    [
        ("full-then-slice", full_tone_slice),
        ("direct-segment", direct_tone_segment),
    ],
    ids=["full-then-slice", "direct-segment"],
)
def test_tone_segment_comparison(benchmark, name, implementation) -> None:
    """Compare equal outputs with and without hidden burst generation."""
    del name
    generator = ToneSignalGenerator(
        signal_duration_in_samples_min=NUM_SAMPLES,
        signal_duration_in_samples_max=NUM_SAMPLES,
    )
    parameters = ToneGenerationParameters(num_samples=NUM_SAMPLES)

    result = benchmark(implementation, generator, parameters)

    assert result.data.shape == (OUTPUT_NUM_SAMPLES,)
    assert result.data.dtype == np.complex64
