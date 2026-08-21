"""Tests for the experimental generator-integrated viewport dataset."""

from __future__ import annotations

import numpy as np
import pytest

from torchsig.datasets.prototype_viewport_dataset import PrototypeViewportDataset
from torchsig.signals.signal_types import Signal
from torchsig.transforms.metadata_transforms import YOLOLabel
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from torchsig.utils.writer import identity_collate_fn


def _tone(frequency: float, sample_rate: float, length: int) -> np.ndarray:
    samples = np.arange(length)
    return np.exp(2j * np.pi * frequency * samples / sample_rate).astype(
        np.complex64
    )


def _canvas() -> Signal:
    component = Signal(
        data=_tone(2, 8, 8),
        class_name="tone",
        class_index=1,
        start_in_samples=2,
        duration_in_samples=8,
        center_freq=2,
        bandwidth=2,
    )
    data = np.zeros(16, dtype=np.complex64)
    data[2:10] = component.data
    return Signal(
        data=data,
        component_signals=[component],
        sample_rate=8,
        num_iq_samples_dataset=16,
        frequency_min=-4,
        frequency_max=4,
        center_freq=0,
        bandwidth=8,
    )


class _FixedCanvasDataset(PrototypeViewportDataset):
    """Return a deterministic canvas while exercising viewport lifecycle code."""

    def __generate_new_signal__(self) -> Signal:
        """Return the fixed test canvas."""
        return _canvas()

    def condition_viewport(self, time_start: int, center_freq: float) -> None:
        """Set deterministic active viewport coordinates for placement tests."""
        self._active_time_start = time_start
        self._active_center_freq = center_freq

    def choose_conditioned_start(self, canvas: np.ndarray, signal: Signal) -> int:
        """Expose conditioned time placement for focused testing."""
        return self._choose_start_sample(canvas, signal)

    def conditioned_frequency_bounds(self, signal: Signal) -> tuple[float, float]:
        """Expose conditioned frequency placement for focused testing."""
        return self._frequency_center_bounds(signal)


def _dataset(**kwargs) -> _FixedCanvasDataset:
    return _FixedCanvasDataset(
        viewport_num_iq_samples=4,
        viewport_sample_rate=4,
        viewport_time_start=2,
        viewport_center_freq=2,
        signal_generators=[],
        validate_init=False,
        metadata={
            "sample_rate": 8,
            "num_iq_samples_dataset": 16,
            "frequency_min": -4,
            "frequency_max": 4,
            "signal_center_freq_min": -3,
            "signal_center_freq_max": 3,
        },
        **kwargs,
    )


def test_dataset_returns_cropped_viewport() -> None:
    """Dataset iteration should expose output rather than canvas geometry."""
    dataset = _dataset()
    output = next(dataset)

    assert isinstance(output, Signal)
    assert output.data.shape == (4,)
    assert output.data.dtype == np.complex64
    assert output.sample_rate == 4
    assert dataset.last_canvas is not None
    assert dataset.last_canvas.data.shape == (16,)
    assert len(output.component_signals) == 1
    component = output.component_signals[0]
    assert component.start_in_samples == 0
    assert component.duration_in_samples == 4
    assert component.data.shape == (0,)
    assert component.data.dtype == np.complex64
    assert component.center_freq == pytest.approx(0)


def test_component_iq_processing_can_be_enabled() -> None:
    """Callers can opt into isolated, viewport-relative component IQ data."""
    output = next(_dataset(include_component_data=True))

    component = output.component_signals[0]
    assert component.data.shape == (4,)
    assert component.data.dtype == np.complex64


def test_output_transforms_run_after_viewport_selection() -> None:
    """A dataset transform should receive the cropped metadata geometry."""
    observed_shapes = []

    def record_shape(signal: Signal) -> Signal:
        observed_shapes.append(signal.data.shape)
        return signal

    output = next(_dataset(transforms=[record_shape]))

    assert isinstance(output, Signal)
    assert observed_shapes == [(4,)]


def test_target_extraction_uses_cropped_metadata() -> None:
    """Target labels should be produced after integrated viewport selection."""
    data, target = next(
        _dataset(
            transforms=[YOLOLabel()],
            target_labels=["yolo_label"],
        )
    )

    assert data.shape == (4,)
    assert target == pytest.approx((1, 0.5, 0.5, 1.0, 0.5))


def test_worker_seeding_dataloader_iterates_integrated_dataset() -> None:
    """The prototype should work through TorchSIG's PyTorch DataLoader path."""
    loader = WorkerSeedingDataLoader(
        _dataset(),
        seed=42,
        batch_size=2,
        num_workers=0,
        collate_fn=identity_collate_fn,
    )

    batch = next(iter(loader))

    assert len(batch) == 2
    assert all(signal.data.shape == (4,) for signal in batch)


def test_start_sampling_always_overlaps_viewport() -> None:
    """Every conditioned start should have a positive time intersection."""
    conditioned = _dataset()
    conditioned.condition_viewport(4, 0)
    signal = Signal(data=np.zeros(5, dtype=np.complex64))
    canvas = np.zeros(16, dtype=np.complex64)

    starts = [conditioned.choose_conditioned_start(canvas, signal) for _ in range(100)]

    viewport_stop = 12
    assert all(start < viewport_stop and start + len(signal.data) > 4 for start in starts)
    assert min(starts) >= 0
    assert max(starts) + len(signal.data) <= len(canvas)


def test_frequency_bounds_guarantee_overlap_and_allow_slicing() -> None:
    """Conditioned frequency bounds should include both viewport boundaries."""
    conditioned = _dataset()
    conditioned.condition_viewport(4, 0)
    signal = Signal(data=np.zeros(8, dtype=np.complex64), bandwidth=2)

    lower, upper = conditioned.conditioned_frequency_bounds(signal)

    assert lower == pytest.approx(-3)
    assert upper == pytest.approx(3)
    assert lower - signal.bandwidth / 2 < -2
    assert upper + signal.bandwidth / 2 > 2


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"viewport_num_iq_samples": 0}, "must be positive"),
        ({"viewport_sample_rate": 3}, "integer ratio"),
    ],
)
def test_invalid_viewport_configuration(overrides: dict, message: str) -> None:
    """Invalid output geometry should fail during dataset construction."""
    arguments = {
        "viewport_num_iq_samples": 4,
        "viewport_sample_rate": 4,
        "signal_generators": [],
        "validate_init": False,
        "metadata": {
            "sample_rate": 8,
            "num_iq_samples_dataset": 16,
            "frequency_min": -4,
            "frequency_max": 4,
            "signal_center_freq_min": -3,
            "signal_center_freq_max": 3,
        },
    }
    arguments.update(overrides)
    with pytest.raises(ValueError, match=message):
        _FixedCanvasDataset(**arguments)
