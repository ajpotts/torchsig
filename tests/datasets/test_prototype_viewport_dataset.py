"""Tests for generation-time viewport construction."""

from __future__ import annotations

import numpy as np
import pytest

from torchsig.datasets.prototype_viewport_dataset import PrototypeViewportDataset
from torchsig.signals.signal_types import Signal
from torchsig.transforms.metadata_transforms import YOLOLabel
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from torchsig.utils.writer import identity_collate_fn


def _tone(frequency: float, length: int) -> np.ndarray:
    samples = np.arange(length)
    return np.exp(2j * np.pi * frequency * samples / 8).astype(np.complex64)


class FixedGenerationDataset(PrototypeViewportDataset):
    """Use fixed noise and components while retaining generation lifecycle."""

    def _build_noise_floor(self) -> np.ndarray:
        """Return deterministic zero-valued canvas noise."""
        return np.zeros(16, dtype=np.complex64)

    def _generate_placed_components(self) -> list[Signal]:
        """Return a component already positioned on the expanded canvas."""
        return [
            Signal(
                data=_tone(2, 8),
                class_name="tone",
                class_index=1,
                start_in_samples=2,
                duration_in_samples=8,
                center_freq=2,
                bandwidth=2,
            )
        ]


def dataset(**kwargs) -> FixedGenerationDataset:
    """Construct a fixed generation-time viewport dataset."""
    return FixedGenerationDataset(
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
        },
        **kwargs,
    )


def test_component_is_cropped_during_generation() -> None:
    """Returned component IQ and metadata should have viewport geometry."""
    output = next(dataset())

    assert output.data.shape == (4,)
    assert output.data.dtype == np.complex64
    assert output.sample_rate == 4
    assert len(output.component_signals) == 1
    component = output.component_signals[0]
    assert component.data.shape == (4,)
    assert component.start_in_samples == 0
    assert component.duration_in_samples == 4
    assert component.center_freq == pytest.approx(0)


def test_debug_canvas_is_optional() -> None:
    """Expanded aggregate state should only be built when explicitly requested."""
    ordinary = dataset()
    next(ordinary)
    debugging = dataset(retain_canvas=True)
    next(debugging)

    assert ordinary.last_canvas is None
    assert debugging.last_canvas is not None
    assert debugging.last_canvas.data.shape == (16,)


def test_transforms_and_labels_follow_generation_crop() -> None:
    """YOLO extraction should consume the constructed viewport metadata."""
    data, target = next(
        dataset(transforms=[YOLOLabel()], target_labels=["yolo_label"])
    )

    assert data.shape == (4,)
    assert target == pytest.approx((1, 0.5, 0.5, 1.0, 0.5))


def test_pytorch_iterator_returns_constructed_viewports() -> None:
    """TorchSIG's worker-seeding loader should yield output-sized samples."""
    loader = WorkerSeedingDataLoader(
        dataset(),
        seed=42,
        batch_size=2,
        num_workers=0,
        collate_fn=identity_collate_fn,
    )

    batch = next(iter(loader))

    assert len(batch) == 2
    assert all(signal.data.shape == (4,) for signal in batch)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"viewport_num_iq_samples": 0}, "must be positive"),
        ({"viewport_sample_rate": 3}, "integer ratio"),
    ],
)
def test_invalid_geometry(overrides: dict, message: str) -> None:
    """Invalid viewport geometry should fail during construction."""
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
        },
    }
    arguments.update(overrides)
    with pytest.raises(ValueError, match=message):
        FixedGenerationDataset(**arguments)
