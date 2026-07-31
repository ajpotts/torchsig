"""Tests for the experimental IQ scene-cropping transform."""

from __future__ import annotations

import numpy as np
import pytest

from torchsig.signals.signal_types import Signal
from torchsig.transforms.metadata_transforms import YOLOLabel
from torchsig.transforms.prototype_scene_crop import PrototypeSceneCrop


def _tone(
    frequency: float,
    sample_rate: float,
    num_samples: int,
) -> np.ndarray:
    """Return a unit-amplitude complex64 tone."""
    samples = np.arange(num_samples)
    return np.exp(2j * np.pi * frequency * samples / sample_rate).astype(
        np.complex64
    )


def _component(
    *,
    start: int,
    duration: int,
    center_freq: float,
    bandwidth: float,
    sample_rate: float = 8.0,
    class_name: str = "tone",
) -> Signal:
    """Create a deterministic placed component."""
    return Signal(
        data=_tone(center_freq, sample_rate, duration),
        start_in_samples=start,
        duration_in_samples=duration,
        center_freq=center_freq,
        bandwidth=bandwidth,
        class_name=class_name,
        class_index=1,
    )


def _canvas(
    *,
    num_samples: int = 16,
    sample_rate: float = 8.0,
    components: list[Signal] | None = None,
) -> Signal:
    """Create a canvas with component IQ inserted at its time positions."""
    components = [] if components is None else components
    data = np.zeros(num_samples, dtype=np.complex64)
    for component in components:
        start = int(component.start_in_samples)
        stop = min(num_samples, start + len(component.data))
        data[start:stop] += component.data[: stop - start]
    return Signal(
        data=data,
        component_signals=components,
        sample_rate=sample_rate,
        num_iq_samples_dataset=num_samples,
        frequency_min=-sample_rate / 2,
        frequency_max=sample_rate / 2,
        center_freq=0.0,
        bandwidth=sample_rate,
    )


def test_time_crop_retains_and_truncates_components() -> None:
    """Time cropping should express every retained burst in output samples."""
    canvas = _canvas(
        components=[
            _component(start=5, duration=2, center_freq=0, bandwidth=1),
            _component(start=2, duration=4, center_freq=0, bandwidth=1),
            _component(start=10, duration=4, center_freq=0, bandwidth=1),
            _component(start=12, duration=2, center_freq=0, bandwidth=1),
        ]
    )
    transform = PrototypeSceneCrop(
        num_iq_samples=8,
        sample_rate=8,
        time_start=4,
        center_freq=0,
    )

    output = transform(canvas)

    assert output.data.shape == (8,)
    assert output.data.dtype == np.complex64
    assert len(output.component_signals) == 3
    assert [component.start_in_samples for component in output.component_signals] == [
        1,
        0,
        6,
    ]
    assert [
        component.duration_in_samples for component in output.component_signals
    ] == [2, 2, 2]
    assert [len(component.data) for component in output.component_signals] == [
        2,
        2,
        2,
    ]


def test_frequency_crop_updates_visible_component_intervals() -> None:
    """Frequency cropping should retain full, partial, and middle intervals."""
    canvas = _canvas(
        components=[
            _component(start=0, duration=16, center_freq=2, bandwidth=1),
            _component(start=0, duration=16, center_freq=0, bandwidth=2),
            _component(start=0, duration=16, center_freq=2, bandwidth=6),
            _component(start=0, duration=16, center_freq=-2, bandwidth=1),
        ]
    )
    transform = PrototypeSceneCrop(
        num_iq_samples=8,
        sample_rate=4,
        time_start=0,
        center_freq=2,
    )

    output = transform(canvas)

    assert len(output.component_signals) == 3
    full, lower_partial, middle = output.component_signals
    assert (full.lower_freq, full.upper_freq) == pytest.approx((-0.5, 0.5))
    assert (lower_partial.lower_freq, lower_partial.upper_freq) == pytest.approx(
        (-2.0, -1.0)
    )
    assert (middle.lower_freq, middle.upper_freq) == pytest.approx((-2.0, 2.0))
    assert output.sample_rate == pytest.approx(4.0)
    assert output.frequency_min == pytest.approx(-2.0)
    assert output.frequency_max == pytest.approx(2.0)


def test_crop_reparents_components_to_viewport_metadata() -> None:
    """Retained components should inherit output rather than canvas metadata."""
    component = _component(
        start=0,
        duration=16,
        center_freq=2,
        bandwidth=1,
    )
    canvas = _canvas(components=[component])
    transform = PrototypeSceneCrop(
        num_iq_samples=8,
        sample_rate=4,
        time_start=0,
        center_freq=2,
    )

    output = transform(canvas)
    cropped = output.component_signals[0]

    assert cropped.parent is output
    assert cropped.sample_rate == pytest.approx(4.0)
    assert cropped.num_iq_samples_dataset == 8
    assert cropped.frequency_min == pytest.approx(-2.0)
    assert cropped.frequency_max == pytest.approx(2.0)


def test_crop_does_not_mutate_input_scene() -> None:
    """One canvas should remain reusable for multiple viewport extractions."""
    component = _component(
        start=2,
        duration=8,
        center_freq=1,
        bandwidth=2,
    )
    canvas = _canvas(components=[component])
    original_data = canvas.data.copy()
    original_component_data = component.data.copy()
    original_metadata = component.metadata

    PrototypeSceneCrop(
        num_iq_samples=4,
        sample_rate=4,
        time_start=2,
        center_freq=1,
    )(canvas)

    np.testing.assert_array_equal(canvas.data, original_data)
    np.testing.assert_array_equal(component.data, original_component_data)
    assert component.metadata == original_metadata
    assert canvas.component_signals == [component]


def test_frequency_crop_moves_selected_tone_to_baseband() -> None:
    """A tone at the viewport center should appear at output DC."""
    component = _component(
        start=0,
        duration=128,
        center_freq=2,
        bandwidth=0.5,
    )
    canvas = _canvas(num_samples=128, components=[component])
    transform = PrototypeSceneCrop(
        num_iq_samples=64,
        sample_rate=4,
        time_start=0,
        center_freq=2,
    )

    output = transform(canvas)

    spectrum = np.abs(np.fft.fftshift(np.fft.fft(output.data)))
    peak_offset = int(np.argmax(spectrum)) - len(spectrum) // 2
    assert abs(peak_offset) <= 1
    assert output.data.dtype == np.complex64
    np.testing.assert_allclose(
        output.data,
        output.component_signals[0].data,
        rtol=1e-6,
        atol=1e-6,
    )


def test_frequency_crop_attenuates_tone_outside_viewport() -> None:
    """Polyphase subband extraction should prevent excluded-tone aliasing."""
    canvas = _canvas(num_samples=512)
    canvas.data = _tone(-2, 8, 512)
    transform = PrototypeSceneCrop(
        num_iq_samples=256,
        sample_rate=4,
        time_start=0,
        center_freq=2,
    )

    output = transform(canvas)

    assert np.sqrt(np.mean(np.abs(output.data) ** 2)) < 0.1


def test_yolo_label_uses_cropped_viewport_metadata() -> None:
    """Downstream labels should use visible viewport-relative metadata."""
    component = _component(
        start=2,
        duration=8,
        center_freq=2,
        bandwidth=2,
    )
    canvas = _canvas(components=[component])
    output = PrototypeSceneCrop(
        num_iq_samples=4,
        sample_rate=4,
        time_start=2,
        center_freq=2,
    )(canvas)

    YOLOLabel()(output)

    cropped = output.component_signals[0]
    assert cropped.yolo_label == pytest.approx((1, 0.5, 0.5, 1.0, 0.5))


def test_seeded_random_crop_is_reproducible() -> None:
    """Seeded transforms should select the same random viewport."""
    canvas = _canvas(num_samples=32)
    first = PrototypeSceneCrop(num_iq_samples=8, sample_rate=4, seed=123)
    second = PrototypeSceneCrop(num_iq_samples=8, sample_rate=4, seed=123)

    first_output = first(canvas)
    second_output = second(canvas)

    assert first_output.scene_crop_input_time_start == (
        second_output.scene_crop_input_time_start
    )
    assert first_output.scene_crop_input_center_freq == pytest.approx(
        second_output.scene_crop_input_center_freq
    )
    np.testing.assert_array_equal(first_output.data, second_output.data)


def test_empty_crop_policy() -> None:
    """Empty viewports should be accepted or rejected by configuration."""
    canvas = _canvas()

    output = PrototypeSceneCrop(
        num_iq_samples=8,
        sample_rate=8,
        time_start=0,
        center_freq=0,
        allow_empty=True,
    )(canvas)
    assert output.component_signals == []

    with pytest.raises(ValueError, match="no visible components"):
        PrototypeSceneCrop(
            num_iq_samples=8,
            sample_rate=8,
            time_start=0,
            center_freq=0,
            allow_empty=False,
        )(canvas)


def test_crop_accepts_torchsig_positive_nyquist_endpoint_convention() -> None:
    """A canvas ending at Fs/2 - 1 Hz should represent the full input band."""
    canvas = _canvas()
    canvas["frequency_max"] = 3.0

    output = PrototypeSceneCrop(
        num_iq_samples=8,
        sample_rate=8,
        time_start=0,
        center_freq=0,
    )(canvas)

    assert output.data.shape == (8,)
    assert output.bandwidth == pytest.approx(8.0)


@pytest.mark.parametrize(
    ("kwargs", "exception_type", "match"),
    [
        ({"num_iq_samples": 0, "sample_rate": 4}, ValueError, "positive integer"),
        ({"num_iq_samples": 8, "sample_rate": 0}, ValueError, "positive finite"),
        (
            {"num_iq_samples": 8, "sample_rate": 4, "time_start": -1},
            ValueError,
            "non-negative",
        ),
        (
            {"num_iq_samples": 8, "sample_rate": 4, "center_freq": np.nan},
            ValueError,
            "finite number",
        ),
        (
            {"num_iq_samples": 8, "sample_rate": 4, "allow_empty": 1},
            TypeError,
            "boolean",
        ),
    ],
)
def test_constructor_validation(kwargs, exception_type, match) -> None:
    """Invalid constructor arguments should fail clearly."""
    with pytest.raises(exception_type, match=match):
        PrototypeSceneCrop(**kwargs)


def test_canvas_validation() -> None:
    """Unsupported data and viewport geometry should be rejected."""
    with pytest.raises(TypeError, match="complex IQ"):
        PrototypeSceneCrop(8, 4)(
            Signal(data=np.ones(16), sample_rate=8),
        )

    with pytest.raises(ValueError, match="integer multiple"):
        PrototypeSceneCrop(8, 3)(_canvas())

    with pytest.raises(ValueError, match="too short"):
        PrototypeSceneCrop(9, 4)(_canvas())

    with pytest.raises(ValueError, match="time viewport"):
        PrototypeSceneCrop(8, 8, time_start=9, center_freq=0)(_canvas())

    with pytest.raises(ValueError, match="frequency viewport"):
        PrototypeSceneCrop(8, 4, time_start=0, center_freq=3)(_canvas())
