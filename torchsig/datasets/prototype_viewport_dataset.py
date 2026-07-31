"""Experimental generator-integrated time-frequency viewport dataset."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import resample_poly

from torchsig.datasets.dataset_utils import frequency_shift_signal
from torchsig.datasets.datasets import (
    TorchSigIterableDataset,
    apply_transforms_and_labels_to_signal,
)
from torchsig.signals.signal_types import Signal
from torchsig.utils.dsp import frequency_shift, update_signal_snr_bandwidth


class PrototypeViewportDataset(TorchSigIterableDataset):
    """Generate an expanded scene and return only a receiver viewport.

    This prototype represents a generator-integrated alternative to applying a
    crop in a user transform pipeline. The dataset's metadata describes the
    expanded generation canvas. ``viewport_*`` arguments describe the returned
    sample. Dataset transforms and target extraction run after selection.

    Args:
        viewport_num_iq_samples: Number of IQ samples returned per item.
        viewport_sample_rate: Sample rate of returned IQ samples in Hz.
        viewport_time_start: Optional fixed viewport start on the canvas, in
            input samples. If omitted, each item selects a random valid start.
        viewport_center_freq: Optional fixed viewport center in canvas-relative
            Hz. If omitted, each item selects a random valid center frequency.
        allow_empty: Whether viewports containing no component signals are
            allowed when ``num_signals_min`` is zero. Generated components are
            otherwise placed so they overlap the viewport.
        **kwargs: Arguments passed to :class:`TorchSigIterableDataset`. Its
            metadata must describe the expanded canvas.
    """

    def __init__(
        self,
        viewport_num_iq_samples: int,
        viewport_sample_rate: float,
        *,
        viewport_time_start: int | None = None,
        viewport_center_freq: float | None = None,
        allow_empty: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if viewport_num_iq_samples <= 0:
            raise ValueError("viewport_num_iq_samples must be positive")
        if viewport_sample_rate <= 0:
            raise ValueError("viewport_sample_rate must be positive")
        ratio = float(self["sample_rate"]) / viewport_sample_rate
        if not ratio.is_integer():
            raise ValueError(
                "canvas and viewport sample rates must have an integer ratio"
            )
        self.viewport_num_iq_samples = int(viewport_num_iq_samples)
        self.viewport_sample_rate = float(viewport_sample_rate)
        self.viewport_time_start = viewport_time_start
        self.viewport_center_freq = viewport_center_freq
        self.allow_empty = allow_empty
        self.decimation = int(ratio)
        self.last_canvas: Signal | None = None
        self._active_time_start: int | None = None
        self._active_center_freq: float | None = None

    def __next__(self) -> Signal | np.ndarray | tuple:
        """Generate a canvas, select its viewport, then apply output transforms."""
        time_start, center_freq = self._choose_viewport_coordinates()
        self._active_time_start = time_start
        self._active_center_freq = center_freq
        canvas = self.__generate_new_signal__()
        self.last_canvas = canvas
        sample = self._extract_viewport(canvas, time_start, center_freq)
        return apply_transforms_and_labels_to_signal(
            sample,
            self.transforms,
            self.target_labels,
        )

    def _choose_viewport_coordinates(self) -> tuple[int, float]:
        """Choose the viewport before generating and positioning components."""
        input_span = self.viewport_num_iq_samples * self.decimation
        max_start = int(self["num_iq_samples_dataset"]) - input_span
        if max_start < 0:
            raise ValueError("viewport time span exceeds the generated canvas")
        time_start = (
            int(self.random_generator.integers(0, max_start + 1))
            if self.viewport_time_start is None
            else self.viewport_time_start
        )
        if not 0 <= time_start <= max_start:
            raise ValueError("viewport_time_start is outside the generated canvas")

        half_bandwidth = self.viewport_sample_rate / 2
        center_min = float(self["frequency_min"]) + half_bandwidth
        center_max = float(self["frequency_max"]) - half_bandwidth
        center_freq = (
            float(self.random_generator.uniform(center_min, center_max))
            if self.viewport_center_freq is None
            else float(self.viewport_center_freq)
        )
        if not center_min <= center_freq <= center_max:
            raise ValueError("viewport_center_freq is outside the generated canvas")
        return int(time_start), center_freq

    def _extract_viewport(
        self,
        canvas: Signal,
        time_start: int,
        center_freq: float,
    ) -> Signal:
        """Select, downconvert, and decimate one viewport from a canvas."""
        input_span = self.viewport_num_iq_samples * self.decimation
        half_bandwidth = self.viewport_sample_rate / 2
        selected = canvas.data[time_start : time_start + input_span]
        shifted = frequency_shift(selected, -center_freq, float(self["sample_rate"]))
        data = resample_poly(shifted, up=1, down=self.decimation).astype(
            np.complex64,
            copy=False,
        )[: self.viewport_num_iq_samples]

        components = self._crop_components(canvas, time_start, input_span, center_freq)
        if not components and not self.allow_empty:
            raise ValueError("selected viewport contains no component signals")

        output = Signal(
            data=data,
            component_signals=components,
            sample_rate=self.viewport_sample_rate,
            num_iq_samples_dataset=self.viewport_num_iq_samples,
            frequency_min=-half_bandwidth,
            frequency_max=half_bandwidth,
            signal_center_freq_min=-half_bandwidth,
            signal_center_freq_max=half_bandwidth,
            center_freq=0.0,
            bandwidth=self.viewport_sample_rate,
            start_in_samples=0,
            duration_in_samples=self.viewport_num_iq_samples,
            viewport_input_time_start=time_start,
            viewport_input_center_freq=center_freq,
            viewport_decimation=self.decimation,
        )
        output.add_parent(self, register=False)
        for component in components:
            component.add_parent(output, register=False)
        return output

    def _choose_start_sample(
        self,
        iq_samples: np.ndarray,
        signal: Signal,
    ) -> int:
        """Choose a canvas start that guarantees time overlap with the viewport."""
        if self._active_time_start is None:
            return super()._choose_start_sample(iq_samples, signal)
        viewport_start = self._active_time_start
        viewport_stop = (
            viewport_start + self.viewport_num_iq_samples * self.decimation
        )
        max_canvas_start = max(len(iq_samples) - len(signal.data), 0)
        start_min = max(0, viewport_start - len(signal.data) + 1)
        start_max = min(max_canvas_start, viewport_stop - 1)
        if start_min > start_max:
            raise ValueError("component cannot overlap the viewport in time")
        return int(self.random_generator.integers(start_min, start_max + 1))

    def _frequency_center_bounds(self, signal: Signal) -> tuple[float, float]:
        """Return center-frequency bounds that guarantee viewport intersection."""
        if self._active_center_freq is None:
            return (
                float(self["signal_center_freq_min"]),
                float(self["signal_center_freq_max"]),
            )
        viewport_lower = self._active_center_freq - self.viewport_sample_rate / 2
        viewport_upper = self._active_center_freq + self.viewport_sample_rate / 2
        half_signal_bandwidth = signal.bandwidth / 2
        lower = max(
            float(self["signal_center_freq_min"]),
            float(self["frequency_min"]) + half_signal_bandwidth,
            np.nextafter(viewport_lower - half_signal_bandwidth, np.inf),
        )
        upper = min(
            float(self["signal_center_freq_max"]),
            float(self["frequency_max"]) - half_signal_bandwidth,
            np.nextafter(viewport_upper + half_signal_bandwidth, -np.inf),
        )
        if lower >= upper:
            raise ValueError("component cannot overlap the viewport in frequency")
        return lower, upper

    def _generate_component_signal(self) -> Signal:
        """Generate a component with a center guaranteed to overlap the viewport."""
        generator = self._random_signal_generator()
        signal = generator()
        for component_transform in self.component_transforms:
            signal = component_transform(signal)
        update_signal_snr_bandwidth(self, signal)
        center_min, center_max = self._frequency_center_bounds(signal)
        return frequency_shift_signal(
            signal,
            center_freq_min=center_min,
            center_freq_max=center_max,
            sample_rate=self["sample_rate"],
            frequency_max=self["frequency_max"],
            frequency_min=self["frequency_min"],
            random_generator=self.random_generator,
        )

    def _crop_components(
        self,
        canvas: Signal,
        time_start: int,
        input_span: int,
        center_freq: float,
    ) -> list[Signal]:
        """Intersect component metadata with the selected viewport."""
        components = []
        viewport_lower = center_freq - self.viewport_sample_rate / 2
        viewport_upper = center_freq + self.viewport_sample_rate / 2
        for component in canvas.component_signals:
            time_lower = max(component.start_in_samples, time_start)
            time_upper = min(component.stop_in_samples, time_start + input_span)
            component_lower = component.center_freq - component.bandwidth / 2
            component_upper = component.center_freq + component.bandwidth / 2
            frequency_lower = max(component_lower, viewport_lower)
            frequency_upper = min(component_upper, viewport_upper)
            if time_lower >= time_upper or frequency_lower >= frequency_upper:
                continue

            local_metadata = component.to_dict()
            local_metadata.pop("_lower_frequency", None)
            local_metadata.pop("_upper_frequency", None)
            local_metadata.update(
                {
                    "start_in_samples": int(
                        (time_lower - time_start) / self.decimation
                    ),
                    "duration_in_samples": int(
                        np.ceil((time_upper - time_lower) / self.decimation)
                    ),
                    "center_freq": (
                        (frequency_lower + frequency_upper) / 2 - center_freq
                    ),
                    "bandwidth": frequency_upper - frequency_lower,
                }
            )
            component_offset = int(time_lower - component.start_in_samples)
            component_length = int(time_upper - time_lower)
            component_data = component.data[
                component_offset : component_offset + component_length
            ]
            component_data = resample_poly(
                frequency_shift(
                    component_data,
                    -center_freq,
                    float(self["sample_rate"]),
                ),
                up=1,
                down=self.decimation,
            ).astype(np.complex64, copy=False)
            component_data = component_data[: local_metadata["duration_in_samples"]]
            components.append(Signal(data=component_data, **local_metadata))
        return components


__all__ = ["PrototypeViewportDataset"]
