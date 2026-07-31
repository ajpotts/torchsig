"""Experimental generation-time time-frequency viewport dataset."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import resample_poly

from torchsig.datasets.datasets import TorchSigIterableDataset
from torchsig.signals.signal_types import Signal
from torchsig.utils.dsp import frequency_shift


class PrototypeViewportDataset(TorchSigIterableDataset):
    """Crop expanded-canvas components while constructing each dataset item.

    Canvas coordinates are chosen before component generation. Each generated
    component is positioned on that canvas and its visible portion is directly
    downconverted, decimated, and inserted into the output. No aggregate scene
    crop or crop transform is used.

    Args:
        viewport_num_iq_samples: Number of returned IQ samples.
        viewport_sample_rate: Returned sample rate in Hz.
        viewport_time_start: Optional fixed start in canvas samples.
        viewport_center_freq: Optional fixed center in canvas-relative Hz.
        allow_empty: Whether a viewport without components is permitted.
        retain_canvas: Retain the expanded scene as ``last_canvas`` for debug.
        **kwargs: Arguments for ``TorchSigIterableDataset``. Metadata describes
            the expanded generation canvas.
    """

    def __init__(
        self,
        viewport_num_iq_samples: int,
        viewport_sample_rate: float,
        *,
        viewport_time_start: int | None = None,
        viewport_center_freq: float | None = None,
        allow_empty: bool = True,
        retain_canvas: bool = False,
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
        self.retain_canvas = retain_canvas
        self.decimation = int(ratio)
        self.last_canvas: Signal | None = None

    def __generate_new_signal__(self) -> Signal:
        """Generate and crop components directly into an output viewport."""
        time_start, center_freq = self._choose_viewport()
        canvas_noise = self._build_noise_floor()
        output_data = self._extract_iq(canvas_noise, time_start, center_freq)
        debug_data = canvas_noise.copy() if self.retain_canvas else None

        placed_components = self._generate_placed_components()
        visible_components = []
        for component in placed_components:
            if debug_data is not None:
                start = int(component.start_in_samples)
                debug_data[start : start + len(component.data)] += component.data
            visible = self._crop_component(component, time_start, center_freq)
            if visible is None:
                continue
            start = int(visible.start_in_samples)
            stop = min(start + len(visible.data), len(output_data))
            output_data[start:stop] += visible.data[: stop - start]
            visible_components.append(visible)

        if not visible_components and not self.allow_empty:
            raise ValueError("selected viewport contains no component signals")

        half_bandwidth = self.viewport_sample_rate / 2
        output = Signal(
            data=output_data,
            component_signals=visible_components,
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
        for component in visible_components:
            component.add_parent(output, register=False)
        self.last_canvas = (
            self._debug_canvas(debug_data, placed_components)
            if debug_data is not None
            else None
        )
        return output

    def _choose_viewport(self) -> tuple[int, float]:
        """Choose valid viewport coordinates before generating components."""
        input_span = self.viewport_num_iq_samples * self.decimation
        max_start = int(self["num_iq_samples_dataset"]) - input_span
        if max_start < 0:
            raise ValueError("viewport time span exceeds the generation canvas")
        time_start = (
            int(self.random_generator.integers(0, max_start + 1))
            if self.viewport_time_start is None
            else int(self.viewport_time_start)
        )
        if not 0 <= time_start <= max_start:
            raise ValueError("viewport_time_start is outside the generation canvas")

        half_bandwidth = self.viewport_sample_rate / 2
        center_min = float(self["frequency_min"]) + half_bandwidth
        center_max = float(self["frequency_max"]) - half_bandwidth
        center_freq = (
            float(self.random_generator.uniform(center_min, center_max))
            if self.viewport_center_freq is None
            else float(self.viewport_center_freq)
        )
        if not center_min <= center_freq <= center_max:
            raise ValueError("viewport_center_freq is outside the generation canvas")
        return time_start, center_freq

    def _extract_iq(
        self, data: np.ndarray, time_start: int, center_freq: float
    ) -> np.ndarray:
        """Extract viewport noise from canvas-rate noise."""
        input_span = self.viewport_num_iq_samples * self.decimation
        selected = data[time_start : time_start + input_span]
        shifted = frequency_shift(selected, -center_freq, float(self["sample_rate"]))
        return resample_poly(shifted, up=1, down=self.decimation).astype(
            np.complex64, copy=False
        )[: self.viewport_num_iq_samples]

    def _generate_placed_components(self) -> list[Signal]:
        """Generate components and assign expanded-canvas coordinates."""
        placed = []
        rectangles = []
        canvas_length = int(self["num_iq_samples_dataset"])
        placement_buffer = np.empty(canvas_length, dtype=np.complex64)
        target_count = int(
            self.random_generator.integers(
                low=self["num_signals_min"], high=self["num_signals_max"] + 1
            )
        )
        for _ in range(10 * target_count):
            if len(placed) >= target_count:
                break
            component = self._generate_component_signal()
            start = self._choose_start_sample(placement_buffer, component)
            self._truncate_component_signal(placement_buffer, component, start)
            rectangle = self._map_to_coordinates(component, start)
            if self._check_if_overlap(rectangle, rectangles) and not (
                self._allow_cochannel_overlap()
            ):
                continue
            component["start_in_samples"] = start
            component["duration_in_samples"] = len(component.data)
            rectangles.append(rectangle)
            placed.append(component)
        return placed

    def _crop_component(
        self, component: Signal, time_start: int, center_freq: float
    ) -> Signal | None:
        """Crop one component during generation into viewport coordinates."""
        input_span = self.viewport_num_iq_samples * self.decimation
        time_lower = max(component.start_in_samples, time_start)
        time_upper = min(component.stop_in_samples, time_start + input_span)
        viewport_lower = center_freq - self.viewport_sample_rate / 2
        viewport_upper = center_freq + self.viewport_sample_rate / 2
        component_lower = component.center_freq - component.bandwidth / 2
        component_upper = component.center_freq + component.bandwidth / 2
        frequency_lower = max(component_lower, viewport_lower)
        frequency_upper = min(component_upper, viewport_upper)
        if time_lower >= time_upper or frequency_lower >= frequency_upper:
            return None

        offset = int(time_lower - component.start_in_samples)
        length = int(time_upper - time_lower)
        data = component.data[offset : offset + length]
        data = resample_poly(
            frequency_shift(data, -center_freq, float(self["sample_rate"])),
            up=1,
            down=self.decimation,
        ).astype(np.complex64, copy=False)
        metadata = component.to_dict()
        metadata.pop("_lower_frequency", None)
        metadata.pop("_upper_frequency", None)
        metadata.update(
            start_in_samples=int((time_lower - time_start) / self.decimation),
            duration_in_samples=len(data),
            center_freq=(frequency_lower + frequency_upper) / 2 - center_freq,
            bandwidth=frequency_upper - frequency_lower,
        )
        return Signal(data=data, **metadata)

    def _debug_canvas(
        self, data: np.ndarray, components: list[Signal]
    ) -> Signal:
        """Construct optional expanded state for visual inspection."""
        canvas = Signal(
            data=data,
            component_signals=[component.copy() for component in components],
            sample_rate=self["sample_rate"],
            num_iq_samples_dataset=self["num_iq_samples_dataset"],
            frequency_min=self["frequency_min"],
            frequency_max=self["frequency_max"],
            center_freq=0.0,
            bandwidth=self["sample_rate"],
        )
        canvas.add_parent(self, register=False)
        return canvas


__all__ = ["PrototypeViewportDataset"]
