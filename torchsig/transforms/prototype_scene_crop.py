"""Experimental time-frequency scene-cropping transform.

This module is intentionally not exported from :mod:`torchsig.transforms`.
It is a prototype for evaluating scene cropping as an IQ-level transform and
does not define a stable public API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import signal as sp

from torchsig.signals.signal_types import Signal
from torchsig.transforms.transforms import SignalTransform
from torchsig.utils.dsp import TorchSigComplexDataType, frequency_shift

_SHARED_VIEWPORT_KEYS = {
    "frequency_max",
    "frequency_min",
    "num_iq_samples_dataset",
    "sample_rate",
    "signal_center_freq_max",
    "signal_center_freq_min",
}


@dataclass(frozen=True)
class _Interval:
    """Half-open interval used by the prototype intersection helpers."""

    start: float
    stop: float

    @property
    def length(self) -> float:
        """Return the non-negative interval length."""
        return max(0.0, self.stop - self.start)

    def intersect(self, other: _Interval) -> _Interval | None:
        """Return the positive intersection with another interval."""
        start = max(self.start, other.start)
        stop = min(self.stop, other.stop)
        return None if start >= stop else _Interval(start, stop)


class PrototypeSceneCrop(SignalTransform):
    """Extract an experimental time-frequency viewport from complex IQ.

    The input is an expanded time-frequency canvas. The transform selects a
    time interval in input samples and a frequency interval of width
    ``sample_rate`` in the input's signed-Hz coordinate system. It mixes the
    selected subband to DC, applies polyphase anti-alias filtering and integer
    decimation, intersects component metadata with the viewport, and returns a
    new :class:`~torchsig.signals.signal_types.Signal`.

    This prototype supports only integer input/output sample-rate ratios. It is
    not exported as a production TorchSIG transform.

    Args:
        num_iq_samples: Number of IQ samples in the returned viewport.
        sample_rate: Sample rate in Hz of the returned viewport.
        time_start: Optional viewport start in input samples. ``None`` selects
            a seeded random valid start.
        center_freq: Optional viewport center in the input frequency
            coordinate. ``None`` selects a seeded random valid center.
        allow_empty: Whether a viewport with no visible components is valid.
        **kwargs: Additional arguments for :class:`SignalTransform`, including
            ``seed`` and ``parent``.

    Raises:
        TypeError: If constructor values have invalid types.
        ValueError: If constructor values are outside their valid ranges.
    """

    def __init__(
        self,
        num_iq_samples: int,
        sample_rate: float,
        *,
        time_start: int | None = None,
        center_freq: float | None = None,
        allow_empty: bool = True,
        **kwargs: Any,
    ) -> None:
        if (
            not isinstance(num_iq_samples, int)
            or isinstance(num_iq_samples, bool)
            or num_iq_samples < 1
        ):
            raise ValueError("num_iq_samples must be a positive integer")
        if (
            not isinstance(sample_rate, (int, float, np.integer, np.floating))
            or isinstance(sample_rate, (bool, np.bool_))
            or not np.isfinite(sample_rate)
            or sample_rate <= 0
        ):
            raise ValueError("sample_rate must be a positive finite number")
        if time_start is not None and (
            not isinstance(time_start, int)
            or isinstance(time_start, bool)
            or time_start < 0
        ):
            raise ValueError("time_start must be a non-negative integer or None")
        if center_freq is not None and (
            not isinstance(center_freq, (int, float, np.integer, np.floating))
            or isinstance(center_freq, (bool, np.bool_))
            or not np.isfinite(center_freq)
        ):
            raise ValueError("center_freq must be a finite number or None")
        if not isinstance(allow_empty, bool):
            raise TypeError("allow_empty must be a boolean")

        self.num_iq_samples = num_iq_samples
        self.output_sample_rate = float(sample_rate)
        self.time_start = time_start
        self.viewport_center_freq = (
            None if center_freq is None else float(center_freq)
        )
        self.allow_empty = allow_empty
        super().__init__(
            required_metadata=["sample_rate"],
            data_dtype=TorchSigComplexDataType,
            **kwargs,
        )

    def __validate__(self, signal: Signal) -> Signal:
        """Validate the canvas and the requested output viewport."""
        signal = super().__validate__(signal)
        if signal.data.ndim != 1 or not np.iscomplexobj(signal.data):
            raise TypeError("PrototypeSceneCrop requires one-dimensional complex IQ")

        input_sample_rate = float(signal.sample_rate)
        ratio = input_sample_rate / self.output_sample_rate
        decimation = round(ratio)
        if decimation < 1 or not np.isclose(ratio, decimation):
            raise ValueError(
                "input sample_rate must be an integer multiple of output sample_rate"
            )

        required_input_samples = self.num_iq_samples * decimation
        if len(signal.data) < required_input_samples:
            raise ValueError("input canvas is too short for the requested viewport")
        if self.time_start is not None and (
            self.time_start + required_input_samples > len(signal.data)
        ):
            raise ValueError("time viewport extends beyond the input canvas")

        frequency_bounds = self._input_frequency_bounds(signal, input_sample_rate)
        if frequency_bounds.length < self.output_sample_rate:
            raise ValueError("input frequency canvas is narrower than the viewport")
        if self.viewport_center_freq is not None:
            viewport = self._frequency_viewport(self.viewport_center_freq)
            if (
                viewport.start < frequency_bounds.start
                or viewport.stop > frequency_bounds.stop
            ):
                raise ValueError("frequency viewport extends beyond the input canvas")
        return signal

    def __apply__(self, signal: Signal) -> Signal:
        """Return a new signal containing one time-frequency viewport."""
        input_sample_rate = float(signal.sample_rate)
        decimation = round(input_sample_rate / self.output_sample_rate)
        required_input_samples = self.num_iq_samples * decimation
        max_time_start = len(signal.data) - required_input_samples
        time_start = (
            self.time_start
            if self.time_start is not None
            else int(self.random_generator.integers(0, max_time_start + 1))
        )
        time_viewport = _Interval(
            float(time_start),
            float(time_start + required_input_samples),
        )

        input_frequency_bounds = self._input_frequency_bounds(
            signal,
            input_sample_rate,
        )
        min_center = input_frequency_bounds.start + self.output_sample_rate / 2
        max_center = input_frequency_bounds.stop - self.output_sample_rate / 2
        viewport_center = (
            self.viewport_center_freq
            if self.viewport_center_freq is not None
            else float(self.random_generator.uniform(min_center, max_center))
        )
        frequency_viewport = self._frequency_viewport(viewport_center)

        input_data = signal.data[time_start : time_start + required_input_samples]
        output_data = self._extract_subband(
            input_data,
            viewport_center,
            input_sample_rate,
            decimation,
        )
        output_data = self._fit_output_length(output_data)

        output = Signal(
            data=output_data,
            component_signals=[],
            parent=signal.parent,
            metadata=signal.get_full_metadata(),
        )
        self._set_output_metadata(
            output,
            time_start=time_start,
            viewport_center=viewport_center,
            decimation=decimation,
        )

        for component in signal.component_signals:
            cropped_component = self._crop_component(
                component,
                output,
                time_viewport=time_viewport,
                frequency_viewport=frequency_viewport,
                viewport_center=viewport_center,
                input_sample_rate=input_sample_rate,
                decimation=decimation,
            )
            if cropped_component is not None:
                output.component_signals.append(cropped_component)

        if not self.allow_empty and not output.component_signals:
            raise ValueError("selected viewport contains no visible components")
        return output

    def _crop_component(
        self,
        component: Signal,
        output: Signal,
        *,
        time_viewport: _Interval,
        frequency_viewport: _Interval,
        viewport_center: float,
        input_sample_rate: float,
        decimation: int,
    ) -> Signal | None:
        """Return a viewport-relative component or ``None`` when invisible."""
        component_start = float(component.start_in_samples)
        component_duration = float(component.duration_in_samples)
        component_time = _Interval(
            component_start,
            component_start + component_duration,
        )
        visible_time = component_time.intersect(time_viewport)
        component_center = float(component.center_freq)
        component_bandwidth = float(component.bandwidth)
        component_frequency = _Interval(
            component_center - component_bandwidth / 2,
            component_center + component_bandwidth / 2,
        )
        visible_frequency = component_frequency.intersect(frequency_viewport)
        if visible_time is None or visible_frequency is None:
            return None

        local_start = max(0, int(np.floor(visible_time.start - component_start)))
        local_stop = min(
            len(component.data),
            int(np.ceil(visible_time.stop - component_start)),
        )
        visible_data = component.data[local_start:local_stop]
        phase_offset = visible_time.start - time_viewport.start
        cropped_data = self._extract_subband(
            visible_data,
            viewport_center,
            input_sample_rate,
            decimation,
            sample_offset=phase_offset,
        )

        output_start = max(
            0,
            int(np.floor((visible_time.start - time_viewport.start) / decimation)),
        )
        output_stop = min(
            self.num_iq_samples,
            int(np.ceil((visible_time.stop - time_viewport.start) / decimation)),
        )
        output_duration = max(0, output_stop - output_start)
        cropped_data = self._fit_length(cropped_data, output_duration)

        local_metadata = component.metadata
        for key in _SHARED_VIEWPORT_KEYS:
            local_metadata.pop(key, None)
        local_metadata.pop("_lower_frequency", None)
        local_metadata.pop("_upper_frequency", None)
        cropped = Signal(
            data=cropped_data,
            component_signals=[],
            metadata=local_metadata,
        )
        cropped.add_parent(output, register=False)
        cropped["start_in_samples"] = output_start
        cropped["duration_in_samples"] = output_duration
        cropped["center_freq"] = (
            (visible_frequency.start + visible_frequency.stop) / 2
            - viewport_center
        )
        cropped["bandwidth"] = visible_frequency.length
        return cropped

    def _extract_subband(
        self,
        data: np.ndarray,
        viewport_center: float,
        input_sample_rate: float,
        decimation: int,
        *,
        sample_offset: float = 0.0,
    ) -> np.ndarray:
        """Mix one input sequence to the viewport center and decimate it."""
        if data.size == 0:
            return np.empty(0, dtype=TorchSigComplexDataType)
        mixed = frequency_shift(data, -viewport_center, input_sample_rate)
        if sample_offset:
            phase = np.exp(
                -2j
                * np.pi
                * (viewport_center / input_sample_rate)
                * sample_offset
            )
            mixed = mixed * phase
        if decimation > 1:
            mixed = sp.resample_poly(mixed, up=1, down=decimation)
        return np.asarray(mixed, dtype=TorchSigComplexDataType)

    def _fit_output_length(self, data: np.ndarray) -> np.ndarray:
        """Fit parent IQ to the exact configured output length."""
        return self._fit_length(data, self.num_iq_samples)

    @staticmethod
    def _fit_length(data: np.ndarray, length: int) -> np.ndarray:
        """Trim or zero-pad IQ to an exact length."""
        if len(data) >= length:
            return np.asarray(data[:length], dtype=TorchSigComplexDataType)
        return np.pad(data, (0, length - len(data))).astype(
            TorchSigComplexDataType,
            copy=False,
        )

    def _frequency_viewport(self, center_freq: float) -> _Interval:
        """Return output-band edges in input frequency coordinates."""
        half_width = self.output_sample_rate / 2
        return _Interval(center_freq - half_width, center_freq + half_width)

    @staticmethod
    def _input_frequency_bounds(
        signal: Signal,
        input_sample_rate: float,
    ) -> _Interval:
        """Return configured canvas bounds or the input Nyquist interval."""
        lower = float(
            getattr(signal, "frequency_min", -input_sample_rate / 2)
        )
        upper = float(
            getattr(signal, "frequency_max", input_sample_rate / 2)
        )
        half_rate = input_sample_rate / 2
        endpoint_tolerance = max(1.0, input_sample_rate / len(signal.data))
        if (
            abs(lower + half_rate) <= endpoint_tolerance
            and abs(upper - half_rate) <= endpoint_tolerance
        ):
            return _Interval(-half_rate, half_rate)
        return _Interval(lower, upper)

    def _set_output_metadata(
        self,
        output: Signal,
        *,
        time_start: int,
        viewport_center: float,
        decimation: int,
    ) -> None:
        """Set viewport metadata on the returned parent signal."""
        half_rate = self.output_sample_rate / 2
        output["sample_rate"] = self.output_sample_rate
        output["num_iq_samples_dataset"] = self.num_iq_samples
        output["frequency_min"] = -half_rate
        output["frequency_max"] = half_rate
        output["signal_center_freq_min"] = -half_rate
        output["signal_center_freq_max"] = half_rate
        output["center_freq"] = 0.0
        output["bandwidth"] = self.output_sample_rate
        output["duration_in_samples"] = self.num_iq_samples
        output["start_in_samples"] = 0
        output["scene_crop_input_time_start"] = time_start
        output["scene_crop_input_center_freq"] = viewport_center
        output["scene_crop_decimation"] = decimation


__all__ = ["PrototypeSceneCrop"]
