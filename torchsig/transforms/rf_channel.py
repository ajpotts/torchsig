"""Deterministic, declarative RF/channel augmentation for generated IQ records."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from torchsig.transforms.functional import awgn, carrier_phase_noise, fading
from torchsig.utils.dsp import TorchSigComplexDataType, frequency_shift

if TYPE_CHECKING:
    from torchsig.signals.signal_types import Signal

_RANGE_LENGTH = 2

__all__ = [
    "FadingConfig",
    "RFChannelImpairmentConfig",
    "RFChannelImpairmentPipeline",
    "UniformRange",
]


@dataclass(frozen=True)
class UniformRange:
    """A closed numeric range sampled uniformly during record generation."""

    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        """Validate range bounds."""
        if not np.isfinite(self.minimum) or not np.isfinite(self.maximum):
            raise ValueError("uniform range bounds must be finite")
        if self.minimum > self.maximum:
            raise ValueError("uniform range minimum must not exceed maximum")

    def sample(self, rng: np.random.Generator) -> float:
        """Draw a value, returning the bound directly for a fixed range."""
        if self.minimum == self.maximum:
            return float(self.minimum)
        return float(rng.uniform(self.minimum, self.maximum))


@dataclass(frozen=True)
class FadingConfig:
    """Configuration for TorchSIG's Rayleigh multipath fading model."""

    coherence_bandwidth: UniformRange
    power_delay_profile: tuple[float, ...] = (1.0, 1.0)

    def __post_init__(self) -> None:
        """Validate fading parameters."""
        if self.coherence_bandwidth.minimum <= 0 or self.coherence_bandwidth.maximum > 1:
            raise ValueError("fading coherence bandwidth must be in (0, 1]")
        if len(self.power_delay_profile) < _RANGE_LENGTH:
            raise ValueError("fading power delay profile must contain at least two values")
        if any(not np.isfinite(value) or value < 0 for value in self.power_delay_profile):
            raise ValueError("fading power delay profile values must be finite and nonnegative")
        if not any(value > 0 for value in self.power_delay_profile):
            raise ValueError("fading power delay profile must contain positive power")


@dataclass(frozen=True)
class RFChannelImpairmentConfig:
    """Declarative generation-time RF/channel impairment configuration.

    Frequency offset is in Hz, phase noise is the standard deviation in
    degrees of independent Gaussian phase samples, and fading coherence
    bandwidth is normalized to sample rate. Noise power is absolute dB.
    """

    frequency_offset_hz: UniformRange | None = None
    phase_noise_degrees: UniformRange | None = None
    fading: FadingConfig | None = None
    noise_power_db: UniformRange | None = None
    normalization: Literal["none", "before", "after"] = "none"

    def __post_init__(self) -> None:
        """Validate impairment parameters."""
        if self.phase_noise_degrees is not None and self.phase_noise_degrees.minimum < 0:
            raise ValueError("phase noise must be nonnegative")
        if self.normalization not in {"none", "before", "after"}:
            raise ValueError("normalization must be 'none', 'before', or 'after'")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RFChannelImpairmentConfig:
        """Build and validate a configuration from a JSON/YAML-style mapping."""
        if not isinstance(value, Mapping):
            raise TypeError("RF impairment configuration must be a mapping")
        allowed = {"frequency_offset_hz", "phase_noise_degrees", "fading", "noise_power_db", "normalization"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown RF impairment configuration keys: {sorted(unknown)}")

        def parse_range(name: str) -> UniformRange | None:
            raw = value.get(name)
            if raw is None:
                return None
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                return UniformRange(float(raw), float(raw))
            if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) == _RANGE_LENGTH:
                return UniformRange(float(raw[0]), float(raw[1]))
            raise TypeError(f"{name} must be a number or a two-value range")

        fading_raw = value.get("fading")
        fading = None
        if fading_raw is not None:
            if not isinstance(fading_raw, Mapping):
                raise TypeError("fading must be a mapping")
            fading_unknown = set(fading_raw) - {"coherence_bandwidth", "power_delay_profile"}
            if fading_unknown:
                raise ValueError(f"unknown fading configuration keys: {sorted(fading_unknown)}")
            coherence = fading_raw.get("coherence_bandwidth")
            if coherence is None:
                raise ValueError("fading requires coherence_bandwidth")
            range_value = RFChannelImpairmentConfig.from_dict({"frequency_offset_hz": coherence}).frequency_offset_hz
            profile = tuple(float(item) for item in fading_raw.get("power_delay_profile", (1.0, 1.0)))
            fading = FadingConfig(coherence_bandwidth=range_value, power_delay_profile=profile)  # type: ignore[arg-type]

        return cls(
            frequency_offset_hz=parse_range("frequency_offset_hz"),
            phase_noise_degrees=parse_range("phase_noise_degrees"),
            fading=fading,
            noise_power_db=parse_range("noise_power_db"),
            normalization=value.get("normalization", "none"),
        )

    @property
    def enabled(self) -> bool:
        """Return whether this configuration changes generated samples."""
        return any((self.frequency_offset_hz, self.phase_noise_degrees, self.fading, self.noise_power_db)) or self.normalization != "none"


@dataclass(frozen=True)
class RFChannelImpairmentPipeline:
    """Apply reproducible RF impairments to records before they are written.

    Randomness is derived from ``global_seed``, the stable ``record_identity``,
    and the effect name. Consequently results do not depend on execution or
    worker order. Readers remain responsible only for faithful stored samples.
    """

    config: RFChannelImpairmentConfig = field(default_factory=RFChannelImpairmentConfig)
    global_seed: int = 0

    def __post_init__(self) -> None:
        """Validate the global generation seed."""
        if not isinstance(self.global_seed, int) or isinstance(self.global_seed, bool) or self.global_seed < 0:
            raise ValueError("global_seed must be a nonnegative integer")

    def _rng(self, record_identity: str | int, effect: str) -> np.random.Generator:
        payload = json.dumps([self.global_seed, str(record_identity), effect], separators=(",", ":")).encode()
        seed = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
        return np.random.default_rng(seed)

    @staticmethod
    def _normalize(data: np.ndarray) -> np.ndarray:
        power = float(np.mean(np.abs(data) ** 2))
        if not np.isfinite(power) or power <= 0:
            raise ValueError("cannot normalize IQ data with zero or non-finite power")
        return (data / np.sqrt(power)).astype(TorchSigComplexDataType)

    def apply(self, signal: Signal, *, record_identity: str | int, sample_rate: float) -> Signal:
        """Apply configured effects and record sampled provenance in metadata."""
        if not self.config.enabled:
            return signal
        if not isinstance(record_identity, (str, int)) or isinstance(record_identity, bool):
            raise TypeError("record_identity must be a string or integer")
        if not np.isfinite(sample_rate) or sample_rate <= 0:
            raise ValueError("sample_rate must be finite and positive")
        if signal.data.ndim != 1 or not np.iscomplexobj(signal.data):
            raise ValueError("RF channel augmentation requires one-dimensional complex IQ data")

        applied: list[dict[str, Any]] = []
        data = signal.data
        if self.config.normalization == "before":
            data = self._normalize(data)
            applied.append({"name": "normalize", "position": "before", "target_power": 1.0})
        if self.config.frequency_offset_hz is not None:
            offset = self.config.frequency_offset_hz.sample(self._rng(record_identity, "frequency_offset"))
            if abs(offset) >= sample_rate / 2:
                raise ValueError("sampled frequency offset must be within the Nyquist interval")
            data = frequency_shift(data, offset, sample_rate).astype(TorchSigComplexDataType)
            applied.append({"name": "frequency_offset", "offset_hz": offset})
        if self.config.phase_noise_degrees is not None:
            rng = self._rng(record_identity, "phase_noise")
            standard_deviation = self.config.phase_noise_degrees.sample(rng)
            data = carrier_phase_noise(data, phase_noise_degrees=standard_deviation, rng=rng)
            applied.append({"name": "phase_noise", "standard_deviation_degrees": standard_deviation})
        if self.config.fading is not None:
            rng = self._rng(record_identity, "fading")
            coherence = self.config.fading.coherence_bandwidth.sample(rng)
            data = fading(data, coherence, np.asarray(self.config.fading.power_delay_profile), rng)
            applied.append({"name": "fading", "model": "rayleigh", "coherence_bandwidth": coherence, "power_delay_profile": list(self.config.fading.power_delay_profile)})
        if self.config.noise_power_db is not None:
            rng = self._rng(record_identity, "awgn")
            power_db = self.config.noise_power_db.sample(rng)
            data = awgn(data, noise_power_db=power_db, rng=rng)
            applied.append({"name": "awgn", "noise_power_db": power_db})
        if self.config.normalization == "after":
            data = self._normalize(data)
            applied.append({"name": "normalize", "position": "after", "target_power": 1.0})

        signal.data = np.asarray(data, dtype=TorchSigComplexDataType)
        signal["rf_channel_impairments"] = {
            "version": 1,
            "global_seed": self.global_seed,
            "record_identity": str(record_identity),
            "sample_rate_hz": float(sample_rate),
            "order": [entry["name"] for entry in applied],
            "applied": applied,
            "configuration": asdict(self.config),
        }
        return signal
