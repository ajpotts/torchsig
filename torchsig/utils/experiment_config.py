"""Validated experiment configuration for signal generation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from torchsig.utils.signal_building import public_generator_names

__all__ = ["ExperimentConfig", "FixedValue", "SignalConfig", "load_experiment_config"]

_FIXED_PARAMETERS = {"qpsk": {"alpha"}}


@dataclass(frozen=True)
class FixedValue:
    """A generator parameter fixed to one value for every generated signal."""

    value: Any


@dataclass(frozen=True)
class SignalConfig:
    """Generator parameter overrides for one concrete signal class."""

    parameters: Mapping[str, FixedValue]

    def __post_init__(self) -> None:
        """Validate parameters and store them in a read-only mapping."""
        if not isinstance(self.parameters, Mapping):
            raise ValueError("SignalConfig.parameters must be a mapping")  # noqa: TRY004
        parameters = dict(self.parameters)
        for name, value in parameters.items():
            if not isinstance(name, str):
                raise ValueError("SignalConfig parameter names must be strings")  # noqa: TRY004
            if not isinstance(value, FixedValue):
                raise ValueError(f"SignalConfig parameter {name!r} must be a FixedValue")  # noqa: TRY004
        object.__setattr__(self, "parameters", MappingProxyType(parameters))


class ExperimentConfig:
    """Validated, programmatic signal-generation configuration.

    Args:
        signals: Mapping of concrete signal class names to their configuration.

    Example:
        Configure a fixed QPSK pulse-shaping rolloff::

            config = ExperimentConfig(
                signals={
                    "qpsk": SignalConfig(
                        parameters={"alpha": FixedValue(0.35)},
                    ),
                },
            )

    The initial schema supports fixed parameter values. ``FixedValue`` leaves
    room for future range, choice, and distribution value specifications.
    """

    def __init__(self, signals: Mapping[str, SignalConfig] | None = None) -> None:
        if signals is None:
            signals = {}
        if not isinstance(signals, Mapping):
            raise ValueError("ExperimentConfig.signals must be a mapping")  # noqa: TRY004

        validated: dict[str, SignalConfig] = {}
        for class_name, signal_config in signals.items():
            if not isinstance(class_name, str) or class_name not in public_generator_names:
                raise ValueError(f"unknown signal class {class_name!r}")
            if not isinstance(signal_config, SignalConfig):
                raise ValueError(f"configuration for signal class {class_name!r} must be a SignalConfig")  # noqa: TRY004

            supported = _FIXED_PARAMETERS.get(class_name, set())
            unknown_parameters = set(signal_config.parameters).difference(supported)
            if unknown_parameters:
                raise ValueError(f"unknown parameters for signal class {class_name!r}: {sorted(unknown_parameters)}; supported parameters: {sorted(supported)}")
            for parameter_name, parameter in signal_config.parameters.items():
                _validate_fixed_value(class_name, parameter_name, parameter.value)
            validated[class_name] = signal_config

        self._signals = MappingProxyType(validated)

    @property
    def signals(self) -> Mapping[str, SignalConfig]:
        """Read-only mapping of configured signal classes."""
        return self._signals

    def parameter_value(self, class_name: str, parameter_name: str) -> float | None:
        """Return a fixed override, or ``None`` when it is not configured."""
        signal_config = self._signals.get(class_name)
        if signal_config is None:
            return None
        parameter = signal_config.parameters.get(parameter_name)
        return None if parameter is None else float(parameter.value)

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical configuration suitable for YAML serialization."""
        return {
            "signals": {class_name: {"parameters": {name: {"value": parameter.value} for name, parameter in signal_config.parameters.items()}} for class_name, signal_config in self._signals.items()}
        }


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be a mapping")  # noqa: TRY004
    return value


def _validate_fixed_value(class_name: str, parameter_name: str, value: Any) -> None:
    location = f"signals.{class_name}.parameters.{parameter_name}.value"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a real number")  # noqa: TRY004
    if parameter_name == "alpha" and not 0.0 < float(value) < 1.0:
        raise ValueError(f"{location} must be between 0 and 1 (exclusive)")


def _config_from_mapping(loaded: Any) -> ExperimentConfig:
    root = _mapping(loaded, "experiment configuration root")
    unknown_root = set(root).difference({"signals"})
    if unknown_root:
        raise ValueError(f"unknown experiment configuration settings: {sorted(unknown_root)}")
    signals = _mapping(root.get("signals", {}), "signals")
    parsed_signals: dict[str, SignalConfig] = {}

    for class_name, class_config_value in signals.items():
        if not isinstance(class_name, str) or class_name not in public_generator_names:
            raise ValueError(f"unknown signal class {class_name!r}")
        class_config = _mapping(class_config_value, f"signals.{class_name}")
        unknown_class_settings = set(class_config).difference({"parameters"})
        if unknown_class_settings:
            raise ValueError(f"unknown settings for signal class {class_name!r}: {sorted(unknown_class_settings)}")
        parameters = _mapping(class_config.get("parameters", {}), f"signals.{class_name}.parameters")

        parsed_parameters: dict[str, FixedValue] = {}
        for parameter_name, parameter_config_value in parameters.items():
            parameter_config = _mapping(parameter_config_value, f"signals.{class_name}.parameters.{parameter_name}")
            unknown_parameter_settings = set(parameter_config).difference({"value"})
            if unknown_parameter_settings:
                raise ValueError(f"unknown settings for parameter {class_name}.{parameter_name}: {sorted(unknown_parameter_settings)}")
            if "value" not in parameter_config:
                raise ValueError(f"signals.{class_name}.parameters.{parameter_name} requires a value")
            parsed_parameters[parameter_name] = FixedValue(parameter_config["value"])
        parsed_signals[class_name] = SignalConfig(parameters=parsed_parameters)

    return ExperimentConfig(signals=parsed_signals)


def load_experiment_config(
    config: ExperimentConfig | str | Path | Mapping[str, Any] | None,
) -> ExperimentConfig:
    """Load and validate an optional experiment configuration.

    Args:
        config: A YAML path, mapping, validated configuration, or ``None``.

    Returns:
        A validated configuration. ``None`` resolves to an empty configuration.

    Raises:
        ValueError: If the schema, class, parameter, type, or value is invalid.
    """
    if config is None:
        return ExperimentConfig()
    if isinstance(config, ExperimentConfig):
        return config
    if isinstance(config, (str, Path)):
        path = Path(config)
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as error:
            raise ValueError(f"invalid experiment configuration YAML in {path}: {error}") from error
        except OSError as error:
            raise ValueError(f"could not read experiment configuration {path}: {error}") from error
    else:
        loaded = config

    return _config_from_mapping(loaded)
