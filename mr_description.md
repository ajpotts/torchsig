# Add YAML and programmatic signal-generation configuration

## Summary

This MR introduces an optional experiment configuration system for overriding signal-generation parameters on a per-signal-class basis. It supports both YAML-based and typed Python configuration while preserving all existing defaults and randomization when no configuration is supplied.

The initial supported override fixes the QPSK SRRC pulse-shaping rolloff (`alpha`) to a value strictly between 0 and 1.

## Motivation

Signal generator parameters such as QPSK rolloff were previously randomized internally, making it inconvenient to reproduce controlled experiments or compare configurations. Experiment settings also had to be expressed in Python, which made them harder to version, review, and diff.

This change provides a validated, serializable configuration model that can grow to support additional generator parameters, distributions, shared defaults, transforms, impairments, and reproducibility settings.

## Changes

- Add the typed configuration classes `ExperimentConfig`, `SignalConfig`, and `FixedValue`.
- Add `load_experiment_config` for loading the same schema from a YAML path or mapping.
- Accept an optional `experiment_config` in `TorchSigIterableDataset` and propagate it through `TorchSigDataModule` and `SplitTorchSigDataModule`.
- Extend dataset YAML files with an optional top-level `signals` section.
- Validate configuration structure, signal class names, parameter names, value types, and allowed ranges before dataset generation.
- Apply fixed QPSK `alpha` values to every generated QPSK signal and select SRRC pulse shaping when the override is present.
- Preserve the existing randomized pulse-shape and rolloff behavior when `alpha` is not configured.
- Record the canonical effective configuration under `experiment_config` in `dataset_info.yaml`.
- Document the YAML and programmatic APIs.
- Add a persistent demonstration script at `examples/scripts/demo_signal_generation_config.py`.

## YAML example

```yaml
signals:
  qpsk:
    parameters:
      alpha:
        value: 0.35
```

The `signals` mapping can be included in an existing TorchSig dataset YAML file. Unspecified signal classes and parameters retain their existing behavior.

## Programmatic example

```python
from torchsig.utils.experiment_config import (
    ExperimentConfig,
    FixedValue,
    SignalConfig,
)

experiment_config = ExperimentConfig(
    signals={
        "qpsk": SignalConfig(
            parameters={"alpha": FixedValue(0.35)},
        ),
    },
)

dataset = TorchSigIterableDataset(
    signal_generators=["qpsk"],
    metadata=dataset_metadata,
    experiment_config=experiment_config,
)
```

The YAML and Python forms use the same validation and serialize to the same canonical representation.

## Demonstration

Run the example from the repository root:

```bash
python examples/scripts/demo_signal_generation_config.py
```

By default, the script retains its configuration and generated dataset under `examples/datasets/signal_generation_config_demo`. Use `--output-dir` to select another location and `--overwrite` to replace an existing demonstration output.

## Validation and errors

Configuration is rejected before dataset generation when it contains:

- an unknown signal class;
- an unsupported parameter;
- an unknown schema key;
- a missing fixed value;
- a non-numeric QPSK `alpha`; or
- an `alpha` outside the exclusive range `(0, 1)`.

Error messages include the relevant class, parameter, or configuration path.

## Backward compatibility

No configuration is required. Existing dataset construction APIs and YAML files continue to work, and generator behavior remains unchanged when no override is supplied.

## Testing

- Added coverage for YAML parsing and canonical serialization.
- Added coverage for typed programmatic configuration.
- Added invalid class, parameter, schema, type, and range cases.
- Verified that every configured QPSK example uses the fixed `alpha` and SRRC pulse shaping.
- Verified randomized fallback behavior without an override.
- Verified persistence of the effective configuration in `dataset_info.yaml`.
- Added a subprocess smoke test for the demonstration script.

Focused regression result: `133 passed, 3 deselected`.
