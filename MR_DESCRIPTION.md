# Add detailed modulation-recognition target labels

## Summary

This MR adds waveform metadata needed to train modulation-recognition models
that report properties beyond the modulation family or class name.

OFDM signals now report whether a cyclic prefix is present and its unoversampled
length. Constellation signals now report the pulse-shaping filter and, when
applicable, its roll-off factor. The values stored in metadata are the same
values used to generate each waveform.

The MR also includes a small end-to-end training example demonstrating that all
four targets can be consumed and learned from generated IQ data.

## New metadata

### OFDM

- `has_cyclic_prefix`: Boolean indicating whether the generated waveform uses
  a cyclic prefix.
- `cyclic_prefix_len`: Cyclic-prefix length before oversampling. A waveform
  without a cyclic prefix uses `0`.

The OFDM generator now selects the cyclic-prefix state and length before calling
the modulator. The selected length is passed through the modulation functions,
which ensures the returned label describes the waveform that was actually
generated. Direct modulator calls can still omit the parameter to retain random
selection inside the baseband modulator.

### Constellation waveforms

- `pulse_shape_name`: Either `"srrc"` or `"rectangular"`.
- `alpha_rolloff`: The generated SRRC roll-off factor, or `None` for a
  rectangular pulse shape.
- `pulse_shape_index`: Tensor-ready pulse-shape class (`0` for rectangular,
  `1` for SRRC).
- `alpha_rolloff_target`: Numeric roll-off target, using `0.0` when roll-off
  does not apply.

## Training example

`examples/scripts/train_modrec_metadata_targets.py` builds deliberately
constrained `TorchSigIterableDataset` instances and consumes them through
seeded `WorkerSeedingDataLoader` instances. The training loops receive the
features and targets directly from each loader:

```python
spectrogram, (has_cp, cp_len) = next(ofdm_batches)
spectrogram, (pulse_shape, alpha) = next(constellation_batches)
```

The example does not manually generate signals, inspect component metadata, or
materialize an intermediate dataset.

The example uses masked losses because some targets are conditional:

- Cyclic-prefix targets apply only to OFDM signals.
- Pulse-shape targets apply only to constellation signals.
- Roll-off regression applies only to SRRC-shaped constellation signals.

TorchSIG's `Spectrogram` transform supplies model features within the dataset
pipeline. Two small two-head models demonstrate the OFDM and constellation
targets independently.

Run the complete example from the repository root:

```bash
python examples/scripts/train_modrec_metadata_targets.py
```

For a quick smoke run:

```bash
python examples/scripts/train_modrec_metadata_targets.py --steps 2
```

The default configuration completes in approximately ten seconds in the
development environment. One representative run produced:

```text
cyclic prefix accuracy: 93.0%
cyclic prefix length MAE: 2.16 samples
pulse shape accuracy: 100.0%
alpha roll-off MAE: 0.035
```

These figures demonstrate the example rather than establishing a model-quality
benchmark. The dataset intentionally omits channel impairments and holds the
sample rate, bandwidth, duration, constellation, and subcarrier count fixed.

## Compatibility

- Existing metadata fields and class labels are unchanged.
- The new fields are additive.
- `ofdm_modulator()` and `ofdm_modulator_baseband()` accept an optional
  `cyclic_prefix_len`; omitting it preserves randomized cyclic-prefix behavior.
- No new runtime dependency is required.

## Validation

```bash
pytest -q \
    tests/signals/builders/test_constellation.py \
    tests/signals/builders/test_ofdm.py
```

Result: `76 passed`.

```bash
ruff format --check examples/scripts/train_modrec_metadata_targets.py
ruff check examples/scripts/train_modrec_metadata_targets.py
python examples/scripts/train_modrec_metadata_targets.py
git diff --check
```

All checks completed successfully. The environment emitted the existing NVML
and non-writable Matplotlib-cache warnings; neither affected execution.
