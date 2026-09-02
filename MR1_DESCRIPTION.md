# Harden canonical frequency metadata serialization

## Summary

This MR completes the first stage of the frequency-metadata repair plan. It
prevents legacy cached frequency edges from being exposed in serialized signal
metadata and documents how frequency metadata changes during boundary clipping.

`center_freq` and full two-sided `bandwidth` remain the canonical fields.
`lower_freq` and `upper_freq` continue to be derived from them. No bandwidth
selection, bandwidth measurement, placement, or label behavior changes in this
MR.

## Changes

- Exclude `_lower_frequency` and `_upper_frequency` from
  `SignalMetadataObject.to_dict()`.
- Retain canonical `center_freq` and `bandwidth` in serialized metadata.
- Clarify in `frequency_shift_signal()` that an out-of-band interval is clipped
  by updating both canonical fields to describe the retained signal.
- Extend serialization coverage to include contradictory legacy cache values
  and verify that they are omitted.
- Strengthen negative-boundary clipping coverage by checking the final derived
  edges and confirming no cached edge fields remain.

## Rationale

The runtime metadata implementation already invalidates cached edges when a
canonical field changes and calculates edges from the canonical pair. However,
legacy cache values could still be supplied when constructing an object and
then emitted by `to_dict()`. Omitting those internal fields ensures persisted
metadata has one authoritative representation of the frequency interval.

## Compatibility

- Public canonical metadata fields are unchanged.
- Runtime frequency calculations are unchanged.
- Serialized output no longer includes the internal `_lower_frequency` and
  `_upper_frequency` implementation fields.
- No dependencies are added.

## Validation

```bash
pytest -q tests/signals/test_signal_types.py tests/datasets/test_dataset_utils.py
```

Result: `35 passed`.

```bash
pytest -q \
    tests/utils/file_handlers/test_metadata_codec.py \
    tests/utils/file_handlers/test_hdf5.py \
    tests/utils/file_handlers/test_packed_hdf5.py \
    tests/utils/file_handlers/test_homogeneous_hdf5.py
```

Result: `97 passed`.

```bash
ruff format --check \
    torchsig/signals/signal_types.py \
    torchsig/datasets/dataset_utils.py \
    tests/signals/test_signal_types.py \
    tests/datasets/test_dataset_utils.py
git diff --check
```

Both checks passed. `ruff check` reports five pre-existing findings in unchanged
test lines; this MR introduces no new lint findings.

The test environment also emitted the existing CUDA NVML warning, which did not
affect the results.
