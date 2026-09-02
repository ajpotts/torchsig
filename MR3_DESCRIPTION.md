# Unify wideband placement and YOLO frequency geometry

## Summary

This MR completes the frequency-metadata repair plan by making wideband
placement and localization labels consume the same canonical full-bandwidth
interval.

Overlap rectangles now use the signal's derived lower and upper frequency edges
and normalize them using the complete sample-rate span. Final generated
components and YOLO inputs are validated through a shared signal-metadata
method.

## Changes

- Add `SignalMetadataObject.validate_frequency_interval()` to validate:

  - finite center frequency;
  - finite, positive full bandwidth;
  - finite and ordered optional dataset bounds;
  - derived lower and upper edges against those bounds.

- Validate every component after frequency shifting and any boundary clipping.
- Change `_map_to_coordinates()` to use `lower_freq` and `upper_freq` rather
  than independently applying a bandwidth interpretation.
- Normalize overlap frequency coordinates by `sample_rate`, fixing the previous
  half-sample-rate scaling.
- Replace side-intersection-based rectangle overlap with inclusive axis-aligned
  interval comparisons, covering partial overlap and collinear edges.
- Validate a component's final frequency interval before creating its YOLO
  label.
- Add deterministic tests for valid and invalid intervals, exact overlap-bin
  geometry, a boundary-clipped YOLO label, and rejection of out-of-bounds label
  input.

## Geometry correction

Previously, overlap placement calculated frequency edges as:

```text
center_freq - bandwidth
center_freq + bandwidth
```

and divided the shifted frequency by `sample_rate / 2`. The corrected mapping
is:

```text
lower_freq = center_freq - bandwidth / 2
upper_freq = center_freq + bandwidth / 2
normalized_edge = (edge + sample_rate / 2) / sample_rate
```

This matches the bandwidth convention already used by metadata properties and
YOLO label height.

## Compatibility

- Existing metadata field names and YOLO tuple format are unchanged.
- Correctly bounded component labels retain their existing formulas.
- Overlap decisions may change because rectangles now describe the actual
  canonical interval instead of an expanded, mis-scaled interval.
- Invalid or out-of-bounds metadata now raises `ValueError` before insertion or
  labeling.
- No dependencies are added.

## Validation

```bash
pytest -q \
    tests/signals/test_signal_types.py \
    tests/transforms/test_metadata_transforms.py \
    tests/datasets/test_dataset_utils.py \
    tests/utils/test_coordinate_system.py
```

Result: `122 passed`.

```bash
pytest -q tests/datasets/test_datasets.py \
    -k 'iterable_dataset_applies_yolo_label or map_to_coordinates or overlap_detection_uses_canonical_frequency_intervals or protocol_and_constellation_keep_generated_bandwidth or generated_component_bandwidth or generated_tone'
```

Result: `17 passed, 106 deselected`.

```bash
pytest -q \
    tests/utils/test_coordinate_system.py \
    tests/utils/test_dsp.py \
    tests/signals/builders/test_wifi.py \
    tests/signals/builders/test_constellation.py \
    tests/signals/builders/test_tone.py \
    tests/datasets/test_dataset_class_sampling.py
```

Result: `170 passed`.

A deterministic end-to-end run generated and labeled 100 samples from
`wideband_clean_train_all.yaml`. All 413 component intervals were within the
dataset bounds and all YOLO tuples matched the final component metadata.

The test environment emitted the existing CUDA NVML warning and expected
signal-duration warnings from class-sampling fixtures. Neither affected the
results.
