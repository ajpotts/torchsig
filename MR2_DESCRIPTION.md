# Preserve configured bandwidth and support estimated-bandwidth YOLO boxes

## Summary

This MR prevents spectral leakage and high-SNR sidelobes from replacing a
component's configured canonical bandwidth.

SNR adjustment and occupied-bandwidth estimation are now separate operations.
The existing threshold-span calculation remains available as separate
metadata, while placement continues to consume the generator-selected full
two-sided `bandwidth`. YOLO labels can independently select either value for
their frequency-axis height and default to `estimated_occupied_bandwidth`.

## Changes

- Add `update_signal_snr()` to adjust signal data and `snr_db` without changing
  canonical bandwidth.
- Add `estimate_occupied_bandwidth()` for the existing max-hold, 3 dB
  threshold-span calculation.
- Keep `update_signal_snr_bandwidth()` as a compatibility wrapper.
- Store successful estimates as `estimated_occupied_bandwidth` instead of
  overwriting `bandwidth`.
- Correct the estimator documentation: it is threshold-based and is not a
  99%-power measurement.
- Validate positive, finite generator output bandwidths before SNR adjustment.
- For generators that explicitly declare `bandwidth_min` and `bandwidth_max` as
  required configuration, reject generated bandwidths outside that range.
- Document and validate tone as the explicit fixed 1 Hz exception.
- Add a `YOLOLabel.bandwidth_key` option supporting `"bandwidth"` and
  `"estimated_occupied_bandwidth"`.
- Default YOLO box heights to `estimated_occupied_bandwidth`; callers can use
  `YOLOLabel(bandwidth_key="bandwidth")` to retain canonical-bandwidth boxes.
- Validate unsupported YOLO bandwidth keys and missing selected bandwidth
  metadata with clear errors.
- Add `examples/scripts/compare_yolo_bandwidth_boxes.py` to generate
  side-by-side spectrogram examples using both YOLO bandwidth modes.

## Behavior

Before this MR, a BPSK component configured for at most 100 kHz could receive a
spectral estimate above 300 kHz and have that estimate become its canonical
bandwidth. After this MR, the generated bandwidth remains canonical and the
wider estimate is retained only as diagnostic metadata.

A deterministic run of 100 samples from
`wideband_clean_train_all.yaml` produced:

- 405 components across 68 generated class names;
- 403 components with an occupied-bandwidth diagnostic;
- zero out-of-range non-tone canonical bandwidths;
- deterministic coverage of 802.11a, BPSK, and QPSK components.

Before this change, the same investigation found 172 out-of-range components
among 394 generated components.

YOLO box generation now uses the threshold-based occupied-bandwidth estimate
by default:

```python
YOLOLabel()
```

Canonical generator bandwidth remains available as an explicit option:

```python
YOLOLabel(bandwidth_key="bandwidth")
```

Only the normalized box height changes between these modes. Box class, time
bounds, and center frequency continue to use their existing metadata.

## Compatibility

- `update_signal_snr_bandwidth()` remains available with the same arguments and
  in-place return behavior.
- Canonical `center_freq` and `bandwidth` field names are unchanged.
- Custom generators that do not declare configurable bandwidth bounds are
  required only to return a finite, positive bandwidth.
- Tone continues to use its existing 1 Hz metadata width.
- The new `estimated_occupied_bandwidth` field is additive.
- `YOLOLabel()` now uses `estimated_occupied_bandwidth` by default. Code that
  requires the previous canonical-bandwidth box height must pass
  `bandwidth_key="bandwidth"`.
- No dependencies are added.

## Validation

```bash
pytest -q \
    tests/utils/test_dsp.py \
    tests/datasets/test_dataset_utils.py \
    tests/signals/test_signal_types.py \
    tests/signals/builders/test_wifi.py \
    tests/signals/builders/test_constellation.py \
    tests/signals/builders/test_tone.py \
    tests/datasets/test_dataset_class_sampling.py
```

Result: `166 passed`.

```bash
pytest -q tests/datasets/test_datasets.py \
    -k 'generated_component_bandwidth or generated_tone or protocol_and_constellation_keep_generated_bandwidth'
```

Result: `11 passed, 108 deselected`.

```bash
pytest -q tests/transforms/test_metadata_transforms.py
```

Result: `42 passed`.

```bash
pytest -q tests/transforms/test_metadata_transforms.py \
    tests/datasets/test_datasets.py \
    -k 'yolo or generated_component_bandwidth or update_signal_snr_bandwidth'
```

Result: `14 passed, 147 deselected`.

```bash
python examples/scripts/compare_yolo_bandwidth_boxes.py \
    --examples 1 \
    --output /tmp/yolo_bandwidth_box_comparison.png
```

The script completes and writes the comparison image.

```bash
ruff format --check \
    torchsig/utils/dsp.py \
    torchsig/datasets/datasets.py \
    torchsig/signals/builders/tone.py \
    tests/utils/test_dsp.py \
    tests/datasets/test_datasets.py
git diff --check
```

Both checks pass. The repository's broader Ruff configuration reports existing
findings in unchanged portions of the large DSP and dataset test modules.

The complete `tests/datasets/test_datasets.py` module was also started. Its
first 26 tests passed, after which an existing dataset-creation case remained
running for several minutes and the check was stopped. The targeted tests for
all behavior changed by this MR pass.
