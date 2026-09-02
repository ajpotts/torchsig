# Preserve configured signal bandwidth during SNR adjustment

## Summary

This MR prevents spectral leakage and high-SNR sidelobes from replacing a
component's configured canonical bandwidth.

SNR adjustment and occupied-bandwidth estimation are now separate operations.
The existing threshold-span calculation remains available as diagnostic
metadata, while placement and downstream labels continue to consume the
generator-selected full two-sided `bandwidth`.

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

## Compatibility

- `update_signal_snr_bandwidth()` remains available with the same arguments and
  in-place return behavior.
- Canonical `center_freq` and `bandwidth` field names are unchanged.
- Custom generators that do not declare configurable bandwidth bounds are
  required only to return a finite, positive bandwidth.
- Tone continues to use its existing 1 Hz metadata width.
- The new `estimated_occupied_bandwidth` field is additive.
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
