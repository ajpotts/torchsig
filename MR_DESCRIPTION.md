# Sampling-clock drift and jitter corrections

**Target branch:** `3.0.0-dev`

## Summary

This MR corrects the sampling-clock impairment engine and replaces the former
constant clock offset with selectable time-varying drift models. It also makes
low-PPM aperture jitter behave as requested, aligns zero-magnitude impairments
with the input, and adds explicit boundary and physical-rate controls.

The relevant review comparison is this branch against `3.0.0-dev`. Earlier
clock-impairment commits and reports on the feature branch provide background,
but are not the implementation baseline for this MR.

## Changes

- Add three clock-drift models:
  - `linear`: ramps between an initial and final PPM error.
  - `random_walk`: accumulated Gaussian rate error.
  - `filtered_noise`: correlated Gaussian rate error.
- Add physical linear-drift parameters:
  - `initial_drift_ppm`
  - `drift_rate_ppm_per_second`
  - `sample_rate`
- Evaluate linear-drift sampling positions with closed-form integration rather
  than accumulated floating-point increments.
- Interpolate between adjacent polyphase branches, including carry from the
  final branch to the next input sample.
- Remove the effective 200 PPM jitter quantization and associated negative
  timing bias.
- Align the resampler using the prototype filter group delay, making a
  zero-magnitude impairment with zero initial phase a no-op for passband data.
- Default `ClockDrift` and `ClockJitter` to zero initial sampling phase.
- Add `initial_phase` support to `clock_jitter` and `ClockJitter`.
- Replace duplicated symmetric trim/pad logic with aligned tail restoration.
- Add boundary policies: `edge` (default), `zeros`, `wrap`, and `raise`.
- Preserve complex dtype when extending output.
- Fix the one-excess-sample case that could return an empty array.
- Keep the NumPy and Numba implementations deterministic and numerically
  consistent for identical seeds.
- Remove unreachable bounds handling, fragile in-place array resizing, and
  redundant integer copying.
- Document drift semantics, boundary behavior, alignment, and the prototype
  filter's near-Nyquist limitations.

## API notes

The constant sampling-clock offset behavior has been removed. The default is
now `drift_model="linear"`.

```python
ClockDrift(
    drift_ppm=(-10.0, 10.0),
    initial_drift_ppm=(0.0, 0.0),
    drift_model="linear",
    initial_phase=(0.0, 0.0),
    boundary_mode="edge",
)
```

Physical linear drift can be requested with explicit time units:

```python
ClockDrift(
    drift_model="linear",
    initial_drift_ppm=(1.0, 1.0),
    drift_rate_ppm_per_second=(0.25, 0.25),
    sample_rate=1_000_000.0,
)
```

For stochastic models, `abs(drift_ppm)` is the RMS scale. The random-walk
model uses it as the endpoint RMS scale; the filtered-noise model uses it as
the stationary RMS scale.

## Compatibility considerations

- `ClockDrift` no longer applies a constant rate offset.
- The default initial phase changed from randomized to zero.
- Output exhaustion now defaults to edge extension instead of zero-padding.
- Applications that require the previous padding behavior can set
  `boundary_mode="zeros"`.
- Nearly full-band input can still experience prototype-filter attenuation or
  image leakage; this limitation is now documented.

## Verification

Focused sampling-clock and transform tests:

```text
pytest -q tests/utils/test_dsp_numba.py \
  tests/transforms/test_functional.py \
  tests/transforms/test_transforms.py

291 passed, 4 skipped in 3.65s
```

Additional checks:

```text
git diff --check
ruff check --select E9,F63,F7,F82 ...
python -m compileall -q ...
```

All additional checks passed.
