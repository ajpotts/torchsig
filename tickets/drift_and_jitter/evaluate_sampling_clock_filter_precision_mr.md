# Document and test sampling-clock filter precision

## Summary

This MR completes the sampling-clock coefficient-precision evaluation and
retains float32 coefficients in the production transforms.

For the production 5,000-phase, 115,101-tap prototype, float32 and float64
responses are effectively identical across the documented validity boundary:

- At normalized frequency 0.35: -0.033526 dB for both representations.
- At normalized frequency 0.70: -81.580246 dB for float32 and -81.580269 dB
  for float64.
- Float32 coefficient storage is 460,404 bytes, compared with 920,808 bytes
  for float64.

Differences become material only in the deep stopband below approximately
-160 dB, outside TorchSIG's complex64 precision and documented 80 dB rejection
contract. Both resampler implementations also construct float32 polyphase
filter banks and return complex64, so using float64 input coefficients would
double prototype storage without a material end-to-end fidelity benefit.

## Changes

- Document the production float32 response contract: less than 0.05 dB loss
  through the +/-0.35 occupied-band edges and at least 80 dB rejection at
  normalized frequency 0.70.
- Make the response regression test exercise the float32 conversion used by
  the public transforms.
- Add a float32-versus-float64 response and storage regression test.
- Add an optional benchmark that records response, coefficient memory, and
  steady-state NumPy/Numba resampling time for both representations.

## Performance

In a local 4,096-sample steady-state benchmark:

- NumPy median execution was 626 us with float32 coefficients and 644 us with
  float64 coefficients.
- Numba median execution was approximately 519 us for both representations.

These measurements support retaining float32 but are diagnostic rather than
cross-platform performance requirements.

## Validation

```text
pytest -q tests/utils/test_sampling_clock_filter_contract.py \
  tests/transforms/test_sampling_clock_wrapper_consolidation.py \
  tests/utils/test_dsp_numba.py \
  -k 'sampling or clock or filter_contract'
```

Result: 19 passed, 1 skipped, 19 deselected.

```text
pytest -q benchmarks/optional/benchmark_sampling_clock_filter_precision.py \
  --benchmark-only --benchmark-min-rounds=3
```

Result: 6 passed.

Ruff checks, formatting checks, and `git diff --check` also pass.

## Compatibility

This MR does not change public APIs, output dtype, or production numerical
behavior. No dependencies are added.
