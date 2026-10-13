## Summary

This merge request clarifies and extends the sampling-clock jitter model.

- Makes independent absolute timing jitter the default, matching ADC aperture
  jitter semantics.
- Retains the previous accumulated sample-period behavior as
  `jitter_model="period"` for free-running oscillator simulations.
- Adds `jitter_model="ar1"` for stationary filtered clock noise and exposes
  `jitter_alpha` to control its correlation.
- Defines `jitter_ppm` as the timing-error standard deviation in millionths of
  one input-sample period and converts it to polyphase units consistently.
- Keeps the NumPy and Numba implementations aligned, including random-number
  ordering and out-of-range sampling-position handling.
- Adds public API documentation and validation for model names and AR(1)
  coefficients.

The public API remains backward compatible: the new arguments are optional.
The default stochastic process changes from accumulated period error to
independent aperture jitter; callers that require the previous behavior can
select `jitter_model="period"` explicitly.

Related issue: #475

## Test Plan

Ran the affected suites against `3.0.0-dev` in full test mode:

```text
pytest -q --test-mode full tests/utils/test_sampling_clock_validation.py tests/utils/test_dsp_numba.py tests/transforms/test_transforms.py tests/transforms/test_functional.py
```

Result: `262 passed in 46.25s`.

Coverage includes:

- independent jitter as the public default;
- distinct independent, accumulated-period, and AR(1) results;
- equivalence of AR(1) with zero correlation and independent jitter;
- invalid model and coefficient handling;
- deterministic seeded behavior;
- NumPy/Numba parity for every jitter model; and
- output length, dtype, and sampling-boundary behavior.

## Before Submitting

- [x] Checked the focused diff for conflict markers and whitespace errors.
- [x] Added and updated unit tests in `tests/`.
- [x] Ran the affected Pytest suites in full test mode.
- [ ] Run the complete repository Pytest suite.
- [ ] Run example notebooks.
- [ ] Run Pylint and confirm a score above 9/10.
- [ ] Add contributors to the MR metadata as appropriate.
