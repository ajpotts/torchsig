# Consolidate sampling-clock transform wrapper setup

## Summary

This change extracts the resampler setup shared by `clock_drift` and
`clock_jitter` into one private helper. Input dtype conversion, the fixed
polyphase rate, prototype-filter construction, impairment dispatch,
group-delay calculation, and aligned output normalization now have a single
implementation.

The public drift and jitter functions remain separate and retain their current
signatures, defaults, model-specific validation, documentation, RNG behavior,
output length, and complex64 output dtype.

## Validation

Regression tests reproduce the wrapper setup that existed before this
refactor and compare it sample-for-sample with the consolidated wrappers. The
coverage includes:

* zero and nonzero drift;
* zero and nonzero jitter;
* nonzero initial sampling phase;
* output length and dtype;
* both the NumPy and Numba-backed impairment implementations.

The existing transform and sampling-clock test suites continue to validate
alignment, jitter-model selection, seeded reproducibility, and implementation
parity.

## Compatibility

This is an internal refactor with no intended public API or behavioral change.
No dependencies are added.
