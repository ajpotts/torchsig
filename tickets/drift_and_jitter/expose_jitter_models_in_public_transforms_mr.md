# Expose sampling-clock jitter models through public transforms

## Summary

This change exposes the sampling-clock jitter models already implemented by
the DSP layer through both public APIs:

* `clock_jitter` now accepts `jitter_model` and `jitter_alpha`.
* `ClockJitter` accepts, validates, stores, and forwards the same settings.
* Public documentation distinguishes independent absolute timing jitter,
  accumulated sampling-period jitter, and correlated AR(1) jitter.

The existing behavior remains the default: `jitter_model="independent"` and
`jitter_alpha=0.9`.

## Validation

Focused tests cover all three public model selections, functional and class
parameter forwarding, invalid model and AR(1) coefficient handling, seeded
reproducibility, and preservation of the existing default. Existing DSP tests
continue to cover timing statistics and NumPy/Numba parity for every jitter
model.

## Compatibility

This is an additive API change. Existing calls retain their current behavior,
output dtype, output length, and random-number semantics.
