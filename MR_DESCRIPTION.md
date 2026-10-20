# Expose configurable drift models through public clock transforms

## Summary

- Expose `constant`, `linear`, and `random_walk` drift selection through
  `clock_drift()` and `ClockDrift`.
- Add public configuration for signed initial period offset, linear drift rate,
  random-walk innovation strength, and physical sample rate.
- Support seeded scalar or range draws for configurable `ClockDrift` numeric
  parameters.
- Validate model names, parameter bounds, finite values, and sample rate in the
  class-based transform.
- Preserve deterministic constant drift as the public default.

## Public API

```python
clock_drift(
    data,
    drift_ppm=10,
    drift_model="constant",
    drift_rate_ppm_per_sec=0.0,
    drift_noise_ppm_per_sqrt_sec=0.0,
    sample_rate_hz=1.0,
)
```

`ClockDrift` exposes the same options. Its offset, rate, and noise parameters
may be fixed scalars or two-value ranges sampled reproducibly from the
transform seed.

The ppm values describe sampling-period error. Positive values advance input
sampling positions faster. Linear rates use ppm/second, while random-walk
innovation strength uses ppm/sqrt(second).

## Compatibility

Callers that omit the new arguments use the `constant` model. Linear rate and
random-walk noise both default to zero, so merely upgrading does not enable a
new stochastic process.

## Testing

- Public functional pass-through for constant, linear, and random-walk models.
- Functional and class-based default configuration.
- Class-based parameter distribution and pass-through behavior.
- Invalid model, offset, rate, noise, and sample-rate validation.
- Sampling-position and Python/Numba parity regression coverage.

Commands run:

```bash
pytest -q tests/utils/test_sampling_clock_drift_models_api.py \
  tests/transforms/test_functional.py tests/transforms/test_transforms.py
```

Result: 209 passed.

```bash
ruff check --ignore RUF100 torchsig/transforms/functional.py \
  torchsig/transforms/transforms.py \
  tests/utils/test_sampling_clock_drift_models_api.py
```

Result: all checks passed. `git diff --check` also passed.
