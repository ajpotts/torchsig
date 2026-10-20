# Add statistical tests for sampling-clock jitter and drift models

## Summary

- Add theory-based statistical unit tests for independent, period, and AR(1)
  sampling-clock jitter.
- Validate random-walk drift innovation variance, period-error variance, and
  integrated timing-error variance across seeded ensembles.
- Verify random-walk scaling at a fixed physical duration across sample rates.
- Confirm constant and linear drift are deterministic across RNG seeds.
- Use fixed seeds and uncertainty-based tolerances without adding dependencies.

## Statistical coverage

The jitter tests verify:

- zero mean and configured variance for independent absolute jitter;
- white, configured-variance increments for period jitter;
- stationary variance and positive or negative lag-one correlation for AR(1)
  jitter.

The drift tests verify:

- Gaussian random-walk innovation statistics;
- linear growth of period-error variance;
- the cubic integrated-random-walk timing-error variance law;
- physical-time scaling across sample rates;
- deterministic constant and linear trajectories.

## Test design

Tests call `generate_sampling_positions()` directly. Long seeded trajectories
are used for stationary jitter estimates, while a moderate seeded ensemble is
used for random-walk endpoint distributions. Variance tolerances are derived
from the ensemble estimator's sampling uncertainty.

No production code or dependencies are changed.

## Testing

```bash
pytest -q tests/utils/test_sampling_clock_model_statistics.py \
  tests/utils/test_sampling_positions.py
```

Result: 27 passed.

```bash
ruff check tests/utils/test_sampling_clock_model_statistics.py
```

Result: all checks passed. `git diff --check` also passed.
