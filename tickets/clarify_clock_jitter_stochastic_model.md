Title: Clarify and extend the clock jitter stochastic model

Summary

The clock jitter implementation accumulates independent Gaussian perturbations
in the sampling position, while its public documentation describes Gaussian
sampling-phase error. This makes the existing behavior an accumulated
sample-period error model rather than independent ADC aperture jitter and does
not give callers a way to choose the physical process they need.

Proposed change

- Add an explicit `jitter_model` option to the functional and transform APIs.
- Support these timing-error processes:

  | Model | Bounded timing variance | Time correlated | Intended use |
  | --- | --- | --- | --- |
  | `independent` | Yes | No | ADC aperture jitter |
  | `period` | No | Yes | Free-running oscillator |
  | `ar1` | Yes | Yes | Filtered clock noise |

- Make `independent` the default so each sampling instant receives an
  independent absolute timing error.
- Preserve the previous accumulated behavior as the opt-in `period` model.
- Add `jitter_alpha`, defaulting to `0.9`, to control AR(1) correlation while
  preserving `jitter_ppm` as the stationary timing-error standard deviation.
- Apply the same equations and random-number ordering in the NumPy and Numba
  implementations.
- Define `jitter_ppm` in millionths of one input-sample period and convert it
  to polyphase units before sampling.

Acceptance criteria

- Calling `clock_jitter` or `ClockJitter` without a model selects
  `independent` jitter.
- The `independent`, `period`, and `ar1` models produce distinct timing
  processes for nonzero correlation.
- `ar1` with `jitter_alpha=0` is equivalent to independent jitter for the same
  random seed.
- `jitter_alpha` rejects nonfinite values and values whose magnitude is at
  least one.
- Unknown jitter-model names raise a clear `ValueError`.
- NumPy and Numba outputs agree to the existing float32 tolerance for all
  three models.
- Output length and complex dtype behavior remain compatible with the current
  clock-jitter transform contract.
- Tests cover default selection, model differences, AR(1) coefficient
  validation, reproducibility, and NumPy/Numba parity.
