# Proposal: Realistic Initial State for Sampling-Clock Drift

## Summary

TorchSIG should model the initial state of a sampling clock with two independent
quantities:

1. An initial sampling phase, which determines where within an input-sample
   interval the receiver takes its first sample.
2. An initial sampling-rate error, which determines how quickly subsequent
   sampling positions advance relative to the nominal clock.

The current implementation models only the second quantity. Its `drift_ppm`
value is a constant rate error for the entire signal, while the first sampling
position is fixed to a legacy polyphase-filter alignment. Consequently, two
captures with the same rate error always begin at the same fractional sampling
phase.

The recommended first change is to add a random initial sampling phase and
allow the existing constant rate error to be sampled symmetrically around
zero. Time-varying oscillator wander should be a later, separate feature rather
than being folded into the existing `drift_ppm` parameter.

## Terminology and current behavior

The note describes a clock frequency of the form

```text
f(t) = f0 + delta_f_drift(t)
```

For the resampler, it is clearer to express the same model as a dimensionless
fractional sampling-rate error in PPM:

```text
epsilon(t) = epsilon0 + epsilon_wander(t)
```

where:

- `epsilon0` is the initial, constant rate error in PPM;
- `epsilon_wander(t)` is a time-varying deviation from that initial error; and
- the sampling phase is the integral of the rate error over time.

At present, `sampling_clock_impairments()` uses

```text
position_increment = drate * (1 + drift_ppm * 1e-6)
```

and advances by that same amount for every output sample. Therefore the
existing `drift_ppm` is `epsilon0`; despite its name, it does not model
time-varying drift or wander.

The starting position is currently fixed at:

```text
initial_position = uprate / drate
```

This preserves historical filter alignment but is not a random physical clock
phase.

## Proposed model

For output sample `n`, use the nominal polyphase position

```text
p[n] = p_legacy + uprate * tau0
       + n * drate * (1 + epsilon0 * 1e-6)
```

where:

- `p_legacy = uprate / drate` preserves the existing alignment convention;
- `tau0` is the initial phase in input-sample periods; and
- `epsilon0` is the existing signed `drift_ppm` value.

For an asynchronous capture, draw:

```text
tau0 ~ Uniform(0, 1)
```

This gives every fractional position within one input-sample interval equal
probability. It is preferable to a Gaussian phase because sampling phase is
periodic: phases separated by one sample period are equivalent.

The initial rate error should be signed. A simple configurable model is:

```text
epsilon0 ~ Uniform(ppm_min, ppm_max)
```

with a range that crosses zero, such as `(-10, 10)` PPM. The correct numerical
range depends on the oscillator and dataset scenario, so TorchSIG should not
claim that one default represents every receiver. A caller should be able to
provide a fixed value or an explicit range.

## API changes

### Low-level implementations

Add `initial_phase` to both the NumPy and Numba sampling-clock functions:

```python
def sampling_clock_impairments(
    ...,
    drift_ppm: float,
    rng: np.random.Generator | None = None,
    initial_phase: float = 0.0,
) -> np.ndarray:
```

`initial_phase` is measured in input-sample periods and must satisfy
`0.0 <= initial_phase < 1.0`. The initial polyphase position becomes:

```python
initial_position = uprate / drate + uprate * initial_phase
```

The default remains `0.0` so direct functional calls and existing golden
results retain the legacy alignment.

The same resolved initial position must be passed to the compiled Numba kernel
so the NumPy and Numba implementations remain numerically equivalent.

### Functional transform

Extend `clock_drift()` with the same deterministic scalar argument:

```python
def clock_drift(
    data: np.ndarray,
    drift_ppm: float = 10,
    initial_phase: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
```

The functional API should not randomly choose the initial phase. Keeping it
deterministic makes the function directly testable and leaves random parameter
selection to the transform class, consistent with other TorchSIG transforms.

### `ClockDrift` transform

Add a transform-level distribution:

```python
class ClockDrift(SignalTransform):
    def __init__(
        self,
        drift_ppm: tuple[float, float] = (-10, 10),
        initial_phase: tuple[float, float] = (0.0, 1.0),
        drift_sampling: Literal["linear", "log10"] = "linear",
        **kwargs,
    ):
        ...
```

Use `drift_sampling` to select the range distribution explicitly. The previous
log-uniform distribution accepted only positive bounds and therefore could not
represent clocks that run slower than nominal. The new signed `(-10, 10)`
default uses linear sampling. Callers that need the prior magnitude-weighted
behavior can request `drift_ppm=(1, 10), drift_sampling="log10"`.

The tuple form follows TorchSIG's existing distribution convention. A fixed
phase is represented by equal bounds, for example `(0.25, 0.25)`. A bare float
should not be advertised as a fixed transform parameter because the current
distribution helper interprets a float as the upper bound of a uniform
distribution from zero.

The transform now draws a phase from `(0.0, 1.0)` by default. This necessarily
changes seeded transform outputs because TorchSIG's distributions share an
RNG. Callers that require the previous alignment can pass `(0.0, 0.0)`; the
functional API retains `initial_phase=0.0` and therefore remains compatible by
default. An independent parameter RNG would be a broader reproducibility
change and is outside this proposal.

## Relationship to jitter and time-varying drift

`ClockJitter` already models an independent Gaussian displacement at each
sampling instant. Its `jitter_ppm` is an RMS timing displacement relative to
one input-sample period. This is distinct from both initial sampling phase and
sampling-rate error.

A parameter called `sigma` or "phase standard deviation" should not be added
to `ClockDrift` without also specifying its process and time scale. The same
standard deviation can describe very different clocks depending on whether
the error is independent per sample, a random walk, or a correlated process.

If time-varying drift is needed later, add an explicit oscillator-wander model,
for example:

```text
epsilon[n + 1] = rho * epsilon[n]
                 + sqrt(1 - rho^2) * sigma_ppm * z[n]
```

where `z[n]` is standard Gaussian noise, `sigma_ppm` controls steady-state rate
variation, and `rho` or a correlation time controls how quickly the clock
wanders. The sampling position would then integrate this rate error. This
should be introduced as a separate transform or clearly named optional mode,
because it has different semantics and computational cost from constant clock
offset.

## Validation and tests

Add tests covering:

- `initial_phase=0.0` exactly reproduces the current legacy output;
- phase values near `0.0` and just below `1.0` select the expected polyphase
  branches;
- invalid phase values, NaN, and infinity raise `ValueError`;
- fixed positive and negative rate errors change output length in the expected
  directions;
- NumPy and Numba results match for the same phase, drift, jitter, and RNG;
- seeded `ClockDrift` instances draw reproducible initial phases;
- the transform accepts equal phase bounds for deterministic datasets;
- output length and complex dtype remain unchanged at the public transform
  boundary; and
- metadata behavior remains unchanged unless a separate decision is made to
  record the realized impairment parameters.

## Recommended implementation sequence

1. Add deterministic `initial_phase=0.0` support to both low-level
   implementations and the functional transform.
2. Initialize `ClockDrift` with a uniform phase over one sample interval and a
   signed, linearly sampled `(-10, 10)` PPM rate error.
3. Retain an explicit log-sampling option for callers that need the previous
   positive magnitude distribution.
4. Only add time-varying wander after selecting a documented stochastic model
   and parameter units based on a target oscillator or dataset use case.

## Recommendation

Implement random initial sampling phase and signed initial rate offset first.
They address the unrealistic starting state with a small, testable extension
to the current resampler. Do not add an unspecified Gaussian phase term or
call the existing constant offset a complete drift model. If realistic
long-duration oscillator behavior is required, define wander separately with
an explicit correlation time and validate it against a stated device class.
