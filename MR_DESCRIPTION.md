# Add configurable sampling-clock drift models

## Summary

- Add selectable constant, linear, and random-walk sampling-period drift models.
- Separate signed initial clock offset from deterministic drift rate and stochastic innovation strength.
- Scale time-varying models using the physical output sample rate and preserve polyphase-position alignment.
- Expose the new settings through DSP, functional, accelerated, and `ClockDrift` transform interfaces.
- Document ppm units, sign convention, configuration, and the seeded-output compatibility change.
- Add deterministic, statistical, validation, parity, and diagnostic-plot coverage.

## Testing

- `pytest -q tests/utils/test_sampling_positions.py tests/utils/test_sampling_clock_validation.py tests/utils/test_sampling_clock_nominal_increment.py tests/utils/test_dsp_numba.py tests/utils/test_sampling_clock_initial_phase.py tests/utils/test_sampling_clock_out_of_range.py tests/utils/test_sampling_clock_output_capacity.py tests/utils/test_polyphase_resample_at_positions.py tests/transforms/test_functional.py tests/transforms/test_transforms.py` — 317 passed, 4 skipped.
- `ruff check --select E4,E9,F ...` and script-specific `B,INP` checks — passed.

The full repository suite was also started and reached 49% without failures,
but was stopped after several minutes in a long-running test; the focused
sampling-clock and transform suites above completed successfully.
