# Sampling-clock MR merge order

Merge these branches in order. Each branch targets the branch immediately above it.

1. `414-sampling-positions-refactor`
2. `414-interpolate-subphase-sampling-positions`
3. `414-align-zero-impairment-output`
4. `414-replace-symmetric-clock-output-padding`
5. `414-define-clock-resampler-bandwidth-contract`
6. `414-correct-clock-transform-documentation`
7. `414-clean-up-clock-resampler-fallback`
8. `414-configurable-jitter-models`
9. `414-expose-jitter-models-in-transforms`

The configurable jitter models intentionally come after interpolation,
alignment, boundary handling, filter-contract documentation, and fallback
cleanup so new stochastic behavior is built on a corrected resampler.
