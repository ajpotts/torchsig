Title: Consolidate sampling-clock transform wrapper setup

Priority: Low

Summary
-------

`clock_drift` and `clock_jitter` duplicate the input conversion, fixed
upsampling-rate setup, prototype-filter construction and conversion,
impairment invocation, filter-delay calculation, and aligned output handling.
This makes it easier for validation, precision, or alignment behavior to
diverge as the clock models evolve.

Proposed change
---------------

* Extract a private helper for the shared resampler setup and aligned output
  handling.
* Keep model-specific parameters and public docstrings in their existing
  wrappers.
* Preserve public function signatures, defaults, RNG behavior, output length,
  and output dtype.
* Do not combine the conceptual drift and jitter APIs or otherwise broaden the
  refactor.

Acceptance criteria
-------------------

* The duplicated filter construction and alignment logic has a single private
  implementation.
* Existing public APIs and default behavior are unchanged.
* Seeded before/after regression fixtures demonstrate equivalent outputs for
  zero and nonzero drift and jitter.
* Tests cover output length, complex dtype, initial phase, and both NumPy and
  Numba-backed execution.
