Title: Evaluate Numba acceleration or caching for SRRC tap generation

Priority: Low
Confidence: Medium for kernel speedup, low for application-level impact
Predicted warm speedup: 3x-15x kernel-only

Summary
-------

`torchsig.utils.dsp.srrc_taps` uses a scalar Python loop containing several
trigonometric operations. It is compatible with a nopython kernel, but typical
tap arrays are short, so dispatch and compilation costs may outweigh the saved
time. Reusing cached tap arrays may provide more value than JIT compilation.

Proposed investigation
----------------------

* Benchmark current tap generation using parameter ranges exercised by the
  signal builders.
* Compare the current implementation with a cached Numba scalar kernel and a
  vectorized NumPy implementation.
* Separately evaluate memoizing immutable tap arrays for repeated parameter
  combinations.
* Do not add a production kernel unless builder-level benchmarks demonstrate a
  measurable improvement.

Acceptance criteria
-------------------

* Implementations agree at discontinuity points and across representative
  samples-per-symbol, span, and alpha values.
* Returned dtype and normalization behavior remain unchanged.
* Benchmarks include cold JIT cost, warm kernel cost, cache-hit cost, and an
  end-to-end modulation workload.
* Any cache has a documented size policy and cannot be mutated through returned
  arrays.
* Production changes are made only if end-to-end modulation improves by at
  least 5% in a representative repeated-generation workload.

