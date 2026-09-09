Title: Accelerate time-varying-noise envelope generation with Numba

Priority: Medium
Confidence: Medium-high
Predicted warm speedup: 1.1x-4x end to end; up to 17x for envelope construction

Summary
-------

`time_varying_noise` constructs its power envelope with a Python loop and one
`linspace` allocation per interval. Isolated measurements show little benefit
for four intervals but approximately 2x for 64 intervals and 17x for 1,024
intervals. Random-number generation remains a significant uncompiled portion
of the complete transform.

Proposed change
---------------

* Add a cached Numba kernel that fills the piecewise-linear power envelope
  directly from the inflection indices.
* Leave selection of random inflections and random-noise generation in the
  Python wrapper so seeded generator behavior remains stable.
* Evaluate whether fusing envelope-to-linear conversion and noise application
  provides a measurable additional benefit without changing numerical results.
* Preserve a NumPy fallback.

Acceptance criteria
-------------------

* Accelerated and fallback envelopes agree for zero, one, many, random, and
  evenly spaced inflections.
* Endpoint behavior and alternating low/high direction match the current
  `np.linspace` implementation.
* Seeded calls remain reproducible and consume random values in the same order.
* Output shape and complex64 dtype remain unchanged.
* Benchmarks report envelope-only and end-to-end results for 4, 64, and 1,024
  intervals.
* The change is retained only if end-to-end workloads with many intervals show
  a useful improvement and small-interval workloads do not materially regress.

