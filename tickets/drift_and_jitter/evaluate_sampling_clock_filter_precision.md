Title: Evaluate sampling-clock prototype filter precision

Priority: Low

Summary
-------

The sampling-clock prototype is designed in float64 for a nominal 120 dB
stopband, but `clock_drift` and `clock_jitter` cast its coefficients to
`TorchSigRealDataType` (currently float32) before interpolation. This can limit
deep-stopband fidelity and may make the stated design attenuation stronger
than the realized transform provides.

Proposed investigation
----------------------

* Measure the end-to-end float32 and float64 passband error, image rejection,
  and stopband attenuation for the production `uprate=5000` configuration.
* Measure runtime and memory costs for the NumPy and Numba paths.
* Decide whether to retain float32, use float64 coefficients internally, lower
  the documented attenuation claim, or make precision configurable.
* Preserve complex-valued input and output dtype contracts unless an explicit
  API change is approved.

Acceptance criteria
-------------------

* Benchmarks report realized frequency response and execution cost for both
  coefficient precisions.
* The documented attenuation and occupied-bandwidth contract match the
  production transform, not only the uncast prototype.
* Regression tests cover passband-edge response and representative stopband or
  image-rejection points after the production dtype conversion.
* Any precision change is applied consistently to NumPy and Numba paths and
  their outputs remain within a documented tolerance.
