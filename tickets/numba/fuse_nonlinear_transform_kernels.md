Title: Evaluate fused Numba kernels for nonlinear signal transforms

Priority: Medium-low
Confidence: Medium
Predicted warm speedup: 1.3x-4x when FFT autoscaling is disabled; 1.1x-2x overall otherwise

Summary
-------

`nonlinear_amplifier`, `nonlinear_amplifier_table`, and
`intermodulation_products` perform multiple full-array operations and allocate
intermediate magnitude, phase, power, interpolation, and polynomial arrays.
Fused Numba kernels may reduce memory traffic. FFT-based power renormalization
cannot benefit from Numba and is expected to limit end-to-end gains.

Proposed change
---------------

* Prototype separate fused kernels for the analytic amplifier, table-based
  amplifier, and odd-order intermodulation calculation.
* Keep FFT windowing, FFTs, argument validation, and public wrappers outside
  the kernels.
* Avoid calculating phase explicitly where equivalent complex arithmetic can
  preserve current numerical behavior.
* Benchmark with autoscaling both enabled and disabled before deciding which
  kernels merit production integration.
* Preserve NumPy fallbacks for optional-Numba installations.

Acceptance criteria
-------------------

* Outputs meet explicitly documented tolerances for complex64 and complex128
  inputs, including zero and near-zero signals.
* Table interpolation preserves endpoint clipping and exact-knot behavior.
* Invalid even-order intermodulation coefficients retain current exceptions.
* Benchmarks cover 1K, 16K, and 256K samples and separate kernel time from FFT
  renormalization time.
* A kernel is integrated only if it provides at least a 1.5x end-to-end gain in
  a representative supported mode.
* Peak temporary memory is compared with the current implementation.

