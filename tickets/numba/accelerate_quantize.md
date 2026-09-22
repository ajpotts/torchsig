Title: Accelerate quantize with a fused Numba kernel

Priority: High
Confidence: High
Predicted warm speedup: 20x-45x

Summary
-------

`torchsig.transforms.functional.quantize` currently builds quantization and
threshold arrays, locates saturation indices, performs repeated `setdiff1d`
operations, and calls `digitize` separately for the real and imaginary parts.
This creates several full-size temporary arrays and makes multiple passes over
the input.

Proposed change
---------------

* Keep input validation, shape handling, and public API behavior in Python.
* Add a cached, nopython Numba kernel that quantizes and clips the real and
  imaginary components in one pass.
* Preserve the current `floor` and `ceiling` threshold semantics exactly,
  including tie behavior, saturation, scaling, and complex64 output.
* Retain a correct NumPy fallback when Numba is unavailable.
* Add focused benchmarks for 1K, 16K, and 256K complex64 samples.

Acceptance criteria
-------------------

* Both rounding modes match the existing implementation for random, boundary,
  saturated, zero, and exactly-on-threshold inputs.
* One-dimensional and supported multidimensional inputs retain their original
  shape and return complex64 data.
* NaN, infinity, invalid bit count, and invalid rounding-mode behavior remains
  unchanged.
* Tests exercise both the accelerated path and the fallback implementation.
* Warm benchmarks show at least a 10x speedup at 16K samples without a
  regression at 1K samples after compilation.
* Cold-start compilation cost is reported separately from steady-state timing.

