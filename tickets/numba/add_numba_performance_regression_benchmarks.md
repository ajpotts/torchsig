Title: Add reproducible performance and parity gates for Numba kernels

Priority: High
Confidence: High
Predicted speedup: Not applicable; enables safe optimization work

Summary
-------

TorchSIG already accelerates sampling-clock impairments and digital AGC with
Numba, and several additional transforms are candidates. The repository has
correctness coverage for the existing kernels, but candidate evaluation needs
a consistent way to distinguish compilation cost, warm execution, wrapper
overhead, and end-to-end transform performance.

Proposed change
---------------

* Add a focused optional benchmark module for current and proposed Numba
  kernels.
* Measure pure-Python/NumPy reference, compiled kernel, and public wrapper paths.
* Warm kernels before steady-state measurement and report first-call timing
  separately.
* Cover representative input sizes, dtypes, parameter branches, and workloads.
* Record environment details including Python, NumPy, Numba, CPU, and thread
  settings with benchmark output.

Acceptance criteria
-------------------

* Sampling-clock impairment and digital AGC have reference-versus-kernel and
  wrapper-level benchmark cases.
* Candidate benchmarks can run independently and do not make the normal test
  suite depend on timing thresholds.
* Correctness checks run before timings and reject mismatched outputs.
* Results distinguish cold compilation, warm kernel, and end-to-end wrapper
  costs.
* Benchmark inputs are deterministic and generated outside timed sections.
* Documentation explains how to reproduce and compare results without
  committing generated benchmark artifacts.
