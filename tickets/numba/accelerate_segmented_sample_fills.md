Title: Accelerate segmented sample-fill transforms with Numba

Priority: Medium
Confidence: Medium-high
Predicted warm speedup: 2x-15x for workloads with many regions

Summary
-------

`drop_samples` and `spectrogram_drop_samples` loop over drop regions in Python
and allocate a temporary array for each region. The overhead becomes material
when transforms contain many short drops, while calls with only one or two
large regions are already dominated by NumPy slice assignment.

Proposed change
---------------

* Add cached Numba kernels for filling validated regions directly into the
  destination arrays without per-region temporary allocations.
* Select the fill behavior outside the inner loop where practical, avoiding
  repeated string dispatch in compiled code.
* Compute global reductions such as mean, minimum, and maximum once per call.
* Preserve in-place mutation, output shape, dtype conversion, overlap order,
  and existing boundary behavior.
* Keep a NumPy fallback for environments without Numba.

Acceptance criteria
-------------------

* All supported fill modes produce results equivalent to the current
  implementations.
* Empty region lists, overlapping regions, adjacent regions, single-sample
  regions, and boundary indices are tested.
* Invalid fill modes retain the existing exception behavior.
* Benchmarks cover 2, 64, and 1,024 regions with both short and long regions.
* The accelerated path improves the 64-region representative workload by at
  least 2x and does not materially regress the two-region workload.
* Cold and warm timings are reported separately.

