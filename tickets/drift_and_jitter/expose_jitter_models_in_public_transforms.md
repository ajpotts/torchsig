Title: Expose sampling-clock jitter models through public transforms

Priority: Medium

Summary
-------

`torchsig.utils.dsp.sampling_clock_impairments` supports independent,
period-accumulating, and AR(1) jitter through `jitter_model` and
`jitter_alpha`. The public `clock_jitter` functional and `ClockJitter` class
currently expose only independent Gaussian timing jitter, leaving the other
implemented models inaccessible through the normal transform API.

Proposed change
---------------

* Add `jitter_model` and `jitter_alpha` to `clock_jitter` while preserving the
  current independent-jitter defaults.
* Add equivalent configuration to `ClockJitter` and pass it through to the
  functional implementation.
* Document the distinction between absolute independent jitter, accumulated
  sampling-period error, and correlated AR(1) jitter.
* Ensure configuration serialization and construction paths accept the new
  parameters where applicable.

Acceptance criteria
-------------------

* Existing callers retain identical default behavior.
* All three jitter models are selectable through both public APIs.
* Invalid model names and AR(1) coefficients outside `(-1, 1)` raise clear
  `ValueError` exceptions.
* Deterministic tests verify parameter forwarding, seeded reproducibility, and
  the expected timing statistics for each model.
* NumPy and Numba-backed execution agree within the existing numerical
  tolerance for every exposed model.
