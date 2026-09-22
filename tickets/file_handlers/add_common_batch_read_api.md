Title: Add a common batch-read API to file handlers

Summary

The general reader interface exposes only `read(index)`, even when a backend
can serve contiguous or grouped records much more efficiently. Homogeneous
HDF5 already has specialized batch methods, but datasets and dataloaders cannot
use a portable batch-read capability.

Proposed change

- Define a common `read_batch(indices)` contract.
- Specify ordering, duplicate-index, empty-batch, bounds, metadata, component,
  and copy/view semantics.
- Provide a correct scalar-read fallback in the base interface.
- Implement optimized paths for homogeneous and packed HDF5 and packed NPY.
- Group audio and legacy NPY requests by backing file to reduce seeks and
  reopen operations.
- Allow datasets or dataloaders to use batch reads when the selected backend
  advertises the capability.

Acceptance criteria

- `read_batch(indices)` produces results equivalent to repeated `read` calls
  in the requested order.
- Empty, duplicate, shuffled, contiguous, and invalid index collections are
  tested.
- Backends without an optimized implementation use a correct fallback.
- Optimized implementations preserve metadata and component-signal behavior.
- Benchmarks demonstrate the effect on representative sequential and shuffled
  workloads.
- Existing scalar reads remain unchanged.
