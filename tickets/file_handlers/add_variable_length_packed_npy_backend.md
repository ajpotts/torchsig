Title: Add a memory-mappable packed NPY backend for variable-length records

Summary

The current `NPYReader` treats first-axis array elements as global records and
opens a memory map on each read. It does not provide a compact representation
for multiple variable-length arrays with independent shapes and metadata.
One-file-per-record layouts also create file-count and file-descriptor pressure
at scale.

Proposed change

- Store samples in one uncompressed, one-dimensional NPY-compatible data
  array.
- Add a versioned record table containing record index, sample offset/count,
  logical shape, dtype declaration, and metadata association.
- Validate ordering, overlap, bounds, shape products, dtype, byte order, and
  metadata references before serving data.
- Reconstruct each record from only its memory-mapped slice.
- Preserve `NPYReader` as the legacy adapter and document migration options.
- Provide a benchmark against per-record NPY and the relevant HDF5 backend.

Acceptance criteria

- Records may have different lengths and shapes without padding.
- Reading one record does not load the complete data array.
- Empty records and boundary records are handled deterministically.
- Corrupt, overlapping, or out-of-bounds descriptors produce actionable
  errors.
- Complex dtype and byte order are preserved exactly.
- Forked and spawned DataLoader workers reopen process-local mappings safely.
- Existing NPY datasets remain readable without conversion.
- Benchmarks report shuffled access, storage overhead, file-descriptor use, and
  single-/multi-worker throughput.
