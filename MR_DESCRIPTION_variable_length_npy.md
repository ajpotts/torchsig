# Merge request description

## Summary

Add a versioned packed NPY backend for variable-length TorchSig records.

This MR introduces `PackedNPYWriter` and `PackedNPYReader`. The format stores a
dataset-wide, uncompressed `data.npy` stream alongside memory-mappable record
and shape tables. Each descriptor records its index, sample offset/count,
logical shape, and metadata association. The manifest declares the format
version, record count, exact NumPy dtype, and byte order.

The reader validates descriptor ordering, overlap, data and shape bounds,
shape/length consistency, metadata indices, dtype, and manifest consistency
before serving samples. Reads return only the selected memory-mapped slice and
reopen process-local mappings under forked or spawned DataLoader workers.

Existing `NPYReader` behavior is unchanged, so legacy one-file-per-record NPY
datasets remain readable without migration. A benchmark script compares
shuffled random access, descriptor/file overhead, file-descriptor changes, and
single-/multi-worker DataLoader throughput against per-record NPY and packed
HDF5 layouts.

## Testing

Tests cover variable and empty lengths, boundary records, multidimensional
shapes, little- and big-endian complex data, real data, corrupt and overlapping
descriptors, invalid bounds and metadata associations, writer failures, legacy
reader availability, handler-pair integration, and deterministic shuffled
access with forked and spawned DataLoader workers.
