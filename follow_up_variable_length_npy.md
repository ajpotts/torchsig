# Follow-up: Variable-length packed NPY datasets

## Summary

Investigate and implement efficient random access for multiple variable-length
IQ records stored in an NPY-compatible dataset layout.

The shared audio manifest cannot be applied directly because WAV and OGG expose
frame-oriented seeking, while packed NPY data requires an explicit array and
offset representation. This work should not block variable-length WAV/OGG
support.

## Requested investigation

- Compare these storage layouts:

  - one `.npy` file per record;
  - one concatenated sample array plus an offset/length table;
  - NumPy memory-mapped arrays;
  - `.npz`, with explicit consideration of its limited partial-read behavior.

- Measure shuffled random access, file-descriptor pressure, storage overhead,
  and multi-worker DataLoader behavior.
- Determine how complex dtype, byte order, and record shape should be declared.
- Define migration and compatibility behavior for existing NPY datasets.

## Proposed format

Use a concatenated one-dimensional sample array with a versioned record table
containing:

- record index;
- sample offset;
- sample count;
- dtype identifier or a dataset-wide dtype declaration;
- logical shape when records may not be one-dimensional;
- metadata-row association.

Prefer a memory-mappable, uncompressed data array for predictable partial
access. If compression is required, treat it as a separate format with explicit
random-access tradeoffs.

## Acceptance criteria

- Records may have different lengths without padding.
- Reading one record does not load the complete sample array.
- Offsets, lengths, shapes, dtypes, and bounds are validated before serving
  data.
- Corrupt or overlapping descriptors fail with actionable errors.
- Existing one-record-per-file NPY datasets remain readable through a legacy
  adapter or documented migration tool.
- Reader state is safe under forked and spawned DataLoader workers.
- Deterministic tests cover boundary records, empty/invalid descriptors,
  multiple dtypes if supported, and shuffled multi-worker access.
- Benchmarks compare the selected layout with one-file-per-record NPY and the
  relevant HDF5 reader.

## Out of scope

- WAV/OGG frame seeking and audio codec validation.
- RF/channel augmentation.
- Unrelated changes to embedded HDF5 schemas.
