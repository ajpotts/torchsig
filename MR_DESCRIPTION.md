# Add packed and homogeneous HDF5 dataset backends

## Summary

This MR adds optimized HDF5 storage paths for TorchSig static datasets:

- **Packed HDF5** supports variable top-level shapes and dtypes while reducing
  the object-per-record overhead of the legacy format.
- **Homogeneous HDF5** stores fixed-shape, fixed-dtype top-level observations in
  one native multidimensional dataset for efficient slicing and batch reads.
- Variable component counts, component shapes, component dtypes, and metadata
  remain supported by both optimized formats.
- `DatasetCreator` and the TorchSig data modules infer the matching reader for
  known writer classes and reject incompatible explicit pairings.
- `StaticTorchSigDataset` uses native homogeneous full-signal batch reads for
  contiguous DataLoader batches while preserving metadata, components,
  transforms, and target labels.
- Packed and homogeneous HDF5 share one versioned TorchSig JSON metadata codec.
- Golden files lock reader compatibility with the first stable packed and
  homogeneous schemas.

The existing HDF5 backend remains available for compatibility. Packed HDF5 is
the general-purpose option, while homogeneous HDF5 is an opt-in optimized
backend for datasets whose top-level observations share a shape and dtype.

## Motivation

The legacy object-per-record HDF5 layout incurs substantial per-sample lookup
and reconstruction overhead. This is particularly expensive during model
training, where observations normally have a fixed input shape and are consumed
in batches.

The homogeneous layout enables a contiguous batch to be read with a small
number of native HDF5 operations. The packed layout retains support for
heterogeneous top-level observations without requiring the legacy group-heavy
schema.

## Schema and compatibility

### Packed HDF5

- Uses the frozen format identifier `torchsig-packed` and schema version `1.0`.
- Readers reject unsupported major versions and unknown required features.
- Minor releases may add optional fields or features while remaining readable
  by version 1 readers.
- Allows variable top-level shapes and dtypes.
- Allows variable component counts, shapes, and dtypes.
- Preserves and reconstructs hierarchical parent relationships. Python parent
  object identity is not guaranteed across separate reader calls.
- Suitable as the flexible default for newly generated datasets.

### Homogeneous HDF5

- Uses the frozen format identifier `torchsig-homogeneous` and schema version
  `1`.
- Readers reject other format identifiers and schema versions.
- Requires every top-level observation to have the same shape and dtype.
- Supports IQ, wideband IQ, and spectrogram observations.
- Allows variable component counts, component shapes, and component dtypes.
- Flattens inherited parent metadata into each stored signal. Parent hierarchy
  and parent object identity are not reconstructed when reading.
- Falls back to per-record reads for shuffled or otherwise non-contiguous
  DataLoader batches.

No existing public backend is removed by this MR.

## Performance

The table below reports mean uncompressed timings from
`benchmarks/optional/benchmark_homogeneous_hdf5.py`. Lower is better.

| Operation | Workload | Legacy standard | Packed | Homogeneous | Homogeneous vs. packed |
|---|---|---:|---:|---:|---:|
| Dataset creation | Narrowband IQ | 326.0 ms | **31.5 ms** | 38.5 ms | 18% slower |
| Dataset creation | Wideband IQ | 153.3 ms | 56.9 ms | **34.3 ms** | 1.66x faster |
| Dataset creation | Spectrogram | 165.1 ms | 34.1 ms | **25.3 ms** | 1.35x faster |
| Random reads | Narrowband IQ | 71.5 ms | 19.9 ms | **19.7 ms** | Approximately equal |
| Random reads | Wideband IQ | 30.6 ms | 9.7 ms | **6.4 ms** | 1.52x faster |
| Random reads | Spectrogram | 28.3 ms | 7.9 ms | **6.3 ms** | 1.24x faster |
| Contiguous full-signal batch | Narrowband IQ | 67.7 ms | 17.6 ms | **2.9 ms** | 6.06x faster |
| Contiguous full-signal batch | Wideband IQ | 40.3 ms | 14.2 ms | **2.6 ms** | 5.37x faster |
| Contiguous full-signal batch | Spectrogram | 30.0 ms | 8.4 ms | **1.9 ms** | 4.54x faster |
| Sequential epoch, 0 workers | Narrowband IQ | 568.5 ms | 195.3 ms | **28.6 ms** | 6.83x faster |
| Sequential epoch, 0 workers | Wideband IQ | 274.6 ms | 123.3 ms | **24.4 ms** | 5.06x faster |
| Sequential epoch, 0 workers | Spectrogram | 229.5 ms | 93.3 ms | **15.8 ms** | 5.91x faster |
| Shuffled epoch, 0 workers | Narrowband IQ | 563.5 ms | 195.4 ms | **148.4 ms** | 1.32x faster |
| Shuffled epoch, 0 workers | Wideband IQ | 278.6 ms | 119.9 ms | **59.9 ms** | 2.00x faster |
| Shuffled epoch, 0 workers | Spectrogram | 234.0 ms | 94.1 ms | **46.4 ms** | 2.03x faster |

Key observations:

- Homogeneous contiguous full-signal batches are approximately **4.5-6x
  faster** than packed batches.
- Homogeneous sequential DataLoader epochs are approximately **5-7x faster**
  than packed epochs.
- Homogeneous shuffled epochs remain **1.3-2x faster** than packed, although
  shuffling cannot use contiguous batch slicing.
- Packed is the fastest option for uncompressed narrowband creation.
- The legacy backend is consistently slower for data creation and epoch reads.
- Additional workers did not improve these read-dominated workloads; zero
  workers performed best in the benchmark environment.

The complete optional matrix contains 216 cases covering:

- narrowband IQ, wideband IQ, and spectrogram workloads;
- standard, packed, and homogeneous formats;
- uncompressed and LZF-compressed storage;
- sequential and shuffled access; and
- 0, 2, and 8 DataLoader workers.

Run it with:

```bash
pytest benchmarks/optional/benchmark_homogeneous_hdf5.py --benchmark-only
```

## Validation

- The complete optional benchmark matrix passes: **216 passed**.
- Round-trip tests cover IQ and spectrogram data with variable component
  counts.
- Tests cover threaded dataset creation, DataLoader workers, reader
  inference, incompatible reader/writer rejection, schema validation,
  corruption detection, process-safe reader reopening, and contiguous
  full-signal batch reads.
- Committed packed `1.0` and homogeneous `1` golden files verify compatibility
  without invoking the current writers.
- Shared metadata-codec tests lock the encoded representation and cover NumPy
  values, bytes, complex values, tuples, reserved marker dictionaries, and
  invalid inputs.
- User documentation compares all three HDF5 readers and covers reader
  selection, compatibility, metadata, components, and batch-read APIs.
- Existing standard and packed writer integration tests continue to pass.

## Recommended usage

- Select **homogeneous HDF5** for fixed-size model inputs, especially sequential
  wideband, spectrogram, and IQ training workloads.
- Select **packed HDF5** when top-level observation shapes or dtypes may vary.
- Retain the legacy reader for existing datasets and migration compatibility.
- Prefer `num_workers=0` for read-dominated static datasets unless downstream
  transforms demonstrably benefit from multiprocessing.

## Follow-up work

- Add user-facing configuration examples for IQ, wideband, and spectrogram
  datasets.
- Add a unified reader factory with automatic schema detection.
- Add migration tooling from the legacy object-per-record format.
- After a compatibility period, consider deprecating direct use of the legacy
  writer while retaining legacy read support.
