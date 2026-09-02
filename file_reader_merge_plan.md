# File reader modernization merge plan

## Overview

The file-handler ticket should be delivered as five ordered MRs. The work spans
shared metadata, WAV/OGG indexing, random-access performance, signal-fidelity
policy, and generation tooling. Keeping these concerns separate makes behavior
changes measurable and allows compatibility to be reviewed at each stage.

The shared metadata work applies to WAV, OGG, and NPY readers because they use
`MetadataReader`. WAV and OGG should share audio-segment indexing, partial
reads, handle caching, and fidelity validation. Existing HDF5 readers already
provide indexed access and lazy handles, so they should be used as design
references rather than rewritten.


## MR 1: Versioned, header-based sidecar metadata

### Scope

Update `MetadataReader`, WAV, OGG, NPY, and dataset generators that emit
`metadata.csv`.

### Changes

- Define and document a versioned sidecar metadata schema.
- Read new `metadata.csv` files using their actual headers.
- Resolve required fields by name so column reordering is safe.
- Preserve unknown columns in returned signal metadata.
- Support both the new header-based format and the legacy headerless
  four-column format.
- Detect the format explicitly; do not interpret a malformed header as data.
- Validate duplicate columns, missing required fields, malformed rows,
  duplicate indices, and negative indices.
- Update generators to emit headers.
- Avoid scanning from the beginning of the CSV on every random lookup. Build a
  lightweight row index or retain a parsed metadata table, with the memory and
  performance tradeoff documented.

### `info.json` behavior

- Treat `info.json` as optional supplementary metadata rather than the
  authoritative record index.
- Infer dataset size and class names from `metadata.csv` when possible.
- Use JSON values only when present and consistent.
- Raise on contradictions instead of silently taking the larger size.

### Reader coverage

- WAV: included.
- OGG: included.
- NPY: included.
- SigMF: excluded because it has its own metadata standard.
- HDF5 variants: excluded because their schemas are embedded.

### Tests

- Reordered headers and additional columns.
- Missing and duplicate columns.
- Legacy headerless compatibility.
- Missing, malformed, partial, and contradictory `info.json`.
- Random metadata access near the beginning and end of a large CSV.


## MR 2: Variable-length record manifest for audio datasets

### Scope

Introduce a shared WAV/OGG audio-record layout abstraction.

### Changes

- Replace `file_index + element_offset * num_iq_samples` with explicit record
  descriptors containing:

  - relative file path;
  - `start_frame`;
  - `num_frames`;
  - optional expected sample rate and channel count.

- Build the global record index from metadata rows.
- Permit multiple variable-length records in one file.
- Permit different record counts in different files.
- Validate that paths remain inside the dataset root.
- Validate nonnegative offsets and lengths.
- Verify every record fits inside its audio file.
- Reject overlapping records unless overlap becomes an explicitly supported
  behavior.
- Verify every metadata record resolves to an audio segment.
- Retain a compatibility adapter for fixed-length datasets using
  `num_iq_samples` and `elements_per_file`.
- Extract shared indexing into an internal helper or base used by WAV and OGG.

### Tests

- Variable-length records within one file.
- Uneven record counts across files.
- Nested audio paths.
- Invalid paths, offsets, lengths, overlaps, and truncated files.
- Legacy fixed-length compatibility.
- WAV/OGG parity for an equivalent manifest.

NPY is excluded from this MR because partial access to packed variable-length
NPY records needs a format-specific offset-table design. That work is captured
in a follow-up ticket.


## MR 3: Partial reads and a process-safe audio handle cache

### Scope

Optimize WAV and supported losslessly seekable OGG containers.

### Changes

- Read only each requested `start_frame:num_frames` segment.
- Use bounded `soundfile.SoundFile.seek()` and `read()` operations instead of
  loading and reshaping an entire file.
- Add a configurable, bounded, per-process LRU handle cache.
- Support cache size zero to disable caching.
- Track the owning PID and reopen handles after a process change.
- Never pickle or share live handles with DataLoader workers.
- Implement `setup()`, `teardown()`, context-manager cleanup, and safe pickle
  state.
- Close evicted handles deterministically.
- Limit OGG support to codecs with demonstrated lossless and accurate seeking.

### Benchmark

Measure:

- sequential and shuffled access;
- cache disabled and several bounded cache sizes;
- one and multiple DataLoader workers;
- small and large records;
- whole-file and partial-read implementations;
- WAV and supported OGG against relevant HDF5 readers.

Report bytes read, records per second, and worker memory. Define an acceptable
regression threshold, but do not commit generated benchmark results.

### Tests

- Requested reads use bounded start and frame counts.
- Repeated access reuses a handle.
- LRU eviction closes the old handle.
- `teardown()` closes every handle.
- Pickling excludes handles.
- A PID change causes handles to reopen.
- Shuffled multi-worker reads return the correct records.


## MR 4: Shared audio fidelity validation and optional normalization

### Scope

Apply consistent fidelity rules to WAV and OGG readers.

### Changes

- Require exactly two channels for I/Q audio.
- Validate file sample rate against record and dataset metadata.
- Permit mixed sample rates only when every affected record declares its rate;
  reject conflicts with a dataset-wide rate.
- Verify returned frame counts and finite sample values.
- Detect unsupported or lossy OGG codecs explicitly.
- Add opt-in normalization while preserving existing samples by default:

  - `normalization="none"` by default;
  - documented RMS or peak normalization modes;
  - one complex-valued scale factor applied jointly to I and Q;
  - deterministic behavior for silence;
  - optional metadata recording the applied scale.

Normalization is separate from MR 3 because it changes signal values and
training semantics, while partial reading should be transparent.

### Tests

- Reject mono and greater-than-stereo input.
- Accept matching sample rates and reject inconsistent rates.
- Cover permitted and invalid mixed-rate datasets.
- Reject truncated segments and non-finite samples.
- Verify joint I/Q normalization preserves phase and relative scaling.
- Cover silence and unchanged default behavior.


## MR 5: Deterministic dataset-generator concurrency and logging

### Scope

Update `IQDatasetGenerator` and related generation scripts rather than reader
implementations.

### Changes

- Replace operational `print()` calls with module logging and configurable
  verbosity.
- Separate deterministic record specification from record execution.
- Generate records with a configurable process pool.
- Derive stable per-record seeds without Python's randomized `hash()`.
- Do not share GNU Radio flowgraphs or mutable RNG state between workers.
- Have workers return metadata records to the parent process.
- Write `metadata.csv` and `info.json` once and deterministically in the parent.
- Emit the MR 1 header-based schema and MR 2 record descriptors.
- Write temporary outputs followed by atomic rename so interrupted work cannot
  appear complete.

### Tests

- Serial and multiprocessing output equivalence.
- Repeated seeded runs produce the same record specifications and metadata.
- Worker-count changes do not change output ordering or seeds.
- Worker failures do not publish partial files as complete records.
- Logging honors configured verbosity.


## Merge order and dependencies

1. MR 1: metadata contract and optional `info.json`.
2. MR 2: shared variable-length audio record index; depends on MR 1.
3. MR 3: partial reads, handle cache, and benchmarks; depends on MR 2.
4. MR 4: fidelity validation and opt-in normalization; depends on the shared
   audio abstraction from MR 2 and should follow MR 3 to keep performance and
   signal-value changes independently measurable.
5. MR 5: deterministic parallel generator and logging; depends on the output
   contracts from MRs 1 and 2.

The RF/channel augmentation follow-up can be designed after MR 1, but it should
integrate with MR 5's deterministic record specification before both efforts
are considered complete.


## Follow-up tickets

- `follow_up_rf_channel_augmentation.md`: configurable and reproducible RF and
  channel impairments.
- `follow_up_variable_length_npy.md`: packed variable-length NPY storage and
  memory-mapped random access.
