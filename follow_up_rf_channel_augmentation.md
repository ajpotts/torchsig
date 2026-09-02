# Follow-up: Configurable RF and channel augmentation

## Summary

Add configurable frequency offset, phase noise, fading, and related RF/channel
effects to IQ dataset generation without coupling signal modeling to file
reader behavior.

This is intentionally separate from the file-reader modernization MRs. Readers
should return stored samples faithfully; augmentation changes the modeled
signal and requires explicit distributions, ordering, provenance, and
reproducibility guarantees.

## Requested investigation

- Audit existing TorchSIG transforms for frequency offset, phase noise, fading,
  and other requested impairments.
- Reuse existing transforms where their semantics match instead of duplicating
  them in GNU Radio generation code.
- Decide whether each impairment belongs at generation time, training-time
  loading, or supports both modes.
- Define parameter units, distributions, valid ranges, and transform ordering.
- Define normalization order relative to impairments and noise injection.

## Proposed changes

- Add a declarative impairment configuration accepted by
  `IQDatasetGenerator` or a reusable generation pipeline.
- Derive all random choices from a stable per-record seed.
- Record the sampled impairment parameters and transform order in record
  metadata.
- Make the disabled configuration preserve current output.
- Keep reader APIs free of implicit augmentation.
- Integrate with the deterministic record specifications introduced by the
  generator-concurrency MR.

## Acceptance criteria

- Frequency offset, phase noise, and fading can be independently enabled and
  configured.
- The same global configuration and record identity produce identical samples
  and metadata regardless of worker count.
- Metadata reports the actual sampled parameters used for every record.
- Transform ordering is documented and tested.
- Disabled impairments preserve existing generator behavior.
- Invalid parameters fail before generation begins.
- Tests cover each impairment independently and representative combinations.
- Statistical tests use deterministic seeds and tolerances appropriate to the
  modeled effect.

## Out of scope

- File seeking, handle caching, and sidecar parsing.
- Implicit augmentation performed by file readers.
- Model-training policy beyond exposing reproducible generated data.
