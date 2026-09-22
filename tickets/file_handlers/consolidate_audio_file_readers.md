Title: Consolidate WAV and OGG readers behind a shared audio reader

Summary

`WAVReader` and `OGGReader` duplicate file discovery, legacy-layout inference,
manifest construction, index validation, normalization, metadata loading,
context management, and read behavior. Differences are primarily file suffix
and codec policy. Maintaining separate copies makes fixes likely to drift.

Proposed change

- Extract a shared audio-backed IQ reader with configurable suffixes and codec
  validation policy.
- Keep `WAVReader` and `OGGReader` as small public compatibility subclasses.
- Centralize legacy layout inference and explicit-manifest handling.
- Reuse one process-local handle-cache implementation.
- Make startup validation configurable while retaining strict validation by
  default.
- Ensure normalization and recorded scale metadata behave identically across
  lossless formats.

Acceptance criteria

- WAV and OGG readers contain no duplicated indexing or read-path logic.
- Existing constructor arguments and public reader names remain supported.
- Lossy OGG content remains rejected according to the documented IQ fidelity
  policy.
- Nested paths, variable record lengths, normalization, and legacy layouts
  behave consistently.
- Shared contract tests run against both concrete readers.
- File discovery and validation errors identify the offending path and record.
