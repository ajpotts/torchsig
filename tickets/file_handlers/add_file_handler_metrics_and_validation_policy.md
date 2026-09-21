Title: Add file-handler metrics and configurable validation policy

Summary

Backend performance and resource behavior are difficult to compare without
instrumentation. Strict startup validation is valuable but can be expensive
for large HDF5 and audio datasets, while cached validation can become stale if
its invalidation rules are incomplete.

Proposed change

- Add optional metrics for bytes read and written, records served, cache hits
  and misses, open handles, validation duration, and read latency.
- Keep metrics disabled or low-overhead by default.
- Define validation policies such as `strict`, `cached`, and `lazy`, with
  strict validation remaining the default for untrusted or newly created data.
- Use robust file fingerprints for cached validation and document their
  guarantees.
- Expose metrics in benchmark output without coupling handlers to a particular
  logging framework.
- Add performance regression benchmarks for representative layouts.

Acceptance criteria

- Enabling metrics does not change returned data or metadata.
- Metrics are process-local or safely aggregated with documented semantics.
- Strict validation detects all corruption covered by existing tests before
  data is served.
- Cached validation is invalidated after relevant files or schemas change.
- Lazy validation never bypasses bounds checks required for safe reads.
- Benchmark reports can compare throughput, startup cost, storage overhead,
  and file-descriptor pressure across registered backends.
- The default configuration does not add a material read-performance penalty.
