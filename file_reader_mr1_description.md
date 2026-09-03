# MR: Add versioned metadata sidecars and indexed row access

## Summary

This MR introduces the version 1.0 header-based `metadata.csv` contract while
preserving compatibility with valid headerless datasets. `MetadataReader` now
parses and validates the sidecar once, resolves columns by name, preserves
additional fields, and serves random lookups from an in-memory table.

`info.json` is optional supplementary metadata. CSV rows determine dataset
size and can supply class names when JSON is absent or incomplete. Explicit
schema versions are validated, and contradictions between JSON, CSV, and the
physical reader layout raise errors instead of silently expanding the dataset.

## Changes

- Add sidecar schema version 1.0 and required named columns.
- Support reordered headers, extra fields, legacy four-column rows, and
  historical six-column rows.
- Validate headers, row widths, required values, numeric fields, and indices.
- Retain a parsed metadata table for O(1) row access.
- Reconcile WAV, OGG, NPY, and inherited SigMF sizes with physical layouts.
- Update GNU Radio generators to write headers and schema version metadata.
- Add tests for compatibility, validation, inference, contradictions, and
  access near both ends of a large CSV.

## Compatibility and tradeoffs

Valid headerless datasets remain readable. The parsed table consumes memory
proportional to the metadata file; this is an intentional tradeoff for
predictable shuffled-access performance. Datasets with duplicate IDs or
inconsistent sidecars now fail during construction.

## Validation

```bash
python -m py_compile examples/scripts/generate_gnuradio_examples.py \
    examples/scripts/generate_gnuradio_wav_files_v2.py
pytest -q tests/utils/file_handlers
```

Result: `208 passed, 2 skipped`.
