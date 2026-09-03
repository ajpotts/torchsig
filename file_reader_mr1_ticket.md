# Ticket: Add a versioned, header-based metadata sidecar schema

## Problem

TorchSIG file readers depend on positional, headerless `metadata.csv` rows and
repeatedly scan the file for random lookups. Column changes are unsafe, extra
metadata is discarded, and shuffled access scales poorly. Readers can also
silently hide contradictions between `info.json`, CSV records, and stored
waveforms.

## Requested change

Define sidecar schema version 1.0 with required `index`, `label`, `modcod`, and
`sample_rate` columns. Resolve header-based fields by name, preserve unknown
columns, and retain compatibility with legacy four-column and historical
six-column headerless files. Validate schemas, rows, numeric fields, and record
indices when constructing the reader.

Treat `info.json` as optional supplementary metadata. Infer record count and
class names from CSV when possible, reject contradictions, and cache parsed
metadata for O(1) row access. Update GNU Radio generators to emit the versioned
header-based format.

Apply this behavior to WAV, OGG, and NPY. HDF5 is out of scope because its
schema is embedded. Keep SigMF functional because its current reader inherits
the shared metadata reader.

## Acceptance criteria

- Reordered required columns and additional columns are read correctly.
- Unknown columns are returned in signal metadata.
- Legacy four-column and historical six-column rows remain readable.
- Invalid headers, malformed rows, duplicate IDs, and negative IDs fail early.
- Missing, malformed, or partial `info.json` does not prevent CSV-backed reads.
- JSON/CSV and CSV/storage size contradictions raise clear errors.
- Size and class names can be inferred from `metadata.csv`.
- Random metadata lookup is O(1) after construction.
- Updated generators write a header and schema version.
- Relevant file-handler tests pass.
