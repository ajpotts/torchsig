Title: Add cross-backend validation, errors, and conformance tests

Summary

Handlers currently differ in index validation, incomplete-file handling,
schema validation, error types, cleanup behavior, metadata fidelity, and array
ownership. Backend-specific tests cover many cases, but there is no reusable
suite that enforces a common contract.

Proposed change

- Define public exception types for invalid indices, incomplete datasets,
  unsupported schemas, corrupt storage, and incompatible options.
- Standardize validation timing and error context, including dataset path,
  record index, and failing descriptor or field.
- Build parameterized conformance tests driven by the handler registry.
- Exercise lifecycle, round trips, bounds, corruption, dtype and shape
  preservation, metadata, components, incomplete writes, multiprocessing, and
  resource cleanup according to advertised capabilities.
- Document copy versus storage-backed-view behavior.

Acceptance criteria

- Every registered built-in backend runs the applicable conformance cases.
- Equivalent failures use the same public exception category.
- Errors contain enough context to locate the corrupt file or record.
- Readers never serve data after mandatory integrity validation fails.
- Teardown is idempotent and verified not to leak handles.
- Capability-specific skips are derived from registry declarations rather than
  hard-coded backend names.
- Existing detailed backend tests remain available for format-specific cases.
