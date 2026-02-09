Title: Make file-handler writes transactional and overwrite-safe

Summary

`FileWriter.setup()` currently resets the destination directory before the
write has succeeded. A generation failure can therefore destroy a valid
dataset and leave an incomplete replacement that looks usable. Cleanup also
depends partly on nondeterministic destructor behavior.

Proposed change

- Move overwrite authorization out of the base writer and into dataset
  orchestration.
- Write new datasets into a sibling staging directory.
- Record an explicit incomplete/complete state in each format.
- Flush and close all data before atomically publishing the staging directory.
- Preserve the previous dataset until publication succeeds.
- Remove or minimize reliance on `__del__` for finalization.
- Provide a deliberate recovery or cleanup path for abandoned staging data.

Acceptance criteria

- A failed write cannot replace or damage an existing complete dataset.
- Readers reject incomplete datasets with a clear error.
- Successful publication is atomic on supported local filesystems.
- Overwrite behavior requires an explicit caller decision.
- Exceptions during setup, batch writing, flush, and teardown are covered by
  deterministic tests.
- Temporary artifacts are removed on ordinary failures and are identifiable
  when recovery is requested.
- DatasetCreator behavior remains compatible with every registered writer.
