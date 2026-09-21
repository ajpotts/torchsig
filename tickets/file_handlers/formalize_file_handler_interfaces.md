Title: Formalize the FileReader and FileWriter interfaces

Summary

The file-handler base classes currently provide partially implemented concrete
methods rather than enforce a consistent contract. Readers do not share a base
context-manager or close protocol, and individual handlers differ in setup,
teardown, length, bounds, and return-value behavior. `BaseFileHandler` also uses
a static factory that references `BaseFileHandler.reader_class` and
`BaseFileHandler.writer_class`, so subclass class attributes are not honored by
the inherited implementation.

Proposed change

- Define `FileReader` and `FileWriter` as abstract base classes or typed
  protocols.
- Require consistent `read`, `write`, `__len__`, `setup`, and `teardown`
  behavior.
- Add an idempotent reader `close`/`teardown` contract and context-manager
  support at the base level.
- Make the handler factory a class method that resolves reader and writer
  classes from the concrete handler subclass.
- Accept both `str` and `Path` roots consistently.
- Document whether returned arrays are copies or storage-backed views.
- Preserve existing public entry points through compatibility wrappers where
  necessary.

Acceptance criteria

- Incomplete reader and writer subclasses cannot be instantiated silently.
- All built-in readers and writers satisfy the same lifecycle contract.
- Repeated teardown or close operations are safe.
- Context-manager exit closes resources for every reader and writer.
- A `BaseFileHandler` subclass uses its own configured reader and writer
  classes without reimplementing the factory.
- Type checking distinguishes reader and writer return types where practical.
- Existing supported construction patterns remain compatible or have a
  documented migration path.
