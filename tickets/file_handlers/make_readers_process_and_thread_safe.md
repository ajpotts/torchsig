Title: Standardize process- and thread-safe reader resource handling

Summary

Resource safety differs between handlers. `HomogeneousHDF5Reader` and the
audio handle cache track the process that opened a handle, while legacy and
packed HDF5 readers can retain a handle inherited across a fork. Pickling and
spawn behavior is likewise implemented only by some readers. Audio seek/read
operations on cached handles are not protected against concurrent threads.

Proposed change

- Introduce a shared process-local resource helper for lazy handles.
- Record the opening PID and reopen resources after a process change.
- Remove live handles from pickle state and initialize clean worker state on
  unpickle.
- Make teardown idempotent and clear every handle-derived object.
- Serialize seek/read operations when one audio handle can be used by multiple
  threads, or explicitly reject concurrent threaded use.
- Apply the same behavior to legacy HDF5, packed HDF5, homogeneous HDF5, NPY
  memory maps, WAV, OGG, and future handlers.

Acceptance criteria

- Readers opened in a parent remain usable through forked DataLoader workers.
- Readers can be serialized into spawned DataLoader workers without carrying
  live file handles.
- Parent handles remain usable after worker shutdown.
- Repeated epochs with persistent workers return deterministic data.
- Tests cover zero, one, and multiple workers under every multiprocessing
  start method available on the test platform.
- Threaded audio access is either safe and tested or rejected with an
  actionable error.
- Worker tests do not leak file descriptors.
