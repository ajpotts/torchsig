Title: Add a central file-handler registry and capability model

Summary

Reader/writer pair mappings and backend choices are repeated across file-handler
exports, dataset creation, datamodules, and command-line tooling. Adding a new
backend requires coordinated edits in several modules, and an omitted mapping
can silently select an incompatible reader.

Proposed change

Create a registry containing one specification per storage backend. A
specification should include:

- stable backend name and format version;
- reader and optional writer classes;
- accepted configuration options;
- capabilities such as variable shape, mixed dtype, component hierarchy,
  batch reads, compression, memory mapping, and multiprocessing safety;
- compatibility aliases and deprecation metadata.

DatasetCreator, datamodules, YAML configuration, CLI choices, and public helper
APIs should resolve handlers through this registry.

Acceptance criteria

- Each built-in backend is declared exactly once.
- Writer-to-reader resolution is derived from the registry.
- Unknown backend names and unsupported capabilities fail before generation.
- Configuration and CLI backend choices are generated from registered names.
- Third-party code can register a backend without modifying TorchSIG source.
- Duplicate names and incompatible reader/writer pairs are rejected.
- Tests verify discovery, aliases, capability queries, and configuration
  round trips.
