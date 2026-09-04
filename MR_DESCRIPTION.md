Summary

- Require `HierarchicalMetadataObject` parents to be another
  `HierarchicalMetadataObject` or `None`.
- Detect cycles during inherited metadata lookup and full-metadata collection.
- Raise a documented `MetadataParentCycleError` instead of recursing indefinitely.
- Preserve metadata inheritance and child override behavior for valid hierarchies.

Testing

- `pytest -q tests/utils/test_abstractions.py`
- `pytest -q tests/utils/test_metadata_logging.py`
