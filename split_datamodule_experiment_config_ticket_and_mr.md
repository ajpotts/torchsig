TICKET

Title: Fix SplitTorchSigDataModule tests after experiment configuration forwarding

Summary:
The SplitTorchSigDataModule prepare_data tests fail because their spec-constrained
TorchSigDatasetConfig mocks do not define experiment_config. Split dataset creation
now reads cfg.experiment_config and forwards it to TorchSigIterableDataset, so the
outdated mocks raise AttributeError before any split is created.

Acceptance criteria:
- The shared test config helper initializes experiment_config.
- Tests verify that each split forwards its own experiment_config to
  TorchSigIterableDataset.
- The SplitTorchSigDataModule unit tests pass.

MERGE REQUEST

Title: Update split datamodule mocks for experiment configuration

Description:
This change updates the spec-constrained TorchSigDatasetConfig test helper to
include an experiment_config value, matching the current public config model. It
also extends the split creation test to verify that the train, validation, and
test experiment configurations are forwarded to their corresponding iterable
datasets.

Root cause:
Production split creation began forwarding cfg.experiment_config, but the test
helper continued to initialize only the older set of config attributes. Because
the helper uses MagicMock(spec=TorchSigDatasetConfig), accessing the missing
attribute raised AttributeError.

Testing:
- python -m pytest tests/datasets/test_datamodules.py::test_split_datamodule_prepare_data_creates_all_splits tests/datasets/test_datamodules.py::test_split_datamodule_uses_split_specific_seeds
- python -m pytest tests/datasets/test_datamodules.py -k "split_datamodule and not smoke and not end_to_end"
