"""Unit Tests for datamodules"""
import random
from pathlib import Path

import pytest
import torch

from torchsig.datasets.datamodules import TorchSigDataModule
from torchsig.utils.defaults import TorchSigDefaults
from torchsig.utils.writer import identity_collate_fn


@pytest.mark.filterwarnings(r"ignore:.*fork\(\) may lead to deadlocks in the child:DeprecationWarning")
@pytest.mark.parametrize(
    "num_workers, overwrite",
    [
        (None, True),  # single worker with overwrite (should create dataset files on disk)
        (None, False), # single worker with no overwrite (should not error, just skip creation)
        (2, True),     # multiworker with overwrite (should create dataset files on disk)
        (3, False),    # multiworker with no overwrite (should not error, just skip creation)
    ],
)
def test_TorchSigDataModule_smoke_and_disk_artifacts(tmp_path, num_workers, overwrite):
    # tests that TorchSigDataModule can prepare data and set up dataloaders without error, and that
    # it creates dataset files on disk after prepare_data/setup (multiworker)
    metadata = TorchSigDefaults().default_dataset_metadata

    dm = TorchSigDataModule(
        root=tmp_path,
        metadata=metadata,
        dataset_size=16,
        overwrite=overwrite,
        impairment_level=0,
        collate_fn=identity_collate_fn,
        num_workers=num_workers,
    )
    dm.prepare_data()
    dm.setup()

    assert dm.impairment_level == 0

    dls = [dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()]
    for dl in dls:
        for _batch_idx, data in enumerate(dl): # check all batches (multiworker issues)
            assert hasattr(data, "__len__")
            print(len(data))


    # At least ensure dataset directory is populated after prepare_data/setup.
    # (Exact filenames depend on file handler; HDF5 usually uses data.h5.)
    assert any(tmp_path.iterdir())


def _first_n_batches(loader, n: int):
    """Return up to ``n`` batches from ``loader``.
    If the loader has fewer than ``n`` batches (e.g. because the split is tiny),
    the function simply returns the available ones instead of raising StopIteration.
    """
    it = iter(loader)
    batches = []
    for _ in range(n):
        try:
            batches.append(next(it))
        except StopIteration:
            break
    return batches

def _tensors_identical(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Return ``True`` iff *both* tensors contain exactly the same samples,
    irrespective of their shape.

    * If the shapes differ → they cannot be identical → ``False``.
    * If the shapes are the same → we compare element-wise with a tolerant
      ``torch.allclose`` (the default tolerance is fine for the synthetic
      signals used in the test suite).

    The function works for 2-D tensors of shape ``(N, …)`` where the first
    dimension is the *batch* (i.e. the number of samples).
    """
    # Different number of samples → definitely not identical.
    if a.shape != b.shape:
        return False

    # Same shape → do a normal allclose check.
    return torch.allclose(a, b)

@pytest.mark.parametrize("num_workers", [0, 1, 2])
def test_dataloader_reproducibility(tmp_path: Path, num_workers: int):
    # -----------------------------------------------------------------
    #    Make the *global* Python RNG deterministic (this influences
    #    the *order* of the shuffling RNG inside DataLoader when `shuffle=True`).
    # -----------------------------------------------------------------
    random.seed(0)

    # -----------------------------------------------------------------
    #    Common configuration for all three DataModule instances
    # -----------------------------------------------------------------
    metadata = TorchSigDefaults().default_dataset_metadata
    dataset_size = 10                     # tiny dataset -- fast to generate
    shared_kwargs = {
        "root": tmp_path,
        "metadata": metadata,
        "dataset_size": dataset_size,
        "dataset_splits": [0.6, 0.2, 0.2],   # explicit train/val/test fractions
        "batch_size": 2,
        "num_workers": num_workers,          # the “problematic” setting we want to test
        "seed": 42,
        "collate_fn": lambda x: x,           # identity -- we only care about the raw Signal objects
    }

    # -----------------------------------------------------------------
    #    Create the on-disk dataset **once** (deterministic because we pass seed=42)
    # -----------------------------------------------------------------
    dm_create = TorchSigDataModule(
        **shared_kwargs,
        create_num_workers=0,
    )
    dm_create.prepare_data()

    # -----------------------------------------------------------------
    #     Build **two** independent DataModules that will read the SAME
    #     on-disk files.  All random seeds are the same, so every step
    #     (split, shuffle, worker-level RNG) must be identical.
    # -----------------------------------------------------------------
    dm1 = TorchSigDataModule(**shared_kwargs)
    dm2 = TorchSigDataModule(**shared_kwargs)

    # -----------------------------------------------------------------
    #    Initialise the splits (random_split uses a seeded Generator)
    # -----------------------------------------------------------------
    dm1.setup()
    dm2.setup()

    # -----------------------------------------------------------------
    #    Grab the first 2 batches (4 samples) from each loader
    # -----------------------------------------------------------------
    train_batches_1 = _first_n_batches(dm1.train_dataloader(), 2)
    train_batches_2 = _first_n_batches(dm2.train_dataloader(), 2)

    val_batches_1   = _first_n_batches(dm1.val_dataloader(),   2)
    val_batches_2   = _first_n_batches(dm2.val_dataloader(),   2)

    test_batches_1  = _first_n_batches(dm1.test_dataloader(),  2)
    test_batches_2  = _first_n_batches(dm2.test_dataloader(),  2)

    # -----------------------------------------------------------------
    #    Convert the list of `Signal` objects into a single Tensor.
    #    (All Signals contain a NumPy array under the `.data` attribute.)
    # -----------------------------------------------------------------
    def _signals_to_tensor(batches):
        # `batches` is a list of batch-lists, e.g. [[sig0, sig1], [sig2, sig3]]
        flat = [sig for batch in batches for sig in batch]
        return torch.stack([torch.from_numpy(sig.data) for sig in flat])

    data_train_1 = _signals_to_tensor(train_batches_1)
    data_train_2 = _signals_to_tensor(train_batches_2)

    data_val_1   = _signals_to_tensor(val_batches_1)
    data_val_2   = _signals_to_tensor(val_batches_2)

    data_test_1  = _signals_to_tensor(test_batches_1)
    data_test_2  = _signals_to_tensor(test_batches_2)

    # -----------------------------------------------------------------
    #    Assertions -- each pair must be *exactly* equal.
    # -----------------------------------------------------------------
    assert torch.allclose(data_train_1, data_train_2), "TRAIN split not reproducible"
    assert torch.allclose(data_val_1,   data_val_2),   "VAL   split not reproducible"
    assert torch.allclose(data_test_1,  data_test_2),  "TEST  split not reproducible"

    # T be extra-sure that the *order* of the splits is the same,
    # check that the concatenation of the three tensors
    # reproduces the original full dataset:
    full_1 = torch.cat([data_train_1, data_val_1, data_test_1])
    full_2 = torch.cat([data_train_2, data_val_2, data_test_2])
    assert torch.allclose(full_1, full_2), "Full-dataset ordering mismatched"

    # -----------------------------------------------------------------
    #  Verify that the three splits are **not** identical.
    #  The helper works even if the tensors have different lengths.
    # -----------------------------------------------------------------
    assert not _tensors_identical(data_train_1, data_test_1), "TRAIN and TEST should differ"
    assert not _tensors_identical(data_val_1,   data_test_1), "VAL   and TEST should differ"
    assert not _tensors_identical(data_val_1,   data_train_1), "VAL   and TRAIN should differ"
