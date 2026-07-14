"""Unit Tests for datamodules"""
import random
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from torchsig.datasets.datamodules import _seed_worker, set_global_seed, TorchSigDataModule
from torch.utils.data import Subset
from torchsig.utils.defaults import TorchSigDefaults
from torchsig.utils.writer import identity_collate_fn


@pytest.mark.parametrize("overwrite", [True, False])
def test_torchsig_datamodule_smoke_and_disk_artifacts(tmp_path, overwrite):
    fft_size = 64
    num_iq_samples_dataset = fft_size**2

    metadata = TorchSigDefaults().default_dataset_metadata.copy()
    metadata.update(
        {
            "num_iq_samples_dataset": num_iq_samples_dataset,
            "fft_size": fft_size,
            "fft_stride": fft_size,
            "num_signals_min": 1,
            "num_signals_max": 1,
            "signal_duration_in_samples_min": int(0.8 * num_iq_samples_dataset),
            "signal_duration_in_samples_max": num_iq_samples_dataset,
        }
    )

    dm = TorchSigDataModule(
        root=tmp_path,
        metadata=metadata,
        dataset_size=12,
        overwrite=overwrite,
        impairment_level=0,
        collate_fn=identity_collate_fn,
        num_workers=0,
    )

    dm.prepare_data()
    dm.setup()

    assert dm.impairment_level == 0
    assert any(tmp_path.iterdir())

    for dataloader in [
        dm.train_dataloader(),
        dm.val_dataloader(),
        dm.test_dataloader(),
    ]:
        assert len(dataloader.dataset) > 0
        batch = next(iter(dataloader))
        assert hasattr(batch, "__len__")

@pytest.mark.slow_no_gpu
@pytest.mark.filterwarnings(
    r"ignore:.*fork\(\) may lead to deadlocks in the child:DeprecationWarning"
)
@pytest.mark.parametrize("num_workers", [2])
def test_torchsig_datamodule_multiworker_smoke(tmp_path, num_workers):
    fft_size = 64
    num_iq_samples_dataset = fft_size**2

    metadata = TorchSigDefaults().default_dataset_metadata.copy()
    metadata.update(
        {
            "num_iq_samples_dataset": num_iq_samples_dataset,
            "fft_size": fft_size,
            "fft_stride": fft_size,
            "num_signals_min": 1,
            "num_signals_max": 1,
            "signal_duration_in_samples_min": int(0.8 * num_iq_samples_dataset),
            "signal_duration_in_samples_max": num_iq_samples_dataset,
        }
    )

    dm = TorchSigDataModule(
        root=tmp_path,
        metadata=metadata,
        dataset_size=8,
        overwrite=True,
        impairment_level=0,
        collate_fn=identity_collate_fn,
        num_workers=num_workers,
    )

    dm.prepare_data()
    dm.setup()

    batch = next(iter(dm.train_dataloader()))

    assert hasattr(batch, "__len__")
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

@pytest.mark.parametrize("num_workers", [0])
def test_dataloader_reproducibility(tmp_path: Path, num_workers: int):
    """DataModules with the same seed should produce matching split batches."""
    random.seed(0)

    fft_size = 64
    num_iq_samples_dataset = fft_size**2

    metadata = TorchSigDefaults().default_dataset_metadata.copy()
    metadata.update(
        {
            "num_iq_samples_dataset": num_iq_samples_dataset,
            "fft_size": fft_size,
            "fft_stride": fft_size,
            "num_signals_min": 1,
            "num_signals_max": 1,
            "signal_duration_in_samples_min": int(0.8 * num_iq_samples_dataset),
            "signal_duration_in_samples_max": num_iq_samples_dataset,
        }
    )

    shared_kwargs = {
        "root": tmp_path,
        "metadata": metadata,
        "dataset_size": 10,
        "dataset_splits": [0.6, 0.2, 0.2],
        "batch_size": 2,
        "num_workers": num_workers,
        "seed": 42,
        "collate_fn": lambda x: x,
    }

    dm_create = TorchSigDataModule(
        **shared_kwargs,
        create_num_workers=0,
    )
    dm_create.prepare_data()

    dm1 = TorchSigDataModule(**shared_kwargs)
    dm2 = TorchSigDataModule(**shared_kwargs)

    dm1.setup()
    dm2.setup()

    def _signals_to_tensor(batches):
        signals = [signal for batch in batches for signal in batch]
        return torch.stack([torch.from_numpy(signal.data) for signal in signals])

    def _first_batch_tensor(dataloader):
        return _signals_to_tensor(_first_n_batches(dataloader, 1))

    train_1 = _first_batch_tensor(dm1.train_dataloader())
    train_2 = _first_batch_tensor(dm2.train_dataloader())

    val_1 = _first_batch_tensor(dm1.val_dataloader())
    val_2 = _first_batch_tensor(dm2.val_dataloader())

    test_1 = _first_batch_tensor(dm1.test_dataloader())
    test_2 = _first_batch_tensor(dm2.test_dataloader())

    assert torch.allclose(train_1, train_2), "TRAIN split not reproducible"
    assert torch.allclose(val_1, val_2), "VAL split not reproducible"
    assert torch.allclose(test_1, test_2), "TEST split not reproducible"


@pytest.mark.slow_no_gpu
@pytest.mark.parametrize("num_workers", [1, 2])
def test_dataloader_reproducibility_multiprocess(tmp_path: Path, num_workers: int):
    """Multiprocess dataloaders should also be reproducible."""
    random.seed(0)

    fft_size = 64
    num_iq_samples_dataset = fft_size**2

    metadata = TorchSigDefaults().default_dataset_metadata.copy()
    metadata.update(
        {
            "num_iq_samples_dataset": num_iq_samples_dataset,
            "fft_size": fft_size,
            "fft_stride": fft_size,
            "num_signals_min": 1,
            "num_signals_max": 1,
            "signal_duration_in_samples_min": int(0.8 * num_iq_samples_dataset),
            "signal_duration_in_samples_max": num_iq_samples_dataset,
        }
    )

    shared_kwargs = {
        "root": tmp_path,
        "metadata": metadata,
        "dataset_size": 10,
        "dataset_splits": [0.6, 0.2, 0.2],
        "batch_size": 2,
        "num_workers": num_workers,
        "seed": 42,
        "collate_fn": lambda x: x,
    }

    TorchSigDataModule(
        **shared_kwargs,
        create_num_workers=0,
    ).prepare_data()

    dm1 = TorchSigDataModule(**shared_kwargs)
    dm2 = TorchSigDataModule(**shared_kwargs)

    dm1.setup()
    dm2.setup()

    def _signals_to_tensor(batches):
        signals = [signal for batch in batches for signal in batch]
        return torch.stack([torch.from_numpy(signal.data) for signal in signals])

    def _first_batch_tensor(dataloader):
        return _signals_to_tensor(_first_n_batches(dataloader, 1))

    assert torch.allclose(
        _first_batch_tensor(dm1.train_dataloader()),
        _first_batch_tensor(dm2.train_dataloader()),
    )
    assert torch.allclose(
        _first_batch_tensor(dm1.val_dataloader()),
        _first_batch_tensor(dm2.val_dataloader()),
    )
    assert torch.allclose(
        _first_batch_tensor(dm1.test_dataloader()),
        _first_batch_tensor(dm2.test_dataloader()),
    )


def test_set_global_seed_reproducibility():
    set_global_seed(123)

    python_1 = random.random()
    numpy_1 = np.random.rand(5)
    torch_1 = torch.rand(5)

    set_global_seed(123)

    python_2 = random.random()
    numpy_2 = np.random.rand(5)
    torch_2 = torch.rand(5)

    assert python_1 == pytest.approx(python_2)
    np.testing.assert_array_equal(numpy_1, numpy_2)
    assert torch.equal(torch_1, torch_2)

    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_set_global_seed_changes_random_sequence():
    set_global_seed(123)
    python_1 = random.random()
    numpy_1 = np.random.rand()
    torch_1 = torch.rand(1)

    set_global_seed(456)
    python_2 = random.random()
    numpy_2 = np.random.rand()
    torch_2 = torch.rand(1)

    assert python_1 != python_2
    assert numpy_1 != numpy_2
    assert not torch.equal(torch_1, torch_2)


class DummySeedableDataset:
    def __init__(self, seed=123):
        self.random_generator = np.random.default_rng(seed)


def test_seed_worker_sets_worker_rng_and_python_random(monkeypatch):
    dataset = DummySeedableDataset(seed=123)

    worker_info = Mock()
    worker_info.dataset = dataset

    monkeypatch.setattr(
        "torchsig.datasets.datamodules.get_worker_info",
        lambda: worker_info,
    )

    _seed_worker(worker_id=0)

    assert hasattr(dataset, "worker_rng")
    assert isinstance(dataset.worker_rng, np.random.Generator)

    worker_value = dataset.worker_rng.random()

    random_value = random.random()
    random.seed(33158374)  # first integer drawn from default_rng(123), range [0, 2**31)
    expected_random_value = random.random()

    assert random_value == expected_random_value
    assert np.isfinite(worker_value)


def test_seed_worker_unwraps_subset(monkeypatch):
    dataset = DummySeedableDataset(seed=123)
    subset = Subset(dataset, indices=[0])

    worker_info = Mock()
    worker_info.dataset = subset

    monkeypatch.setattr(
        "torchsig.datasets.datamodules.get_worker_info",
        lambda: worker_info,
    )

    _seed_worker(worker_id=0)

    assert hasattr(dataset, "worker_rng")
    assert not hasattr(subset, "worker_rng")


def test_seed_worker_uses_distinct_seed_per_worker(monkeypatch):
    dataset_0 = DummySeedableDataset(seed=123)
    dataset_1 = DummySeedableDataset(seed=123)

    def run_seed_worker(dataset, worker_id):
        worker_info = Mock()
        worker_info.dataset = dataset

        monkeypatch.setattr(
            "torchsig.datasets.datamodules.get_worker_info",
            lambda: worker_info,
        )

        _seed_worker(worker_id=worker_id)
        return dataset.worker_rng.random()

    value_0 = run_seed_worker(dataset_0, worker_id=0)
    value_1 = run_seed_worker(dataset_1, worker_id=1)

    assert value_0 != value_1
