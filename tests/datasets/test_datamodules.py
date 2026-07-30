"""Unit Tests for datamodules"""
import random
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

import numpy as np
import pytest
import torch
import h5py

from torchsig.datasets.datamodules import (
    SplitTorchSigDataModule,
    TorchSigDataModule,
    _seed_worker,
    set_global_seed,
)
from torchsig.datasets.datasets import TorchSigDatasetConfig
from torch.utils.data import Subset
from torchsig.utils.defaults import TorchSigDefaults
from torchsig.transforms.transforms import Spectrogram
from torchsig.utils.file_handlers import (
    PackedHDF5Reader,
    PackedHDF5Writer,
    HDF5Reader,
    HDF5Writer,
    HomogeneousHDF5Reader,
    HomogeneousHDF5Writer,
)
from torchsig.utils.writer import identity_collate_fn


def _signal_summary_collate(batch):
    return [
        (
            signal["duration_in_samples"],
            signal.data.shape,
            signal.data.dtype.str,
        )
        for signal in batch
    ]


@pytest.fixture
def split_configs():
    """Return distinct train, validation, and test dataset configs."""
    metadata = TorchSigDefaults().default_dataset_metadata.copy()
    metadata.update(
        {
            "num_iq_samples_dataset": 4096,
            "fft_size": 64,
            "fft_stride": 64,
            "num_signals_min": 1,
            "num_signals_max": 1,
            "signal_duration_in_samples_min": 3276,
            "signal_duration_in_samples_max": 4096,
        }
    )

    common = {
        "dataset_id": "test_dataset",
        "dataset_metadata": metadata,
        "output_representation": "iq",
        "output_spectrogram_fft": None,
        "signal_sampling_mode": "random",
        "impairment_level": 0,
    }

    return (
        TorchSigDatasetConfig(
            **common,
            dataset_length=12,
            seed=11,
        ),
        TorchSigDatasetConfig(
            **common,
            dataset_length=6,
            seed=22,
        ),
        TorchSigDatasetConfig(
            **common,
            dataset_length=4,
            seed=33,
        ),
    )

def _mock_config(
    *,
    dataset_id: str = "test_dataset",
    dataset_length: int,
    seed: int,
    output_representation: str = "iq",
) -> MagicMock:
    cfg = MagicMock(spec=TorchSigDatasetConfig)
    cfg.dataset_id = dataset_id
    cfg.dataset_length = dataset_length
    cfg.seed = seed
    cfg.output_representation = output_representation
    cfg.dataset_metadata = {}
    return cfg


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


def test_split_datamodule_initializes_from_three_configs(tmp_path):
    train_cfg = _mock_config(dataset_length=12, seed=11)
    val_cfg = _mock_config(dataset_length=6, seed=22)
    test_cfg = _mock_config(dataset_length=4, seed=33)

    dm = SplitTorchSigDataModule(
        train_cfg=train_cfg,
        val_cfg=val_cfg,
        test_cfg=test_cfg,
        root=tmp_path,
        batch_size=8,
        num_workers=None,
        create_batch_size=4,
        create_num_workers=2,
        signal_generators=["bpsk"],
    )

    assert dm.train_cfg is train_cfg
    assert dm.val_cfg is val_cfg
    assert dm.test_cfg is test_cfg

    assert dm.root == tmp_path / "test_dataset"
    assert dm.batch_size == 8
    assert dm.num_workers == 0
    assert dm.create_batch_size == 4
    assert dm.create_num_workers == 2
    assert dm.signal_generators == ["bpsk"]

    assert dm.train is None
    assert dm.val is None
    assert dm.test is None

def test_split_datamodule_rejects_mismatched_output_representations(tmp_path):
    train_cfg = _mock_config(
        dataset_length=12,
        seed=11,
        output_representation="iq",
    )
    val_cfg = _mock_config(
        dataset_length=6,
        seed=22,
        output_representation="spectrogram",
    )
    test_cfg = _mock_config(
        dataset_length=4,
        seed=33,
        output_representation="iq",
    )

    with pytest.raises(
        ValueError,
        match="same output representation",
    ):
        SplitTorchSigDataModule(
            train_cfg=train_cfg,
            val_cfg=val_cfg,
            test_cfg=test_cfg,
            root=tmp_path,
        )

@patch("torchsig.datasets.datamodules.DatasetCreator")
@patch("torchsig.datasets.datamodules.WorkerSeedingDataLoader")
@patch("torchsig.datasets.datamodules.TorchSigIterableDataset")
def test_split_datamodule_prepare_data_creates_all_splits(
    iterable_dataset_cls,
    dataloader_cls,
    dataset_creator_cls,
    tmp_path,
):
    train_cfg = _mock_config(dataset_length=12, seed=11)
    val_cfg = _mock_config(dataset_length=6, seed=22)
    test_cfg = _mock_config(dataset_length=4, seed=33)

    datasets = [MagicMock(), MagicMock(), MagicMock()]
    iterable_dataset_cls.side_effect = datasets

    loaders = [MagicMock(), MagicMock(), MagicMock()]
    dataloader_cls.side_effect = loaders

    creators = [MagicMock(), MagicMock(), MagicMock()]
    dataset_creator_cls.side_effect = creators

    dm = SplitTorchSigDataModule(
        train_cfg=train_cfg,
        val_cfg=val_cfg,
        test_cfg=test_cfg,
        root=tmp_path,
        create_batch_size=4,
        create_num_workers=2,
        overwrite=True,
        signal_generators=["bpsk", "qpsk"],
    )

    dm.prepare_data()

    assert iterable_dataset_cls.call_count == 3
    assert dataloader_cls.call_count == 3
    assert dataset_creator_cls.call_count == 3

    assert dataset_creator_cls.call_args_list[0].kwargs[
        "dataset_length"
    ] == 12
    assert dataset_creator_cls.call_args_list[1].kwargs[
        "dataset_length"
    ] == 6
    assert dataset_creator_cls.call_args_list[2].kwargs[
        "dataset_length"
    ] == 4

    assert Path(dataset_creator_cls.call_args_list[0].kwargs["root"]) == (
        tmp_path / "test_dataset" / "train"
    )
    assert Path(dataset_creator_cls.call_args_list[1].kwargs["root"]) == (
        tmp_path / "test_dataset" / "val"
    )
    assert Path(dataset_creator_cls.call_args_list[2].kwargs["root"]) == (
        tmp_path / "test_dataset" / "test"
    )

    for creator in creators:
        creator.create.assert_called_once_with()

@patch("torchsig.datasets.datamodules.DatasetCreator")
@patch("torchsig.datasets.datamodules.WorkerSeedingDataLoader")
@patch("torchsig.datasets.datamodules.TorchSigIterableDataset")
def test_split_datamodule_uses_split_specific_seeds(
    iterable_dataset_cls,
    dataloader_cls,
    dataset_creator_cls,
    tmp_path,
):
    train_cfg = _mock_config(dataset_length=12, seed=11)
    val_cfg = _mock_config(dataset_length=6, seed=22)
    test_cfg = _mock_config(dataset_length=4, seed=33)

    dm = SplitTorchSigDataModule(
        train_cfg=train_cfg,
        val_cfg=val_cfg,
        test_cfg=test_cfg,
        root=tmp_path,
    )

    dm.prepare_data()

    dataset_seeds = [
        call_args.kwargs["seed"]
        for call_args in iterable_dataset_cls.call_args_list
    ]
    loader_seeds = [
        call_args.kwargs["seed"]
        for call_args in dataloader_cls.call_args_list
    ]

    assert dataset_seeds == [11, 22, 33]
    assert loader_seeds == [11, 22, 33]


@patch("torchsig.datasets.datamodules.StaticTorchSigDataset")
def test_split_datamodule_setup_fit_loads_train_and_val_only(
    static_dataset_cls,
    tmp_path,
):
    train_cfg = _mock_config(dataset_length=12, seed=11)
    val_cfg = _mock_config(dataset_length=6, seed=22)
    test_cfg = _mock_config(dataset_length=4, seed=33)

    train_dataset = MagicMock()
    val_dataset = MagicMock()
    static_dataset_cls.side_effect = [train_dataset, val_dataset]

    dm = SplitTorchSigDataModule(
        train_cfg=train_cfg,
        val_cfg=val_cfg,
        test_cfg=test_cfg,
        root=tmp_path,
    )

    dm.setup("fit")

    assert dm.train is train_dataset
    assert dm.val is val_dataset
    assert dm.test is None

    assert static_dataset_cls.call_count == 2

    assert Path(static_dataset_cls.call_args_list[0].kwargs["root"]) == (
        tmp_path / "test_dataset" / "train"
    )
    assert Path(static_dataset_cls.call_args_list[1].kwargs["root"]) == (
        tmp_path / "test_dataset" / "val"
    )


@patch("torchsig.datasets.datamodules.StaticTorchSigDataset")
def test_split_datamodule_setup_test_loads_test_only(
    static_dataset_cls,
    tmp_path,
):
    train_cfg = _mock_config(dataset_length=12, seed=11)
    val_cfg = _mock_config(dataset_length=6, seed=22)
    test_cfg = _mock_config(dataset_length=4, seed=33)

    test_dataset = MagicMock()
    static_dataset_cls.return_value = test_dataset

    dm = SplitTorchSigDataModule(
        train_cfg=train_cfg,
        val_cfg=val_cfg,
        test_cfg=test_cfg,
        root=tmp_path,
    )

    dm.setup("test")

    assert dm.train is None
    assert dm.val is None
    assert dm.test is test_dataset

    static_dataset_cls.assert_called_once()

    assert Path(static_dataset_cls.call_args.kwargs["root"]) == (
        tmp_path / "test_dataset" / "test"
    )

@patch("torchsig.datasets.datamodules.StaticTorchSigDataset")
def test_split_datamodule_setup_none_loads_all_splits(
    static_dataset_cls,
    tmp_path,
):
    train_cfg = _mock_config(dataset_length=12, seed=11)
    val_cfg = _mock_config(dataset_length=6, seed=22)
    test_cfg = _mock_config(dataset_length=4, seed=33)

    datasets = [MagicMock(), MagicMock(), MagicMock()]
    static_dataset_cls.side_effect = datasets

    dm = SplitTorchSigDataModule(
        train_cfg=train_cfg,
        val_cfg=val_cfg,
        test_cfg=test_cfg,
        root=tmp_path,
    )

    dm.setup(None)

    assert dm.train is datasets[0]
    assert dm.val is datasets[1]
    assert dm.test is datasets[2]


@patch("torchsig.datasets.datamodules.random_split")
@patch("torchsig.datasets.datamodules.StaticTorchSigDataset")
def test_split_datamodule_does_not_use_random_split(
    static_dataset_cls,
    random_split_mock,
    tmp_path,
):
    train_cfg = _mock_config(dataset_length=12, seed=11)
    val_cfg = _mock_config(dataset_length=6, seed=22)
    test_cfg = _mock_config(dataset_length=4, seed=33)

    static_dataset_cls.side_effect = [
        MagicMock(),
        MagicMock(),
        MagicMock(),
    ]

    dm = SplitTorchSigDataModule(
        train_cfg=train_cfg,
        val_cfg=val_cfg,
        test_cfg=test_cfg,
        root=tmp_path,
    )

    dm.setup(None)

    random_split_mock.assert_not_called()


@patch("torchsig.datasets.datamodules.DataLoader")
def test_split_datamodule_dataloader_shuffle_behavior(
    dataloader_cls,
    tmp_path,
):
    train_cfg = _mock_config(dataset_length=12, seed=11)
    val_cfg = _mock_config(dataset_length=6, seed=22)
    test_cfg = _mock_config(dataset_length=4, seed=33)

    dataloader_cls.side_effect = [
        MagicMock(),
        MagicMock(),
        MagicMock(),
    ]

    dm = SplitTorchSigDataModule(
        train_cfg=train_cfg,
        val_cfg=val_cfg,
        test_cfg=test_cfg,
        root=tmp_path,
        shuffle=True,
    )
    dm.train = MagicMock()
    dm.val = MagicMock()
    dm.test = MagicMock()

    dm.train_dataloader()
    dm.val_dataloader()
    dm.test_dataloader()

    assert dataloader_cls.call_args_list[0].kwargs["shuffle"] is True
    assert dataloader_cls.call_args_list[1].kwargs["shuffle"] is False
    assert dataloader_cls.call_args_list[2].kwargs["shuffle"] is False


@pytest.mark.parametrize(
    "method_name",
    [
        "train_dataloader",
        "val_dataloader",
        "test_dataloader",
    ],
)
def test_split_datamodule_dataloader_requires_setup(
    method_name,
    tmp_path,
):
    train_cfg = _mock_config(dataset_length=12, seed=11)
    val_cfg = _mock_config(dataset_length=6, seed=22)
    test_cfg = _mock_config(dataset_length=4, seed=33)

    dm = SplitTorchSigDataModule(
        train_cfg=train_cfg,
        val_cfg=val_cfg,
        test_cfg=test_cfg,
        root=tmp_path,
    )

    with pytest.raises(RuntimeError, match="setup"):
        getattr(dm, method_name)()

    
@pytest.mark.slow_no_gpu
def test_split_datamodule_smoke(tmp_path, split_configs):
    train_cfg, val_cfg, test_cfg = split_configs

    dm = SplitTorchSigDataModule(
        train_cfg=train_cfg,
        val_cfg=val_cfg,
        test_cfg=test_cfg,
        root=tmp_path,
        batch_size=2,
        num_workers=0,
        create_batch_size=2,
        create_num_workers=0,
        overwrite=True,
        collate_fn=identity_collate_fn,
    )

    dm.prepare_data()
    dm.setup(None)

    assert len(dm.train) == train_cfg.dataset_length
    assert len(dm.val) == val_cfg.dataset_length
    assert len(dm.test) == test_cfg.dataset_length

    assert next(iter(dm.train_dataloader()))
    assert next(iter(dm.val_dataloader()))
    assert next(iter(dm.test_dataloader()))


@pytest.mark.parametrize(
    ("transforms", "expected_ndim"),
    [([], 1), ([Spectrogram(fft_size=64)], 2)],
    ids=["iq", "spectrogram"],
)
def test_torchsig_datamodule_infers_packed_reader_end_to_end(
    tmp_path, transforms, expected_ndim
):
    metadata = TorchSigDefaults().default_dataset_metadata.copy()
    metadata.update(
        {
            "num_iq_samples_dataset": 4_096,
            "fft_size": 64,
            "fft_stride": 64,
            "num_signals_min": 1,
            "num_signals_max": 1,
            "signal_duration_in_samples_min": 3_276,
            "signal_duration_in_samples_max": 4_096,
        }
    )
    writer_options = {
        "compression": None,
        "shuffle": False,
        "fletcher32": False,
        "max_batches_in_memory": 1,
    }
    dm = TorchSigDataModule(
        root=tmp_path,
        metadata=metadata,
        dataset_size=6,
        dataset_splits=[4, 1, 1],
        create_batch_size=2,
        file_writer=PackedHDF5Writer,
        file_writer_kwargs=writer_options,
        overwrite=True,
        impairment_level=0,
        transforms=transforms,
        collate_fn=identity_collate_fn,
        num_workers=0,
    )
    writer_options["compression"] = "lzf"

    assert dm.file_reader is PackedHDF5Reader
    assert dm.file_writer_kwargs["compression"] is None
    dm.prepare_data()
    dm.setup()

    full_dataset = dm.train.dataset
    assert isinstance(full_dataset.reader, PackedHDF5Reader)
    assert full_dataset[0].data.ndim == expected_ndim
    full_dataset.reader.teardown()
    with h5py.File(tmp_path / "data.h5", "r") as handle:
        assert handle.attrs["compression"] == "none"
        assert handle["data/0"].compression is None
        assert not handle["data/0"].shuffle
        assert not handle["data/0"].fletcher32
    writer_info = (tmp_path / "writer_info.yaml").read_text()
    assert (
        "torchsig.utils.file_handlers.packed_hdf5.PackedHDF5Writer"
        in writer_info
    )
    assert (
        "torchsig.utils.file_handlers.packed_hdf5.PackedHDF5Reader"
        in writer_info
    )


@pytest.mark.parametrize("num_workers", [0, 2])
@pytest.mark.parametrize(
    ("transforms", "expected_ndim"),
    [([], 1), ([Spectrogram(fft_size=64)], 2)],
    ids=["iq", "spectrogram"],
)
def test_torchsig_datamodule_infers_homogeneous_reader_end_to_end(
    tmp_path,
    transforms,
    expected_ndim,
    num_workers,
):
    metadata = TorchSigDefaults().default_dataset_metadata.copy()
    metadata.update(
        {
            "num_iq_samples_dataset": 4_096,
            "fft_size": 64,
            "fft_stride": 64,
            "num_signals_min": 1,
            "num_signals_max": 1,
            "signal_duration_in_samples_min": 3_276,
            "signal_duration_in_samples_max": 4_096,
        }
    )
    dm = TorchSigDataModule(
        root=tmp_path,
        metadata=metadata,
        dataset_size=6,
        dataset_splits=[4, 1, 1],
        create_batch_size=2,
        create_num_workers=0,
        file_writer=HomogeneousHDF5Writer,
        file_writer_kwargs={
            "compression": None,
            "shuffle": False,
            "fletcher32": False,
            "chunk_samples": 2,
        },
        overwrite=True,
        impairment_level=0,
        transforms=transforms,
        collate_fn=identity_collate_fn,
        num_workers=num_workers,
    )

    assert dm.file_reader is HomogeneousHDF5Reader
    dm.prepare_data()
    dm.setup()
    full_dataset = dm.train.dataset
    assert isinstance(full_dataset.reader, HomogeneousHDF5Reader)
    assert full_dataset[0].data.ndim == expected_ndim
    dm.collate_fn = _signal_summary_collate
    batch = next(iter(dm.train_dataloader()))
    assert batch
    assert all(len(item[1]) == expected_ndim for item in batch)
    full_dataset.reader.teardown()
    with h5py.File(tmp_path / "data.h5", "r") as handle:
        assert handle.attrs["compression"] == "none"
        assert handle["data"].compression is None
        assert handle["data"].chunks[0] == 2


@pytest.mark.parametrize(
    ("file_writer", "file_reader"),
    [
        (PackedHDF5Writer, HDF5Reader),
        (HDF5Writer, PackedHDF5Reader),
        (HomogeneousHDF5Writer, PackedHDF5Reader),
        (PackedHDF5Writer, HomogeneousHDF5Reader),
    ],
)
def test_torchsig_datamodule_rejects_incompatible_handler_pair(
    tmp_path, file_writer, file_reader
):
    with pytest.raises(ValueError, match="Incompatible file handler pair"):
        TorchSigDataModule(
            root=tmp_path,
            metadata=TorchSigDefaults().default_dataset_metadata,
            dataset_size=1,
            file_writer=file_writer,
            file_reader=file_reader,
        )


@pytest.mark.parametrize(
    "file_writer_kwargs",
    [{"unknown_option": True}, ["not", "a", "dictionary"]],
)
def test_torchsig_datamodule_rejects_invalid_writer_options(
    tmp_path, file_writer_kwargs
):
    with pytest.raises(TypeError, match="file_writer_kwargs|Invalid options"):
        TorchSigDataModule(
            root=tmp_path,
            metadata=TorchSigDefaults().default_dataset_metadata,
            dataset_size=1,
            file_writer=PackedHDF5Writer,
            file_writer_kwargs=file_writer_kwargs,
        )


def test_split_datamodule_infers_packed_reader_end_to_end(
    tmp_path, split_configs
):
    train_cfg, val_cfg, test_cfg = split_configs
    dm = SplitTorchSigDataModule(
        train_cfg=train_cfg,
        val_cfg=val_cfg,
        test_cfg=test_cfg,
        root=tmp_path,
        create_batch_size=2,
        create_num_workers=0,
        file_writer=PackedHDF5Writer,
        file_writer_kwargs={
            "compression": None,
            "shuffle": False,
            "fletcher32": False,
            "max_batches_in_memory": 1,
        },
        overwrite=True,
        collate_fn=identity_collate_fn,
    )

    assert dm.file_reader is PackedHDF5Reader
    dm.prepare_data()
    dm.setup(None)

    assert isinstance(dm.train.reader, PackedHDF5Reader)
    assert isinstance(dm.val.reader, PackedHDF5Reader)
    assert isinstance(dm.test.reader, PackedHDF5Reader)
    dm.train.reader.teardown()
    dm.val.reader.teardown()
    dm.test.reader.teardown()
    with h5py.File(dm.root / "train" / "data.h5", "r") as handle:
        assert handle.attrs["compression"] == "none"
        assert handle["data/0"].compression is None
