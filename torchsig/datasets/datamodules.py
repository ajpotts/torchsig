"""PyTorch Lightning DataModules
Learn More: https://lightning.ai/docs/pytorch/stable/data/datamodule.html
If dataset does not exist at root, creates new dataset and writes to disk
If dataset does exist, simply loaded it back in
"""
from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Generator
from torch.utils.data import DataLoader, Subset, get_worker_info, random_split
from torch.utils.data._utils.collate import default_collate

from torchsig.datasets.datasets import (
    StaticTorchSigDataset,
    TorchSigDatasetConfig,
    TorchSigIterableDataset,
)
from torchsig.transforms.impairments import Impairments
from torchsig.transforms.metadata_transforms import YOLOLabel
from torchsig.transforms.transforms import ComplexTo2D, Spectrogram
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from torchsig.utils.file_handlers.hdf5 import HDF5Reader, HDF5Writer
from torchsig.utils.writer import DatasetCreator

if TYPE_CHECKING:
    from torchsig.utils.file_handlers import BaseFileHandler

__all__ = ["set_global_seed", "TorchSigDataModule"]

# --------------------------------------------------------------
#  GLOBAL REPRODUCIBILITY HELPERS
# --------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    """Set *all* relevant RNGs to the same seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Force deterministic algorithms (fails loudly if an op is nondet.)
    torch.use_deterministic_algorithms(True)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------
#  DATA MODULE
# --------------------------------------------------------------

def _seed_worker(worker_id: int) -> None:
    """Initialise deterministic NumPy / Python RNGs **inside a DataLoader worker**.

    * If the dataset that the worker sees is a ``torch.utils.data.Subset``,
      we unwrap it to reach the underlying concrete dataset (which inherits
      from ``Seedable`` and therefore owns ``random_generator``).
    * The ``worker_id`` is combined with the master seed to give each
      worker a *different* deterministic seed.
    """
    # -----------------------------------------------------------------
    #    Grab the dataset that lives inside the worker.
    # -----------------------------------------------------------------
    parent = get_worker_info().dataset

    # -----------------------------------------------------------------
    #    ``Subset`` is just a thin wrapper -- fetch the real dataset.
    # -----------------------------------------------------------------
    dataset = parent.dataset if isinstance(parent, Subset) else parent

    # -----------------------------------------------------------------
    #    Pull a deterministic integer from the *parent* generator.
    # -----------------------------------------------------------------
    master_seed = int(dataset.random_generator.integers(0, 2**31))

    # -----------------------------------------------------------------
    #    Derive a unique seed for THIS worker.
    # -----------------------------------------------------------------
    # Multiplying by (worker_id + 1) guarantees distinct seeds across workers.
    worker_seed = master_seed * (worker_id + 1)

    # -----------------------------------------------------------------
    #    Give the worker its own NumPy Generator (so we never touch the
    #    global legacy RNG).
    # -----------------------------------------------------------------
    dataset.worker_rng = np.random.default_rng(worker_seed)

    # -----------------------------------------------------------------
    #    Seed the std-lib ``random`` module (still needed by some TorchSig code).
    # -----------------------------------------------------------------
    random.seed(worker_seed)


class TorchSigDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for creating and loading TorchSig datasets.

    This DataModule handles:
      - Dataset creation or loading from disk via a file handler.
      - Splitting into train/val/test subsets.
      - Batching, collation, and worker seeding for training.

    Attributes:
        root: Directory where datasets are stored or created.
        dataset_size: Total number of samples in the dataset.
        dataset_splits: Fractions or counts for train/val/test splits.
        dataset_metadata: Metadata describing the dataset.
        impairment_level: Optional interference level for synthetic impairments.
        transforms: Transforms applied to the input data.
        target_labels: Names of target metadata fields to include.
        batch_size: Batch size for the training/validation/testing DataLoaders.
        num_workers: Number of worker processes for data loading.
        collate_fn: Custom collate function for batching.
        shuffle: Whether to shuffle the data.
        create_batch_size: Batch size used during on-disk dataset creation.
        create_num_workers: Number of workers used during dataset creation.
        file_writer: FileHandler class for disk I/O.
        file_reader: FileReader class for disk I/O.
        overwrite: If True, existing on-disk data will be overwritten.
        seed: Optional random seed for reproducibility.
        train: Initialized training dataset (set in `setup()`).
        val: Initialized validation dataset (set in `setup()`).
        test: Initialized test dataset (set in `setup()`).
    """
    def __init__(
        self,
        root: str,
        metadata,
        dataset_size: int,
        dataset_splits: list[float] | list[int] = [0.70, 0.20, 0.10],
        # dataloader params
        batch_size: int = 1,
        num_workers: int | None = None,          # ← can be None → default to 0
        collate_fn: Callable | None = None,
        shuffle: bool = True,
        # dataset creator params
        create_batch_size: int = 8,
        create_num_workers: int = 4,
        file_writer: BaseFileHandler = HDF5Writer,
        file_reader: BaseFileHandler = HDF5Reader,
        overwrite: bool = False,
        # transforms
        impairment_level: int = 0,
        transforms: list | None = None,
        target_labels: list[str] | None = None,
        seed: int | None = None,
    ):
        """Initialize the TorchSigDataModule.

        Args:
            root: Path to store or load the dataset.
            metadata: Metadata object, YAML file path, or dict describing classes and settings.
            dataset_size: Total number of samples to generate or load.
            dataset_splits: Fractions or counts for train/val/test splits. Defaults to [0.70, 0.20, 0.10].
            batch_size: Batch size for data loaders. Defaults to 1.
            num_workers: Number of worker processes for data loading. Defaults to 1.
            collate_fn: Custom function to collate batch samples. Defaults to None.
            create_batch_size: Batch size when writing data to disk. Defaults to 8.
            create_num_workers: Workers used when creating the on-disk dataset. Defaults to 4.
            file_writer: FileWriter class for disk I/O.
            file_reader: FileReader class for disk I/O.
            overwrite: If True, existing data at `root` will be overwritten. Defaults to False.
            impairment_level: Level of synthetic impairment to apply. Defaults to 0 (no impairment).
            transforms: List of transforms applied to each sample's input. Defaults to [].
            target_labels: Names of metadata fields to include. Defaults to None.
            seed: Seed for randomness and reproducibility. Defaults to None.

        Raises:
            ValueError: If dataset_splits don't sum to 1.0 (when using fractions).
            FileNotFoundError: If metadata file path is invalid.
        """
        super().__init__()

        # ---- filesystem -------------------------------------------------
        self.root = Path(root)
        self.dataset_size = dataset_size
        self.dataset_splits = dataset_splits

        # ---- meta / transforms -------------------------------------------
        self.metadata = metadata
        self.impairment_level = impairment_level
        impairments = Impairments(level=impairment_level)
        self.burst_impairments = impairments.signal_transforms
        self.whole_signal_impairments = impairments.dataset_transforms
        self.transforms = [self.whole_signal_impairments, *(transforms or [])]

        self.target_labels = target_labels

        # ---- dataloader configuration ------------------------------------
        self.batch_size = batch_size
        self.num_workers = 0 if num_workers is None else num_workers
        self.collate_fn = collate_fn or default_collate
        self.shuffle = shuffle

        # ---- dataset-creation configuration -------------------------------
        self.create_batch_size = create_batch_size
        self.create_num_workers = create_num_workers
        self.file_writer = file_writer
        self.file_reader = file_reader
        self.overwrite = overwrite

        # ---- placeholders ------------------------------------------------
        self.train: StaticTorchSigDataset | None = None
        self.val:   StaticTorchSigDataset | None = None
        self.test:  StaticTorchSigDataset | None = None

        # ---- reproducibility ---------------------------------------------
        self.seed = seed if seed is not None else 42


    @classmethod
    def from_config(
        cls,
        cfg: TorchSigDatasetConfig | str | Path,
        root: str | Path,
        *,
        dataset_size: int | None = None,
        dataset_splits: list[float] | list[int] = [0.70, 0.20, 0.10],
        batch_size: int = 1,
        num_workers: int | None = None,
        create_batch_size: int = 8,
        create_num_workers: int = 4,
        file_writer: type[BaseFileHandler] = HDF5Writer,
        file_reader: type[BaseFileHandler] = HDF5Reader,
        overwrite: bool = False,
        shuffle: bool = True,
        collate_fn: Callable | None = None,
        target_labels: list[str] | None = None,
        **kwargs
    ) -> TorchSigDataModule:
        """Create a TorchSigDataModule from a TorchSigDatasetConfig or YAML path.

        Args:
            cfg: Either a TorchSigDatasetConfig instance or path (str/Path) to a YAML config file
            root: Directory where datasets are stored or created
            dataset_size: Optional override for the dataset size (default: None → uses cfg.dataset_length)
            dataset_splits: Fractions or counts for train/val/test splits (default: [0.70, 0.20, 0.10])
            batch_size: Batch size for data loaders (default: 1)
            num_workers: Number of worker processes for data loading (default: None)
            create_batch_size: Batch size when writing data to disk (default: 8)
            create_num_workers: Workers used when creating dataset (default: 4)
            file_writer: File writer class for disk I/O (default: HDF5Writer)
            file_reader: File reader class for disk I/O (default: HDF5Reader)
            overwrite: Whether to overwrite existing data (default: False)
            shuffle: Whether to shuffle training data (default: True)
            collate_fn: Custom collate function for batching (default: None → identity function)
            target_labels: List of target label names (default: None → auto-select based on output_representation)
            **kwargs: Additional arguments passed to TorchSigDataModule constructor

        Returns:
            Configured TorchSigDataModule instance ready for training

        Raises:
            ValueError: If required parameters for spectrogram output are missing
        """
        from torchsig.utils.defaults import TorchSigDefaults
        from torchsig.utils.yaml import load_config_from_yaml
        from torchsig.utils.defaults import TorchSigDefaults

        # Convert path to config if needed
        if isinstance(cfg, (str, Path)):
            cfg = load_config_from_yaml(Path(cfg))

        # Use provided dataset_size if given, otherwise fall back to config
        final_dataset_size = dataset_size if dataset_size is not None else cfg.dataset_length

        # Merge default metadata with custom metadata from config
        base_metadata = TorchSigDefaults().default_dataset_metadata
        dataset_metadata = {**base_metadata, **cfg.dataset_metadata}

        use_default_target_labels = False

        # Configure output-specific transforms
        additional_transforms: list = []
        if target_labels is None:
            use_default_target_labels = True
            target_labels = []

        if cfg.output_representation == "spectrogram":
            fft_size = cfg.output_spectrogram_fft or dataset_metadata.get("fft_size")
            if fft_size is None:
                raise ValueError(
                    "For spectrogram output, either `output_spectrogram_fft` must be set "
                    "in the config or `fft_size` must be present in dataset_metadata"
                )
            additional_transforms.append(Spectrogram(fft_size=int(fft_size)))
            # Only add YOLOLabel if explicitly requested via target_labels
            if "yolo_label" in target_labels or use_default_target_labels:
                additional_transforms.append(YOLOLabel())
                target_labels = list({*target_labels, "yolo_label"})  # Avoid duplicates
        elif cfg.output_representation == "iq":
            additional_transforms.append(ComplexTo2D())

        return cls(
            root=root,
            metadata=dataset_metadata,
            dataset_size=final_dataset_size,
            dataset_splits=dataset_splits,
            batch_size=batch_size,
            num_workers=num_workers,
            create_batch_size=create_batch_size,
            create_num_workers=create_num_workers,
            file_writer=file_writer,
            file_reader=file_reader,
            overwrite=overwrite,
            shuffle=shuffle,
            collate_fn=collate_fn,
            impairment_level=cfg.impairment_level,
            transforms=additional_transforms,
            target_labels=target_labels or None,
            seed=cfg.seed,
            **kwargs
        )


    def prepare_data(self) -> None:
        """Prepares the dataset by creating new datasets if they do not exist on disk.

        The datasets are created using the `DatasetCreator` class.
        If the dataset already exists on disk, it is loaded back into memory.

        Raises:
            FileNotFoundError: If the root directory cannot be created.
            RuntimeError: If dataset creation fails.
        """
        dataset = TorchSigIterableDataset(
            metadata=self.metadata,
            transforms=self.transforms,
            component_transforms=[self.burst_impairments],
            target_labels=self.target_labels,
            seed=self.seed,
        )
        loader = WorkerSeedingDataLoader(
            dataset=dataset,
            batch_size=self.create_batch_size,
            collate_fn=self.collate_fn,
            seed=self.seed,
        )
        creator = DatasetCreator(
            dataloader=loader,
            dataset_length=self.dataset_size,
            root=self.root,
            overwrite=self.overwrite,
            file_writer=self.file_writer,
        )
        print(f"Full Dataset: Impairment Level {self.impairment_level}, "
              f"{self.dataset_size} samples")
        creator.create()


    def setup(self, stage: str = "fit") -> None:
        """Sets up the train and validation datasets for the given stage.

        Args:
            stage: The stage of the DataModule, typically 'train' or 'test'. Defaults to 'train'.

        Raises:
            FileNotFoundError: If the dataset files are not found at the specified root.
            ValueError: If dataset splits are invalid.
        """
        full_dataset = StaticTorchSigDataset(
            root=self.root,
            file_handler_class=self.file_reader,
            target_labels=self.target_labels,
        )
        self.train, self.val, self.test = random_split(
            full_dataset,
            self.dataset_splits,
            generator=Generator().manual_seed(self.seed),
        )


    #-----------------------------------------------------------------
    # Helper that builds a *deterministic* DataLoader
    #-----------------------------------------------------------------
    def _build_dataloader(self, dataset, shuffle: bool) -> DataLoader:
        gen = torch.Generator()
        gen.manual_seed(self.seed)          # same seed for every epoch

        # ``persistent_workers`` must be a bool; we evaluate it safely.
        persistent = bool(self.num_workers) and (self.num_workers > 0)

        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            generator=gen,
            worker_init_fn=_seed_worker,
            persistent_workers=persistent,
        )


    # -----------------------------------------------------------------
    #  Lightning-specific hooks
    # -----------------------------------------------------------------
    def train_dataloader(self) -> DataLoader:
        """Returns the DataLoader for the training dataset.

        Returns:
            A PyTorch DataLoader for the training dataset.

        Raises:
            RuntimeError: If the training dataset is not initialized.
        """
        return self._build_dataloader(self.train, shuffle=self.shuffle)

    def val_dataloader(self) -> DataLoader:
        """Returns the DataLoader for the validation dataset.

        Returns:
            A PyTorch DataLoader for the validation dataset.

        Raises:
            RuntimeError: If the validation dataset is not initialized.
        """
        return self._build_dataloader(self.val, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Returns the DataLoader for the test dataset.

        Returns:
            A PyTorch DataLoader for the test dataset.

        Raises:
            RuntimeError: If the test dataset is not initialized.
        """
        return self._build_dataloader(self.test, shuffle=False)
