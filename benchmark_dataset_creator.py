#!/usr/bin/env python3
"""Benchmark realistic TorchSig DatasetCreator configurations.

This benchmark uses a TorchSig dataset YAML configuration to construct the same
kind of dataset pipeline used by official dataset-generation scripts. It varies:

- DataLoader worker count
- DatasetCreator multithreading
- repeat count

Each run creates a fresh output directory, generates the requested number of
samples, validates the resulting HDF5 and YAML artifacts, and reports elapsed
time, samples per second, and speedup relative to the single-process baseline.

Example:

    python benchmark_dataset_creator.py \
        --config torchsig/datasets/default_configs/narrowband_impaired_train_all.yaml \
        --samples 512 \
        --batch-size 32 \
        --num-workers 0 2 4 8 \
        --multithreading both \
        --repeats 3 \
        --output-root /tmp/torchsig_dataset_creator_benchmark

For a quick smoke run:

    python benchmark_dataset_creator.py \
        --config path/to/config.yaml \
        --samples 32 \
        --batch-size 8 \
        --num-workers 0 2 \
        --multithreading false \
        --repeats 1
"""

from __future__ import annotations

import argparse
import os
import shutil
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import h5py
import yaml

from torchsig.datasets.datasets import TorchSigIterableDataset
from torchsig.signals.signal_lists import FAMILY_SHARED_LIST
from torchsig.transforms.impairments import Impairments
from torchsig.transforms.metadata_transforms import YOLOLabel
from torchsig.transforms.transforms import ComplexTo2D, Spectrogram
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from torchsig.utils.defaults import TorchSigDefaults
from torchsig.utils.signal_building import lookup_signal_generator_by_string
from torchsig.utils.writer import DatasetCreator, identity_collate_fn
from torchsig.utils.yaml import load_config_from_yaml


@dataclass(frozen=True)
class BenchmarkCase:
    """One DatasetCreator configuration."""

    num_workers: int
    multithreading: bool


@dataclass(frozen=True)
class BenchmarkResult:
    """Median result for one benchmark case."""

    case: BenchmarkCase
    elapsed_seconds: float
    samples_per_second: float
    speedup: float
    output_path: Path
    hdf5_size_bytes: int


def configure_signal_generators(
    dataset: TorchSigIterableDataset,
    mode: Literal["per_signal", "per_family"],
) -> None:
    """Configure per-signal or per-family signal sampling."""
    if mode == "per_signal":
        return

    dataset.signal_generators = []
    dataset.signal_likelihoods = []
    dataset.signal_probabilities = []
    dataset.total_likelihood = 0

    for family in FAMILY_SHARED_LIST:
        family_generator = lookup_signal_generator_by_string(family)
        dataset.add_signal_generator(family_generator, likelihood=1)


def _build_dataset_and_loader(
    *,
    config_path: Path,
    dataset_length: int,
    batch_size: int,
    num_workers: int,
    signal_weighting: Literal["per_signal", "per_family"] | None,
) -> tuple[WorkerSeedingDataLoader, str]:
    """Create the realistic TorchSig dataset and DataLoader."""
    cfg = load_config_from_yaml(config_path)
    mode = signal_weighting or cfg.signal_sampling_mode

    base_metadata = TorchSigDefaults().default_dataset_metadata
    dataset_metadata = dict(base_metadata)
    dataset_metadata.update(cfg.dataset_metadata)

    impairments = Impairments(level=cfg.impairment_level)
    burst_impairments = impairments.signal_transforms
    whole_signal_impairments = impairments.dataset_transforms

    transforms = [whole_signal_impairments]
    target_labels = None

    if cfg.output_representation == "spectrogram":
        transforms.append(
            Spectrogram(fft_size=int(dataset_metadata["fft_size"]))
        )
        transforms.append(YOLOLabel())
        target_labels = ["yolo_label"]
    elif cfg.output_representation == "iq":
        transforms.append(ComplexTo2D())
    else:
        raise ValueError(
            f"Unsupported output_representation: {cfg.output_representation!r}"
        )

    signal_generators = "all" if mode == "per_signal" else []
    dataset = TorchSigIterableDataset(
        signal_generators=signal_generators,
        metadata=dataset_metadata,
        transforms=transforms,
        component_transforms=[burst_impairments],
        target_labels=target_labels,
    )
    configure_signal_generators(dataset, mode)

    dataloader = WorkerSeedingDataLoader(
        dataset,
        seed=cfg.seed,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=identity_collate_fn,
    )

    # DatasetCreator receives the benchmark override, not the YAML's original
    # dataset length, so each benchmark case performs identical work.
    return dataloader, cfg.dataset_id


def _validate_output(root: Path, expected_samples: int) -> int:
    """Validate HDF5 and YAML artifacts and return HDF5 file size."""
    data_path = root / "data.h5"
    dataset_info_path = root / "dataset_info.yaml"
    writer_info_path = root / "writer_info.yaml"

    for path in (data_path, dataset_info_path, writer_info_path):
        if not path.exists():
            raise RuntimeError(f"Missing expected benchmark artifact: {path}")

    with h5py.File(data_path, "r") as hdf5_file:
        indexed_samples = len(hdf5_file["index"])
        if indexed_samples != expected_samples:
            raise RuntimeError(
                f"{data_path} contains {indexed_samples} indexed samples; "
                f"expected {expected_samples}"
            )

    with dataset_info_path.open() as file_obj:
        dataset_info = yaml.safe_load(file_obj) or {}
    if int(dataset_info.get("dataset_length", -1)) != expected_samples:
        raise RuntimeError(
            "dataset_info.yaml reports "
            f"{dataset_info.get('dataset_length')}; expected {expected_samples}"
        )

    with writer_info_path.open() as file_obj:
        writer_info = yaml.safe_load(file_obj) or {}
    if not bool(writer_info.get("complete", False)):
        raise RuntimeError("writer_info.yaml does not mark the dataset complete")
    if int(writer_info.get("items_written", -1)) != expected_samples:
        raise RuntimeError(
            "writer_info.yaml reports "
            f"{writer_info.get('items_written')} items; "
            f"expected {expected_samples}"
        )

    return data_path.stat().st_size


def _run_once(
    *,
    config_path: Path,
    dataset_length: int,
    batch_size: int,
    case: BenchmarkCase,
    signal_weighting: Literal["per_signal", "per_family"] | None,
    output_root: Path,
    repeat: int,
) -> tuple[float, Path, int]:
    """Run one DatasetCreator benchmark case."""
    dataloader, dataset_id = _build_dataset_and_loader(
        config_path=config_path,
        dataset_length=dataset_length,
        batch_size=batch_size,
        num_workers=case.num_workers,
        signal_weighting=signal_weighting,
    )

    case_name = (
        f"workers_{case.num_workers}_"
        f"multithreading_{str(case.multithreading).lower()}"
    )
    root = output_root / dataset_id / case_name / f"repeat_{repeat}"

    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    creator = DatasetCreator(
        dataloader=dataloader,
        dataset_length=dataset_length,
        root=root,
        overwrite=True,
        multithreading=case.multithreading,
    )

    start = time.perf_counter()
    creator.create()
    elapsed = time.perf_counter() - start

    hdf5_size = _validate_output(root, dataset_length)
    return elapsed, root, hdf5_size


def _median_result(
    *,
    case: BenchmarkCase,
    runs: list[tuple[float, Path, int]],
    samples: int,
    baseline_seconds: float,
) -> BenchmarkResult:
    """Create one median benchmark result."""
    median_elapsed = statistics.median(run[0] for run in runs)
    representative = min(
        runs,
        key=lambda run: abs(run[0] - median_elapsed),
    )

    return BenchmarkResult(
        case=case,
        elapsed_seconds=median_elapsed,
        samples_per_second=samples / median_elapsed,
        speedup=baseline_seconds / median_elapsed,
        output_path=representative[1],
        hdf5_size_bytes=representative[2],
    )


def _format_bytes(value: int) -> str:
    size = float(value)
    for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or suffix == "TiB":
            return f"{size:.2f} {suffix}"
        size /= 1024.0
    return f"{size:.2f} TiB"


def _print_results(results: list[BenchmarkResult]) -> None:
    print()
    print("DatasetCreator benchmark results")
    print(
        f"{'Workers':>7}  "
        f"{'Threaded':>8}  "
        f"{'Time (s)':>10}  "
        f"{'Samples/s':>12}  "
        f"{'Speedup':>8}  "
        f"{'HDF5 size':>11}"
    )
    print("-" * 75)

    for result in results:
        print(
            f"{result.case.num_workers:>7d}  "
            f"{str(result.case.multithreading):>8}  "
            f"{result.elapsed_seconds:>10.3f}  "
            f"{result.samples_per_second:>12.2f}  "
            f"{result.speedup:>7.2f}x  "
            f"{_format_bytes(result.hdf5_size_bytes):>11}"
        )

    fastest = min(results, key=lambda result: result.elapsed_seconds)
    print()
    print(
        "Best result: "
        f"num_workers={fastest.case.num_workers}, "
        f"multithreading={fastest.case.multithreading}, "
        f"{fastest.speedup:.2f}x baseline speedup"
    )
    print(
        "Representative dataset output: "
        f"{fastest.output_path.resolve()}"
    )


def _parse_multithreading_modes(value: str) -> list[bool]:
    normalized = value.strip().lower()
    if normalized == "both":
        return [False, True]
    if normalized in {"true", "1", "yes", "on"}:
        return [True]
    if normalized in {"false", "0", "no", "off"}:
        return [False]
    raise ValueError(
        "--multithreading must be one of: false, true, both"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark realistic TorchSig DatasetCreator generation using "
            "a dataset YAML configuration."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="TorchSig dataset YAML configuration.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help=(
            "Benchmark sample count. Defaults to the YAML dataset_length."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--num-workers",
        type=int,
        nargs="+",
        default=[0, 2, 4],
    )
    parser.add_argument(
        "--multithreading",
        default="both",
        choices=["false", "true", "both"],
    )
    parser.add_argument(
        "--signal-weighting",
        choices=["per_signal", "per_family"],
        default=None,
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help=(
            "Retained for symmetry with other benchmarks. Explicit "
            "--output-root paths are always preserved."
        ),
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not args.config.exists():
        raise FileNotFoundError(args.config)
    if args.samples is not None and args.samples < 1:
        raise ValueError("--samples must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if any(worker_count < 0 for worker_count in args.num_workers):
        raise ValueError("--num-workers values cannot be negative")
    if 0 not in args.num_workers:
        raise ValueError(
            "--num-workers must include 0 to establish the baseline"
        )


def main() -> None:
    args = _parse_args()
    _validate_args(args)

    cfg = load_config_from_yaml(args.config)
    dataset_length = (
        cfg.dataset_length if args.samples is None else args.samples
    )

    temporary_root: Path | None = None
    if args.output_root is None:
        temporary_root = Path(
            os.path.abspath(
                os.path.join(
                    "/tmp",
                    f"torchsig_dataset_creator_benchmark_{os.getpid()}",
                )
            )
        )
        temporary_root.mkdir(parents=True, exist_ok=True)
        output_root = temporary_root
    else:
        output_root = args.output_root.resolve()
        output_root.mkdir(parents=True, exist_ok=True)

    multithreading_modes = _parse_multithreading_modes(
        args.multithreading
    )
    cases = [
        BenchmarkCase(
            num_workers=worker_count,
            multithreading=multithreading,
        )
        for worker_count in dict.fromkeys(args.num_workers)
        for multithreading in multithreading_modes
    ]

    baseline_case = BenchmarkCase(
        num_workers=0,
        multithreading=False,
    )
    if baseline_case not in cases:
        cases.insert(0, baseline_case)

    print("TorchSig DatasetCreator benchmark")
    print(f"Config: {args.config.resolve()}")
    print(f"Output root: {output_root.resolve()}")
    print(f"Samples per run: {dataset_length}")
    print(f"Batch size: {args.batch_size}")
    print(f"Repeats: {args.repeats}")
    print(
        "Signal weighting: "
        f"{args.signal_weighting or cfg.signal_sampling_mode}"
    )

    all_runs: dict[BenchmarkCase, list[tuple[float, Path, int]]] = {}

    for case in cases:
        print()
        print(
            "Running "
            f"num_workers={case.num_workers}, "
            f"multithreading={case.multithreading}"
        )
        runs = [
            _run_once(
                config_path=args.config,
                dataset_length=dataset_length,
                batch_size=args.batch_size,
                case=case,
                signal_weighting=args.signal_weighting,
                output_root=output_root,
                repeat=repeat,
            )
            for repeat in range(args.repeats)
        ]
        all_runs[case] = runs

    baseline_seconds = statistics.median(
        run[0] for run in all_runs[baseline_case]
    )

    results = [
        _median_result(
            case=case,
            runs=all_runs[case],
            samples=dataset_length,
            baseline_seconds=baseline_seconds,
        )
        for case in cases
    ]

    _print_results(results)

    if temporary_root is not None:
        print()
        print("Benchmark artifacts were written to:")
        print(temporary_root.resolve())


if __name__ == "__main__":
    main()


