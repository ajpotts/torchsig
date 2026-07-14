#!/usr/bin/env python3
"""Benchmark DatasetCreator with the experimental sharded HDF5 backend.

This benchmark builds a realistic TorchSig dataset from a YAML configuration
and varies:

- DataLoader worker count
- HDF5 shard count
- DatasetCreator multithreading
- repeat count

The sharded backend must be available as either:

    torchsig.utils.file_handlers.sharded_hdf5

or as a local module:

    hdf5_sharded.py

Each run creates a fresh output directory, generates the requested number of
samples, validates the manifest, all shard files, and DatasetCreator YAML
artifacts, then reports median elapsed time, throughput, and speedup relative
to the baseline:

    num_workers=0, num_shards=1, multithreading=False

Example:

    python benchmark_dataset_creator_sharded.py \
        --config torchsig/datasets/default_configs/narrowband_impaired_train_all.yaml \
        --samples 512 \
        --batch-size 32 \
        --num-workers 0 2 4 8 \
        --num-shards 1 2 4 8 \
        --multithreading false \
        --repeats 3 \
        --output-root /tmp/torchsig_sharded_benchmark
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

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

try:
    from torchsig.utils.file_handlers import HDF5Writer
except ImportError:
    try:
        from hdf5_sharded import HDF5Writer
    except ImportError as exc:
        raise ImportError(
            "Could not import the sharded HDF5Writer. Install it as "
            "'torchsig.utils.file_handlers.sharded_hdf5' or place "
            "'hdf5_sharded.py' beside this benchmark."
        ) from exc


@dataclass(frozen=True)
class BenchmarkCase:
    """One DatasetCreator and sharded-writer configuration."""

    num_workers: int
    num_shards: int
    multithreading: bool


@dataclass(frozen=True)
class BenchmarkRun:
    """One completed benchmark run."""

    elapsed_seconds: float
    output_path: Path
    total_hdf5_bytes: int
    shard_sizes: tuple[int, ...]


@dataclass(frozen=True)
class BenchmarkResult:
    """Median result for one benchmark case."""

    case: BenchmarkCase
    elapsed_seconds: float
    samples_per_second: float
    speedup: float
    output_path: Path
    total_hdf5_bytes: int
    shard_sizes: tuple[int, ...]


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

    transforms: list[Any] = [whole_signal_impairments]
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
    return dataloader, cfg.dataset_id


def _validate_manifest(
    root: Path,
    expected_samples: int,
    expected_shards: int,
) -> tuple[int, tuple[int, ...]]:
    """Validate the sharded HDF5 manifest and all shard files."""
    manifest_path = root / HDF5Writer.manifest_filename
    if not manifest_path.exists():
        raise RuntimeError(f"Missing HDF5 manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())

    total_samples = int(manifest.get("total_samples", -1))
    if total_samples != expected_samples:
        raise RuntimeError(
            f"Manifest reports {total_samples} samples; "
            f"expected {expected_samples}"
        )

    shard_entries = manifest.get("shards", [])
    if len(shard_entries) != expected_shards:
        raise RuntimeError(
            f"Manifest reports {len(shard_entries)} shards; "
            f"expected {expected_shards}"
        )

    batches = sorted(
        manifest.get("batches", []),
        key=lambda item: int(item["batch_idx"]),
    )
    actual_batch_indices = [int(batch["batch_idx"]) for batch in batches]
    if actual_batch_indices != list(range(len(actual_batch_indices))):
        raise RuntimeError(
            "Manifest batch indices are not contiguous: "
            f"{actual_batch_indices}"
        )

    manifest_sample_count = sum(int(batch["length"]) for batch in batches)
    if manifest_sample_count != expected_samples:
        raise RuntimeError(
            f"Manifest batch lengths sum to {manifest_sample_count}; "
            f"expected {expected_samples}"
        )

    shard_sizes: list[int] = []
    indexed_across_shards = 0

    for shard_entry in shard_entries:
        shard_id = int(shard_entry["shard_id"])
        shard_path = root / shard_entry["filename"]

        if not shard_path.exists():
            raise RuntimeError(f"Missing shard file: {shard_path}")

        with h5py.File(shard_path, "r") as hdf5_file:
            if int(hdf5_file.attrs["shard_id"]) != shard_id:
                raise RuntimeError(
                    f"{shard_path} has unexpected shard_id attribute"
                )
            indexed_across_shards += len(hdf5_file["index"])

        shard_sizes.append(shard_path.stat().st_size)

    if indexed_across_shards != expected_samples:
        raise RuntimeError(
            f"Shard indices total {indexed_across_shards}; "
            f"expected {expected_samples}"
        )

    return sum(shard_sizes), tuple(shard_sizes)


def _validate_dataset_creator_yamls(
    root: Path,
    expected_samples: int,
) -> None:
    """Validate DatasetCreator completion metadata."""
    dataset_info_path = root / "dataset_info.yaml"
    writer_info_path = root / "writer_info.yaml"

    for path in (dataset_info_path, writer_info_path):
        if not path.exists():
            raise RuntimeError(f"Missing DatasetCreator artifact: {path}")

    with dataset_info_path.open() as file_obj:
        dataset_info = yaml.safe_load(file_obj) or {}
    if int(dataset_info.get("dataset_length", -1)) != expected_samples:
        raise RuntimeError(
            "dataset_info.yaml reports "
            f"{dataset_info.get('dataset_length')} samples; "
            f"expected {expected_samples}"
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


def _run_once(
    *,
    config_path: Path,
    dataset_length: int,
    batch_size: int,
    case: BenchmarkCase,
    signal_weighting: Literal["per_signal", "per_family"] | None,
    output_root: Path,
    repeat: int,
    max_batches_in_memory: int,
    compression: str | None,
) -> BenchmarkRun:
    """Run one DatasetCreator benchmark case."""
    dataloader, dataset_id = _build_dataset_and_loader(
        config_path=config_path,
        batch_size=batch_size,
        num_workers=case.num_workers,
        signal_weighting=signal_weighting,
    )

    case_name = (
        f"workers_{case.num_workers}_"
        f"shards_{case.num_shards}_"
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
        file_handler=HDF5Writer,
        num_shards=case.num_shards,
        max_batches_in_memory=max_batches_in_memory,
        compression=compression,
        shuffle=compression is not None,
        fletcher32=False,
        multiprocessing_context="spawn",
    )

    start = time.perf_counter()
    creator.create()
    elapsed = time.perf_counter() - start

    _validate_dataset_creator_yamls(root, dataset_length)
    total_hdf5_bytes, shard_sizes = _validate_manifest(
        root,
        expected_samples=dataset_length,
        expected_shards=case.num_shards,
    )

    return BenchmarkRun(
        elapsed_seconds=elapsed,
        output_path=root,
        total_hdf5_bytes=total_hdf5_bytes,
        shard_sizes=shard_sizes,
    )


def _median_result(
    *,
    case: BenchmarkCase,
    runs: list[BenchmarkRun],
    samples: int,
    baseline_seconds: float,
) -> BenchmarkResult:
    """Create one median result from repeated runs."""
    median_elapsed = statistics.median(
        run.elapsed_seconds for run in runs
    )
    representative = min(
        runs,
        key=lambda run: abs(run.elapsed_seconds - median_elapsed),
    )

    return BenchmarkResult(
        case=case,
        elapsed_seconds=median_elapsed,
        samples_per_second=samples / median_elapsed,
        speedup=baseline_seconds / median_elapsed,
        output_path=representative.output_path,
        total_hdf5_bytes=representative.total_hdf5_bytes,
        shard_sizes=representative.shard_sizes,
    )


def _normalize_compression(value: str) -> str | None:
    """Convert CLI compression text to an HDF5 setting."""
    normalized = value.strip().lower()
    if normalized in {"none", "off", "false", "0"}:
        return None
    return normalized


def _format_bytes(value: int) -> str:
    """Format a byte count for display."""
    size = float(value)
    for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or suffix == "TiB":
            return f"{size:.2f} {suffix}"
        size /= 1024.0
    return f"{size:.2f} TiB"


def _print_results(results: list[BenchmarkResult]) -> None:
    """Print the benchmark result table."""
    print()
    print("Sharded DatasetCreator benchmark results")
    print(
        f"{'Workers':>7}  "
        f"{'Shards':>6}  "
        f"{'Threaded':>8}  "
        f"{'Time (s)':>10}  "
        f"{'Samples/s':>12}  "
        f"{'Speedup':>8}  "
        f"{'HDF5 total':>11}"
    )
    print("-" * 88)

    for result in results:
        print(
            f"{result.case.num_workers:>7d}  "
            f"{result.case.num_shards:>6d}  "
            f"{str(result.case.multithreading):>8}  "
            f"{result.elapsed_seconds:>10.3f}  "
            f"{result.samples_per_second:>12.2f}  "
            f"{result.speedup:>7.2f}x  "
            f"{_format_bytes(result.total_hdf5_bytes):>11}"
        )

    fastest = min(results, key=lambda result: result.elapsed_seconds)
    print()
    print(
        "Best result: "
        f"num_workers={fastest.case.num_workers}, "
        f"num_shards={fastest.case.num_shards}, "
        f"multithreading={fastest.case.multithreading}, "
        f"{fastest.speedup:.2f}x baseline speedup"
    )
    print(
        "Representative dataset output: "
        f"{fastest.output_path.resolve()}"
    )
    print(
        "Representative shard sizes: "
        + ", ".join(_format_bytes(size) for size in fastest.shard_sizes)
    )


def _parse_multithreading_modes(value: str) -> list[bool]:
    """Parse one or both DatasetCreator threading modes."""
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
            "the experimental sharded HDF5 backend."
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
        help="Sample count; defaults to YAML dataset_length.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--num-workers",
        type=int,
        nargs="+",
        default=[0, 2, 4, 8],
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        nargs="+",
        default=[1, 2, 4],
    )
    parser.add_argument(
        "--multithreading",
        default="false",
        choices=["false", "true", "both"],
    )
    parser.add_argument(
        "--signal-weighting",
        choices=["per_signal", "per_family"],
        default=None,
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--max-batches-in-memory",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--compression",
        default="none",
        help="HDF5 compression filter: none, lzf, gzip, etc.",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    """Validate benchmark CLI arguments."""
    if not args.config.exists():
        raise FileNotFoundError(args.config)
    if args.samples is not None and args.samples < 1:
        raise ValueError("--samples must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if args.max_batches_in_memory < 1:
        raise ValueError("--max-batches-in-memory must be at least 1")
    if any(worker_count < 0 for worker_count in args.num_workers):
        raise ValueError("--num-workers values cannot be negative")
    if any(shard_count < 1 for shard_count in args.num_shards):
        raise ValueError("--num-shards values must be at least 1")
    if 0 not in args.num_workers:
        raise ValueError(
            "--num-workers must include 0 for the baseline"
        )
    if 1 not in args.num_shards:
        raise ValueError(
            "--num-shards must include 1 for the baseline"
        )


def main() -> None:
    """Run the sharded DatasetCreator benchmark suite."""
    args = _parse_args()
    _validate_args(args)

    cfg = load_config_from_yaml(args.config)
    dataset_length = (
        cfg.dataset_length if args.samples is None else args.samples
    )

    if args.output_root is None:
        output_root = Path(
            os.path.abspath(
                os.path.join(
                    "/tmp",
                    f"torchsig_sharded_benchmark_{os.getpid()}",
                )
            )
        )
    else:
        output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    multithreading_modes = _parse_multithreading_modes(
        args.multithreading
    )
    cases = [
        BenchmarkCase(
            num_workers=worker_count,
            num_shards=shard_count,
            multithreading=multithreading,
        )
        for worker_count in dict.fromkeys(args.num_workers)
        for shard_count in dict.fromkeys(args.num_shards)
        for multithreading in multithreading_modes
    ]

    baseline_case = BenchmarkCase(
        num_workers=0,
        num_shards=1,
        multithreading=False,
    )
    if baseline_case not in cases:
        cases.insert(0, baseline_case)

    compression = _normalize_compression(args.compression)

    print("TorchSig sharded DatasetCreator benchmark")
    print(f"Config: {args.config.resolve()}")
    print(f"Output root: {output_root.resolve()}")
    print(f"Samples per run: {dataset_length}")
    print(f"Batch size: {args.batch_size}")
    print(f"Repeats: {args.repeats}")
    print(f"Compression: {compression or 'none'}")
    print(
        "Signal weighting: "
        f"{args.signal_weighting or cfg.signal_sampling_mode}"
    )

    all_runs: dict[BenchmarkCase, list[BenchmarkRun]] = {}

    for case in cases:
        print()
        print(
            "Running "
            f"num_workers={case.num_workers}, "
            f"num_shards={case.num_shards}, "
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
                max_batches_in_memory=args.max_batches_in_memory,
                compression=compression,
            )
            for repeat in range(args.repeats)
        ]
        all_runs[case] = runs

    baseline_seconds = statistics.median(
        run.elapsed_seconds for run in all_runs[baseline_case]
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

    print()
    print("Benchmark artifacts were written to:")
    print(output_root.resolve())


if __name__ == "__main__":
    main()


