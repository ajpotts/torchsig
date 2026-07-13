#!/usr/bin/env python3
"""Benchmark TorchSig HDF5 generation and writing.

This benchmark separates three costs:

1. Generation only
   Measures sequential generation versus a persistent process pool.

2. Writer only
   Pre-generates all batches once, then measures only HDF5 submission and
   teardown.

3. End to end
   Uses a persistent process pool for generation while the parent submits
   completed batches to one HDF5Writer.

Unlike the earlier benchmark, this script does not spawn one process per batch.
Each worker process stays alive and generates multiple batches.

Example:

    python benchmark_hdf5_multiprocessing.py \
        --samples 256 \
        --sample-length 131072 \
        --batch-size 8 \
        --workers 1 2 4 8 \
        --fft-passes 3 \
        --repeats 3 \
        --output-root /tmp/torchsig_hdf5_benchmark
"""

from __future__ import annotations

import argparse
import concurrent.futures
import multiprocessing as mp
import shutil
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

from torchsig.signals.signal_types import Signal
from torchsig.utils.file_handlers.hdf5 import HDF5Writer


@dataclass(frozen=True)
class BenchmarkConfig:
    samples: int
    sample_length: int
    batch_size: int
    fft_passes: int
    compression: str | None
    max_batches_in_memory: int
    repeats: int


@dataclass(frozen=True)
class TimedResult:
    workers: int
    mode: str
    elapsed_seconds: float
    samples_per_second: float
    speedup: float
    output_path: Path | None = None
    file_size_bytes: int | None = None


def _normalize_compression(value: str) -> str | None:
    normalized = value.strip().lower()
    if normalized in {"none", "off", "false", "0"}:
        return None
    return normalized


def _iter_batch_specs(
    samples: int,
    batch_size: int,
) -> Iterable[tuple[int, tuple[int, ...]]]:
    batch_idx = 0
    for start in range(0, samples, batch_size):
        stop = min(start + batch_size, samples)
        yield batch_idx, tuple(range(start, stop))
        batch_idx += 1


def _generate_signal(
    sample_index: int,
    sample_length: int,
    fft_passes: int,
) -> Signal:
    rng = np.random.default_rng(sample_index)

    iq = (
        rng.standard_normal(sample_length)
        + 1j * rng.standard_normal(sample_length)
    ).astype(np.complex64)

    frequency = np.fft.fftfreq(sample_length)
    response = (
        1.0
        + 0.12 * np.cos(2.0 * np.pi * 7.0 * frequency)
        + 0.05 * np.sin(2.0 * np.pi * 19.0 * frequency)
    ).astype(np.float32)

    for _ in range(fft_passes):
        spectrum = np.fft.fft(iq)
        iq = np.fft.ifft(spectrum * response).astype(np.complex64)

    return Signal(
        data=iq,
        metadata={
            "sample_index": sample_index,
            "duration_in_samples": sample_length,
        },
    )


def _generate_batch(
    batch_idx: int,
    sample_indices: tuple[int, ...],
    sample_length: int,
    fft_passes: int,
) -> tuple[int, list[Signal]]:
    batch = [
        _generate_signal(
            sample_index=sample_index,
            sample_length=sample_length,
            fft_passes=fft_passes,
        )
        for sample_index in sample_indices
    ]
    return batch_idx, batch


def _generate_batches_sequential(
    config: BenchmarkConfig,
) -> list[tuple[int, list[Signal]]]:
    return [
        _generate_batch(
            batch_idx=batch_idx,
            sample_indices=sample_indices,
            sample_length=config.sample_length,
            fft_passes=config.fft_passes,
        )
        for batch_idx, sample_indices in _iter_batch_specs(
            config.samples,
            config.batch_size,
        )
    ]


def _generate_batches_parallel(
    config: BenchmarkConfig,
    workers: int,
) -> list[tuple[int, list[Signal]]]:
    context = mp.get_context("spawn")
    specs = list(_iter_batch_specs(config.samples, config.batch_size))
    results: list[tuple[int, list[Signal]]] = []

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
    ) as executor:
        futures = [
            executor.submit(
                _generate_batch,
                batch_idx,
                sample_indices,
                config.sample_length,
                config.fft_passes,
            )
            for batch_idx, sample_indices in specs
        ]

        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    return results


def _write_batches(
    config: BenchmarkConfig,
    batches: list[tuple[int, list[Signal]]],
    root: Path,
) -> tuple[float, Path, int]:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    writer = HDF5Writer(
        root=root,
        compression=config.compression,
        shuffle=config.compression is not None,
        fletcher32=False,
        max_batches_in_memory=config.max_batches_in_memory,
        multiprocessing_context="spawn",
    )

    start = time.perf_counter()
    for batch_idx, batch in batches:
        writer.write(batch_idx, batch)
    writer.teardown()
    elapsed = time.perf_counter() - start

    output_path = root / "data.h5"
    if not output_path.exists():
        raise RuntimeError(f"Expected output was not created: {output_path}")

    with h5py.File(output_path, "r") as hdf5_file:
        indexed_samples = len(hdf5_file["index"])
        if indexed_samples != config.samples:
            raise RuntimeError(
                f"{output_path} contains {indexed_samples} indexed samples; "
                f"expected {config.samples}"
            )

    return elapsed, output_path, output_path.stat().st_size


def _benchmark_generation(
    config: BenchmarkConfig,
    workers: int,
) -> float:
    start = time.perf_counter()

    if workers == 1:
        batches = _generate_batches_sequential(config)
    else:
        batches = _generate_batches_parallel(config, workers)

    elapsed = time.perf_counter() - start

    generated = sum(len(batch) for _, batch in batches)
    if generated != config.samples:
        raise RuntimeError(
            f"Generated {generated} samples; expected {config.samples}"
        )

    return elapsed


def _benchmark_writer_only(
    config: BenchmarkConfig,
    batches: list[tuple[int, list[Signal]]],
    root: Path,
) -> tuple[float, Path, int]:
    return _write_batches(config, batches, root)


def _benchmark_end_to_end(
    config: BenchmarkConfig,
    workers: int,
    root: Path,
) -> tuple[float, Path, int]:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    writer = HDF5Writer(
        root=root,
        compression=config.compression,
        shuffle=config.compression is not None,
        fletcher32=False,
        max_batches_in_memory=config.max_batches_in_memory,
        multiprocessing_context="spawn",
    )

    start = time.perf_counter()

    if workers == 1:
        for batch_idx, sample_indices in _iter_batch_specs(
            config.samples,
            config.batch_size,
        ):
            _, batch = _generate_batch(
                batch_idx,
                sample_indices,
                config.sample_length,
                config.fft_passes,
            )
            writer.write(batch_idx, batch)
    else:
        context = mp.get_context("spawn")
        specs = list(_iter_batch_specs(config.samples, config.batch_size))

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
        ) as executor:
            futures = [
                executor.submit(
                    _generate_batch,
                    batch_idx,
                    sample_indices,
                    config.sample_length,
                    config.fft_passes,
                )
                for batch_idx, sample_indices in specs
            ]

            for future in concurrent.futures.as_completed(futures):
                batch_idx, batch = future.result()
                writer.write(batch_idx, batch)

    writer.teardown()
    elapsed = time.perf_counter() - start

    output_path = root / "data.h5"
    if not output_path.exists():
        raise RuntimeError(f"Expected output was not created: {output_path}")

    with h5py.File(output_path, "r") as hdf5_file:
        indexed_samples = len(hdf5_file["index"])
        if indexed_samples != config.samples:
            raise RuntimeError(
                f"{output_path} contains {indexed_samples} indexed samples; "
                f"expected {config.samples}"
            )

    return elapsed, output_path, output_path.stat().st_size


def _median(values: list[float]) -> float:
    return statistics.median(values)


def _run_benchmarks(
    config: BenchmarkConfig,
    worker_counts: list[int],
    output_root: Path,
) -> list[TimedResult]:
    results: list[TimedResult] = []

    # Generation-only measurements
    generation_times: dict[int, float] = {}
    for workers in worker_counts:
        runs = [
            _benchmark_generation(config, workers)
            for _ in range(config.repeats)
        ]
        generation_times[workers] = _median(runs)

    generation_baseline = generation_times[1]
    for workers in worker_counts:
        elapsed = generation_times[workers]
        results.append(
            TimedResult(
                workers=workers,
                mode="generation",
                elapsed_seconds=elapsed,
                samples_per_second=config.samples / elapsed,
                speedup=generation_baseline / elapsed,
            )
        )

    # Pre-generate once for writer-only timing.
    writer_batches = _generate_batches_sequential(config)
    writer_runs: list[tuple[float, Path, int]] = []

    for repeat in range(config.repeats):
        writer_runs.append(
            _benchmark_writer_only(
                config,
                writer_batches,
                output_root / "writer_only" / f"repeat_{repeat}",
            )
        )

    writer_elapsed = _median([run[0] for run in writer_runs])
    representative_writer_run = min(
        writer_runs,
        key=lambda run: abs(run[0] - writer_elapsed),
    )
    results.append(
        TimedResult(
            workers=1,
            mode="writer",
            elapsed_seconds=writer_elapsed,
            samples_per_second=config.samples / writer_elapsed,
            speedup=1.0,
            output_path=representative_writer_run[1],
            file_size_bytes=representative_writer_run[2],
        )
    )

    # End-to-end measurements
    end_to_end_times: dict[int, tuple[float, Path, int]] = {}
    for workers in worker_counts:
        runs = [
            _benchmark_end_to_end(
                config,
                workers,
                output_root
                / "end_to_end"
                / f"workers_{workers}"
                / f"repeat_{repeat}",
            )
            for repeat in range(config.repeats)
        ]

        median_elapsed = _median([run[0] for run in runs])
        representative = min(
            runs,
            key=lambda run: abs(run[0] - median_elapsed),
        )
        end_to_end_times[workers] = (
            median_elapsed,
            representative[1],
            representative[2],
        )

    end_to_end_baseline = end_to_end_times[1][0]
    for workers in worker_counts:
        elapsed, output_path, file_size = end_to_end_times[workers]
        results.append(
            TimedResult(
                workers=workers,
                mode="end_to_end",
                elapsed_seconds=elapsed,
                samples_per_second=config.samples / elapsed,
                speedup=end_to_end_baseline / elapsed,
                output_path=output_path,
                file_size_bytes=file_size,
            )
        )

    return results


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "-"

    size = float(value)
    for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or suffix == "TiB":
            return f"{size:.2f} {suffix}"
        size /= 1024.0
    return f"{size:.2f} TiB"


def _print_mode_table(
    title: str,
    rows: list[TimedResult],
) -> None:
    print()
    print(title)
    print(
        f"{'Workers':>7}  "
        f"{'Time (s)':>10}  "
        f"{'Samples/s':>12}  "
        f"{'Speedup':>8}  "
        f"{'File size':>11}"
    )
    print("-" * 60)

    for row in rows:
        print(
            f"{row.workers:>7d}  "
            f"{row.elapsed_seconds:>10.3f}  "
            f"{row.samples_per_second:>12.2f}  "
            f"{row.speedup:>7.2f}x  "
            f"{_format_bytes(row.file_size_bytes):>11}"
        )


def _print_results(results: list[TimedResult]) -> None:
    generation_rows = [
        result for result in results if result.mode == "generation"
    ]
    writer_rows = [
        result for result in results if result.mode == "writer"
    ]
    end_to_end_rows = [
        result for result in results if result.mode == "end_to_end"
    ]

    _print_mode_table("Generation only", generation_rows)
    _print_mode_table("Writer only", writer_rows)
    _print_mode_table("End to end", end_to_end_rows)

    fastest = min(
        end_to_end_rows,
        key=lambda result: result.elapsed_seconds,
    )
    baseline = next(
        result for result in end_to_end_rows if result.workers == 1
    )

    print()
    print(
        f"Best end-to-end result: {fastest.workers} worker(s), "
        f"{baseline.elapsed_seconds / fastest.elapsed_seconds:.2f}x speedup"
    )
    if fastest.output_path is not None:
        print(
            "Representative HDF5 output: "
            f"{fastest.output_path.resolve()}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark TorchSig generation, HDF5 writing, and the combined "
            "multiprocessing pipeline."
        )
    )
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--sample-length", type=int, default=65536)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=[1, 2, 4],
    )
    parser.add_argument("--fft-passes", type=int, default=2)
    parser.add_argument("--compression", default="none")
    parser.add_argument("--max-batches-in-memory", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--keep-output", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "samples",
        "sample_length",
        "batch_size",
        "fft_passes",
        "max_batches_in_memory",
        "repeats",
    ):
        if getattr(args, name) < 1:
            raise ValueError(
                f"--{name.replace('_', '-')} must be at least 1"
            )

    if 1 not in args.workers:
        raise ValueError(
            "--workers must include 1 to establish the baseline"
        )

    if any(worker_count < 1 for worker_count in args.workers):
        raise ValueError("Every --workers value must be at least 1")


def main() -> None:
    args = _parse_args()
    _validate_args(args)

    temporary_root: Path | None = None
    if args.output_root is None:
        temporary_root = Path(
            tempfile.mkdtemp(prefix="torchsig_hdf5_benchmark_")
        )
        output_root = temporary_root
    else:
        output_root = args.output_root.resolve()
        output_root.mkdir(parents=True, exist_ok=True)

    config = BenchmarkConfig(
        samples=args.samples,
        sample_length=args.sample_length,
        batch_size=args.batch_size,
        fft_passes=args.fft_passes,
        compression=_normalize_compression(args.compression),
        max_batches_in_memory=args.max_batches_in_memory,
        repeats=args.repeats,
    )

    worker_counts = list(dict.fromkeys(args.workers))

    print("TorchSig HDF5 multiprocessing benchmark")
    print(f"Output root: {output_root.resolve()}")
    print(f"Samples: {config.samples}")
    print(f"Sample length: {config.sample_length}")
    print(f"Batch size: {config.batch_size}")
    print(f"FFT passes: {config.fft_passes}")
    print(f"Compression: {config.compression or 'none'}")
    print(f"Repeats: {config.repeats}")

    results = _run_benchmarks(
        config=config,
        worker_counts=worker_counts,
        output_root=output_root,
    )
    _print_results(results)

    if temporary_root is not None and not args.keep_output:
        print()
        print(
            "Temporary output remains available for inspection at:"
        )
        print(temporary_root.resolve())
        print(
            "Use --output-root PATH to choose a persistent location."
        )


if __name__ == "__main__":
    main()


