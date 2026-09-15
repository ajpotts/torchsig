#!/usr/bin/env python3
"""Generate deterministic TorchSig-compatible WAV or SigMF IQ datasets."""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import logging
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.io import wavfile

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

LOGGER = logging.getLogger(__name__)
METADATA_FIELDS = (
    "index", "label", "modcod", "sample_rate", "snr_db", "seed",
    "file_path", "start_frame", "num_frames", "expected_sample_rate", "channel_count",
)


def _default_modulations() -> dict[str, np.ndarray]:
    """Return the default unit-power constellations."""
    return {
        "BPSK": np.array([1 + 0j, -1 + 0j]),
        "QPSK": np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2),
        "8PSK": np.exp(1j * np.arange(8) * 2 * np.pi / 8),
    }


@dataclass(frozen=True)
class RecordSpec:
    """Immutable description of one independently generated record."""

    index: int
    mod_name: str
    modcod: int
    snr_db: float
    seed: int
    relative_path: str
    num_frames: int

    def metadata_row(self, sample_rate: int) -> dict[str, object]:
        """Return this record in the versioned sidecar schema."""
        return {
            "index": self.index, "label": self.mod_name, "modcod": self.modcod,
            "sample_rate": sample_rate, "snr_db": self.snr_db, "seed": self.seed,
            "file_path": self.relative_path, "start_frame": 0, "num_frames": self.num_frames,
            "expected_sample_rate": sample_rate, "channel_count": 2,
        }


@dataclass(frozen=True)
class _WorkerConfig:
    root: str
    modulations: dict[str, np.ndarray]
    audio_rate: int
    base_rate: int
    output_format: str


def stable_record_seed(base_seed: int, index: int, mod_name: str, snr_db: float) -> int:
    """Derive a stable uint32 seed without Python's randomized ``hash()``."""
    value = json.dumps([base_seed, index, mod_name, float(snr_db)], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.blake2s(value, digest_size=4).digest(), "big")


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    return Path(name)


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = _temporary_path(path)
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _generate_record_worker(config: _WorkerConfig, spec: RecordSpec) -> dict[str, object]:
    """Generate one record with process-local state and return its metadata."""
    generator = IQDatasetGenerator(
        config.root, config.modulations, snr_db=[], base_rate=config.base_rate,
        audio_rate=config.audio_rate, output_format=config.output_format,
    )
    generator.execute_record(spec)
    return spec.metadata_row(config.audio_rate)


class IQDatasetGenerator:
    """Generate a deterministic, optionally parallel stereo-IQ dataset.

    Record specifications and seeds are fixed before execution. Each worker
    constructs its own GNU Radio flowgraph, and only the parent publishes the
    ordered metadata sidecars.
    """

    def __init__(
        self,
        root: str | Path,
        modulations: Mapping[str, np.ndarray] | None = None,
        modcod_map: Mapping[str, int] | None = None,
        snr_db: Sequence[float] | None = None,
        duration_s: float = 1.0,
        base_rate: int = 1_000_000,
        audio_rate: int = 48_000,
        seed: int = 20230612,
        chunk_size: int = 1,
        output_format: str = "wav",
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.output_format = output_format.lower()
        if self.output_format not in {"wav", "sigmf"}:
            raise ValueError("output_format must be either 'wav' or 'sigmf'")
        self.modulations = {name: np.asarray(points) for name, points in (modulations or _default_modulations()).items()}
        if not self.modulations:
            raise ValueError("modulations must not be empty")
        self.mod_names = list(self.modulations)
        self.modcod_map = dict(modcod_map or {name: index for index, name in enumerate(self.mod_names)})
        if set(self.modcod_map) != set(self.mod_names):
            raise ValueError("modcod_map keys must exactly match modulations")
        self.snr_db = list(snr_db if snr_db is not None else [-5, 0, 5, 10, 15, 20])
        self.duration_s, self.base_rate, self.audio_rate = float(duration_s), int(base_rate), int(audio_rate)
        self.seed, self.chunk_size = int(seed), int(chunk_size)
        if self.duration_s <= 0 or self.base_rate <= 0 or self.audio_rate <= 0:
            raise ValueError("duration_s, base_rate, and audio_rate must be positive")
        self.metadata_rows: list[dict[str, object]] = []
        self.info: dict[str, Any] = {}

    def record_specs(self) -> tuple[RecordSpec, ...]:
        """Build deterministic record specifications without executing them."""
        suffix = "wav" if self.output_format == "wav" else "sigmf-data"
        num_frames = int(self.audio_rate * self.duration_s)
        specs = []
        pairs = ((name, snr) for name in self.mod_names for snr in self.snr_db)
        for index, (name, snr) in enumerate(pairs):
            record_seed = stable_record_seed(self.seed, index, name, snr)
            filename = f"{name}_{snr:+03.0f}dB_seed{record_seed}.{suffix}"
            specs.append(RecordSpec(index, name, self.modcod_map[name], float(snr), record_seed, (Path(name) / filename).as_posix(), num_frames))
        return tuple(specs)

    def _worker_config(self) -> _WorkerConfig:
        return _WorkerConfig(str(self.root), self.modulations, self.audio_rate, self.base_rate, self.output_format)

    def generate_one(self, spec: RecordSpec, output_path: Path) -> None:
        """Generate one record using a new GNU Radio flowgraph."""
        try:
            blocks = importlib.import_module("gnuradio.blocks")
            channels = importlib.import_module("gnuradio.channels")
            digital = importlib.import_module("gnuradio.digital")
            gr_filter = importlib.import_module("gnuradio.filter")
            gr = importlib.import_module("gnuradio.gr")
        except ModuleNotFoundError as error:
            raise RuntimeError("GNU Radio is required to execute IQ record generation") from error

        constellation = self.modulations[spec.mod_name]
        bits_per_symbol = int(np.ceil(np.log2(constellation.size)))
        bits = np.random.default_rng(spec.seed).integers(0, 2, spec.num_frames * bits_per_symbol, dtype=np.uint8)
        flowgraph = gr.top_block()
        source = blocks.vector_source_b(bits.tolist(), repeat=False)
        packer = blocks.pack_k_bits_bb(bits_per_symbol)
        mapper = digital.chunks_to_symbols_bc(constellation.tolist(), 1)
        channel = channels.channel_model(
            noise_voltage=float(np.sqrt(0.5 * 10 ** (-spec.snr_db / 10.0))),
            frequency_offset=0.0, epsilon=1.0, taps=[1.0], noise_seed=spec.seed & 0x7FFFFFFF,
        )
        divisor = gcd(self.audio_rate, self.base_rate)
        resampler = gr_filter.rational_resampler_ccc(
            interpolation=self.audio_rate // divisor, decimation=self.base_rate // divisor, taps=[1.0],
        )
        splitter, sink_i, sink_q = blocks.complex_to_float(), blocks.vector_sink_f(), blocks.vector_sink_f()
        head_i, head_q = blocks.head(gr.sizeof_float, spec.num_frames), blocks.head(gr.sizeof_float, spec.num_frames)
        flowgraph.connect(source, packer, mapper, channel, resampler, splitter)
        flowgraph.connect((splitter, 0), head_i, sink_i)
        flowgraph.connect((splitter, 1), head_q, sink_q)
        flowgraph.run()
        i_data, q_data = np.asarray(sink_i.data(), dtype=np.float32), np.asarray(sink_q.data(), dtype=np.float32)
        if len(i_data) < spec.num_frames:
            padding = spec.num_frames - len(i_data)
            i_data, q_data = np.pad(i_data, (0, padding)), np.pad(q_data, (0, padding))
        self._write_iq_recording(output_path, i_data[:spec.num_frames], q_data[:spec.num_frames], spec)

    def _write_iq_recording(self, path: Path, i_data: np.ndarray, q_data: np.ndarray, spec: RecordSpec) -> None:
        if self.output_format == "wav":
            wavfile.write(path, self.audio_rate, np.column_stack((i_data, q_data)).astype(np.float32))
            return
        (i_data.astype(np.float32) + 1j * q_data.astype(np.float32)).astype(np.complex64).tofile(path)
        metadata = {
            "global": {"core:datatype": "cf32_le", "core:sample_rate": self.audio_rate, "core:version": "1.0.0", "torchsig:seed": spec.seed},
            "captures": [{"core:sample_start": 0, "core:frequency": 0}],
            "annotations": [{"core:sample_start": 0, "core:sample_count": spec.num_frames, "core:label": spec.mod_name}],
        }
        _atomic_write_text(path.with_suffix(".sigmf-meta"), json.dumps(metadata, indent=2) + "\n")

    def execute_record(self, spec: RecordSpec) -> None:
        """Generate and atomically publish one record specification."""
        destination = self.root / spec.relative_path
        if destination.is_file():
            LOGGER.debug("Record already exists: %s", destination)
            return
        LOGGER.info("Generating record %d at %s", spec.index, destination)
        temporary = _temporary_path(destination)
        temporary_metadata = temporary.with_suffix(".sigmf-meta")
        try:
            self.generate_one(spec, temporary)
            temporary.replace(destination)
            if self.output_format == "sigmf":
                temporary_metadata.replace(destination.with_suffix(".sigmf-meta"))
        finally:
            temporary.unlink(missing_ok=True)
            temporary_metadata.unlink(missing_ok=True)

    def _write_sidecars(self, rows: list[dict[str, object]]) -> None:
        metadata_path = self.root / "metadata.csv"
        temporary = _temporary_path(metadata_path)
        try:
            with temporary.open("w", encoding="utf-8", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=METADATA_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            temporary.replace(metadata_path)
        finally:
            temporary.unlink(missing_ok=True)
        self.info = {
            "num_iq_samples": int(self.audio_rate * self.duration_s), "elements_per_file": 1,
            "num_files": len(rows), "size": len(rows), "class_list": self.mod_names,
            "sample_rate": self.audio_rate, "output_format": self.output_format,
            "datatype": "cf32_le" if self.output_format == "sigmf" else "float32_stereo",
            "sidecar_schema_version": "1.0",
        }
        _atomic_write_text(self.root / "info.json", json.dumps(self.info, indent=2) + "\n")

    def generate(self, workers: int = 1, verbosity: int | None = None) -> None:
        """Generate records and atomically publish ordered sidecar files."""
        if workers < 1:
            raise ValueError("workers must be at least 1")
        if verbosity is not None:
            LOGGER.setLevel(verbosity)
        self.root.mkdir(parents=True, exist_ok=True)
        specs = self.record_specs()
        if workers == 1:
            for spec in specs:
                self.execute_record(spec)
            rows = [spec.metadata_row(self.audio_rate) for spec in specs]
        else:
            config = self._worker_config()
            with ProcessPoolExecutor(max_workers=workers) as executor:
                rows = list(executor.map(_generate_record_worker, [config] * len(specs), specs))
        rows.sort(key=lambda row: int(row["index"]))
        self.metadata_rows = rows
        self._write_sidecars(rows)
        LOGGER.info("Generated %d records in %s", len(rows), self.root)


def main() -> None:
    """Generate the default example dataset."""
    logging.basicConfig(level=logging.INFO)
    IQDatasetGenerator(root="training").generate()


if __name__ == "__main__":
    main()
