"""Tests for deterministic GNU Radio example dataset generation."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import pytest

from examples.scripts import generate_gnuradio_examples as generator_module
from examples.scripts.generate_gnuradio_examples import IQDatasetGenerator


class _InlineExecutor:
    """ProcessPoolExecutor-compatible helper for dependency-free tests."""

    def __init__(self, max_workers: int) -> None:
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def map(self, _function, configs, specs):
        return [spec.metadata_row(config.audio_rate) for config, spec in zip(configs, specs, strict=True)]


def _generator(root: Path) -> IQDatasetGenerator:
    return IQDatasetGenerator(
        root=root,
        modulations={"BPSK": [1, -1], "QPSK": [1 + 1j, -1 - 1j]},
        snr_db=[-5, 5],
        duration_s=0.01,
        audio_rate=1_000,
        seed=1234,
    )


def test_record_specs_are_stable_and_have_manifest_fields(tmp_path):
    first = _generator(tmp_path / "first").record_specs()
    second = _generator(tmp_path / "second").record_specs()

    assert first == second
    assert len({spec.seed for spec in first}) == len(first)
    assert [spec.index for spec in first] == list(range(4))
    row = first[0].metadata_row(1_000)
    assert row["start_frame"] == 0
    assert row["num_frames"] == 10
    assert row["expected_sample_rate"] == 1_000
    assert row["channel_count"] == 2


def test_serial_and_parallel_sidecars_are_equivalent(tmp_path, monkeypatch):
    serial = _generator(tmp_path / "serial")
    monkeypatch.setattr(serial, "execute_record", lambda _spec: None)
    serial.generate(workers=1)

    monkeypatch.setattr(generator_module, "ProcessPoolExecutor", _InlineExecutor)
    parallel = _generator(tmp_path / "parallel")
    parallel.generate(workers=3)

    assert (serial.root / "metadata.csv").read_text() == (parallel.root / "metadata.csv").read_text()
    assert (serial.root / "info.json").read_text() == (parallel.root / "info.json").read_text()
    with (parallel.root / "metadata.csv").open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert [int(row["index"]) for row in rows] == list(range(4))


def test_failed_record_is_not_published(tmp_path, monkeypatch):
    generator = _generator(tmp_path)
    spec = generator.record_specs()[0]

    def fail_after_partial_write(_spec, temporary):
        temporary.write_bytes(b"partial")
        raise RuntimeError("worker failed")

    monkeypatch.setattr(generator, "generate_one", fail_after_partial_write)
    with pytest.raises(RuntimeError, match="worker failed"):
        generator.generate()

    assert not (tmp_path / spec.relative_path).exists()
    assert not (tmp_path / "metadata.csv").exists()
    assert not (tmp_path / "info.json").exists()
    assert not list((tmp_path / Path(spec.relative_path).parent).glob("*.tmp"))


def test_logging_respects_configured_verbosity(tmp_path, monkeypatch, caplog):
    generator = _generator(tmp_path)
    monkeypatch.setattr(generator, "execute_record", lambda _spec: None)

    with caplog.at_level(logging.DEBUG, logger=generator_module.__name__):
        generator.generate(verbosity=logging.WARNING)
    assert not caplog.records

    with caplog.at_level(logging.INFO, logger=generator_module.__name__):
        generator.generate(verbosity=logging.INFO)
    assert any("Generated 4 records" in record.message for record in caplog.records)


@pytest.mark.parametrize("workers", [0, -1])
def test_invalid_worker_count_is_rejected(tmp_path, workers):
    with pytest.raises(ValueError, match="at least 1"):
        _generator(tmp_path).generate(workers=workers)
