"""Tests for the MR13 Stage-2 bottleneck decision gate."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[1] / "benchmarks" / "analyze_stage2_bottlenecks.py"
_SPEC = importlib.util.spec_from_file_location("stage2_bottlenecks", _SCRIPT)
assert _SPEC is not None
assert _SPEC.loader is not None
analysis = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(analysis)


def _row(*, pipeline="materialized", workers=0, rate=100.0, layout="full", compression="none", repetitions=3, stdev=2.0):
    return {
        "pipeline": pipeline,
        "access": "shuffled",
        "workers": workers,
        "compression": compression,
        "chunk_samples": 1,
        "layout": layout,
        "repetitions": repetitions,
        "median_samples_per_second": rate,
        "stdev_samples_per_second": stdev,
    }


def test_no_go_without_stable_sharding_evidence() -> None:
    payload = {
        "aggregates": [
            _row(workers=0, rate=100.0, repetitions=1, stdev=0.0),
            _row(workers=2, rate=125.0, repetitions=1, stdev=0.0),
            _row(pipeline="device_only", rate=250.0, compression="memory", repetitions=1, stdev=0.0),
        ]
    }

    result = analysis.analyze([payload])

    assert result["decision"]["status"] == "no-go"
    assert result["decision"]["mr14_recommended"] is False
    assert result["device_only_ceiling_fraction"] == pytest.approx(0.5)
    assert result["worker_scaling"][0]["ratio"] == pytest.approx(1.25)
    assert any("Fewer than three" in signal for signal in result["signals"])


def test_go_requires_stable_material_sharded_comparison() -> None:
    payload = {
        "aggregates": [
            _row(workers=0, rate=120.0),
            _row(workers=4, rate=80.0),
            _row(workers=4, rate=130.0, layout="sharded"),
            _row(pipeline="device_only", rate=200.0, compression="memory"),
        ]
    }

    result = analysis.analyze([payload])

    assert result["decision"]["status"] == "go"
    assert result["decision"]["mr14_recommended"] is True


def test_compression_and_layout_comparisons_are_matched() -> None:
    payload = {
        "aggregates": [
            _row(rate=100.0),
            _row(rate=70.0, compression="lzf"),
            _row(rate=105.0, layout="separate"),
        ]
    }

    result = analysis.analyze([payload])

    assert result["compression_comparisons"][0]["lzf_vs_none_ratio"] == pytest.approx(0.7)
    assert result["layout_comparisons"][0]["separate_vs_full_ratio"] == pytest.approx(1.05)


def test_cli_writes_reviewable_json(tmp_path) -> None:
    source = tmp_path / "results.json"
    output = tmp_path / "analysis.json"
    source.write_text(json.dumps({"aggregates": [_row(rate=100.0)]}))

    completed = subprocess.run(  # noqa: S603
        [sys.executable, str(_SCRIPT), str(source), "--output", str(output)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Decision: NO-GO for MR14" in completed.stdout
    assert json.loads(output.read_text())["analysis_version"] == 1
