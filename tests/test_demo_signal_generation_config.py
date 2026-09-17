"""Smoke test for the signal-generation configuration demonstration."""

import subprocess
import sys

import yaml


def test_demo_signal_generation_config(tmp_path):
    output_dir = tmp_path / "demo"

    result = subprocess.run(  # noqa: S603 - fixed interpreter and repository script
        [
            sys.executable,
            "examples/scripts/demo_signal_generation_config.py",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Generated alpha values: [0.35, 0.35, 0.35, 0.35, 0.35]" in result.stdout
    artifact = yaml.safe_load((output_dir / "generated_dataset" / "dataset_info.yaml").read_text())
    assert artifact["experiment_config"]["signals"]["qpsk"]["parameters"]["alpha"]["value"] == 0.35
