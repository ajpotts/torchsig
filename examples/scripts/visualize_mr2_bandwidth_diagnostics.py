"""Reproduce and visualize the MR2 canonical-bandwidth investigation.

This script generates raw wideband samples so their component metadata remains
available for inspection. It compares each generator-selected canonical
``bandwidth`` with the threshold-based ``estimated_occupied_bandwidth`` that
would have replaced it before MR2.

Example:
    python examples/scripts/visualize_mr2_bandwidth_diagnostics.py
"""

# ruff: noqa: INP001

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from torchsig.datasets.datasets import TorchSigIterableDataset
from torchsig.utils.defaults import TorchSigDefaults
from torchsig.utils.yaml import load_config_from_yaml

if TYPE_CHECKING:
    from torchsig.signals.signal_types import Signal


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPOSITORY_ROOT / "torchsig/datasets/default_configs/wideband_clean_train_all.yaml"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "mr2_bandwidth_diagnostics.png"
DOCUMENTED_LEGACY_COMPONENTS = 394
DOCUMENTED_LEGACY_OUT_OF_RANGE = 172
TONE_BANDWIDTH_HZ = 1.0


@dataclass(frozen=True)
class ComponentRecord:
    """Metadata used to audit one generated component."""

    sample_index: int
    class_name: str
    canonical_bandwidth: float
    estimated_occupied_bandwidth: float | None
    bandwidth_min: float
    bandwidth_max: float
    canonical_out_of_range: bool
    legacy_estimate_out_of_range: bool | None


def build_dataset(config_path: Path) -> tuple[TorchSigIterableDataset, int]:
    """Build the raw deterministic dataset described by a TorchSig YAML file."""
    config = load_config_from_yaml(config_path)
    metadata = TorchSigDefaults().default_dataset_metadata.copy()
    metadata.update(config.dataset_metadata)
    dataset = TorchSigIterableDataset(
        signal_generators="all",
        metadata=metadata,
        transforms=[],
        component_transforms=[],
        target_labels=None,
        sampling_grouping=("family" if config.signal_sampling_mode == "per_family" else None),
        seed=config.seed,
    )
    return dataset, config.seed


def component_record(
    component: Signal,
    sample_index: int,
) -> ComponentRecord:
    """Extract canonical, diagnostic, and configured bandwidth information."""
    class_name = str(component.class_name)
    canonical = float(component.bandwidth)
    bandwidth_min = float(component.bandwidth_min)
    bandwidth_max = float(component.bandwidth_max)
    is_tone = class_name == "tone"
    canonical_out_of_range = canonical != TONE_BANDWIDTH_HZ if is_tone else not bandwidth_min <= canonical <= bandwidth_max

    estimated = float(component.estimated_occupied_bandwidth) if hasattr(component, "estimated_occupied_bandwidth") else None
    legacy_estimate_out_of_range = None
    if estimated is not None:
        legacy_estimate_out_of_range = estimated != TONE_BANDWIDTH_HZ if is_tone else not bandwidth_min <= estimated <= bandwidth_max

    return ComponentRecord(
        sample_index=sample_index,
        class_name=class_name,
        canonical_bandwidth=canonical,
        estimated_occupied_bandwidth=estimated,
        bandwidth_min=bandwidth_min,
        bandwidth_max=bandwidth_max,
        canonical_out_of_range=canonical_out_of_range,
        legacy_estimate_out_of_range=legacy_estimate_out_of_range,
    )


def collect_records(
    dataset: TorchSigIterableDataset,
    sample_count: int,
) -> list[ComponentRecord]:
    """Generate samples and collect metadata for every component."""
    records: list[ComponentRecord] = []
    for sample_index in range(sample_count):
        sample = next(dataset)
        records.extend(component_record(component, sample_index) for component in sample.component_signals)
    return records


def write_csv(records: list[ComponentRecord], output_path: Path) -> None:
    """Write the component audit records to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(ComponentRecord.__dataclass_fields__),
        )
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)


def print_summary(
    records: list[ComponentRecord],
    sample_count: int,
    seed: int,
) -> None:
    """Print the reproducibility and bandwidth checks from MR2."""
    classes = Counter(record.class_name for record in records)
    diagnostics = sum(record.estimated_occupied_bandwidth is not None for record in records)
    canonical_failures = sum(record.canonical_out_of_range for record in records)
    legacy_proxy_failures = sum(record.legacy_estimate_out_of_range is True for record in records)
    coverage = {
        "802.11a": sum(name.startswith("80211a") for name in classes),
        "BPSK": int("bpsk" in classes),
        "QPSK": int("qpsk" in classes),
    }

    print(f"seed: {seed}")
    print(f"samples: {sample_count}")
    print(f"components: {len(records)}")
    print(f"generated class names: {len(classes)}")
    print(f"components with occupied-bandwidth diagnostic: {diagnostics}")
    print(f"out-of-range non-tone canonical bandwidths: {canonical_failures}")
    print(f"out-of-range diagnostic estimates (legacy-behavior proxy): {legacy_proxy_failures}")
    print("required coverage: " + ", ".join(f"{name}={'yes' if found else 'no'}" for name, found in coverage.items()))
    print(f"documented pre-change baseline: {DOCUMENTED_LEGACY_OUT_OF_RANGE} / {DOCUMENTED_LEGACY_COMPONENTS} components out of range")


def plot_records(
    records: list[ComponentRecord],
    output_path: Path,
    *,
    show: bool,
) -> None:
    """Create a dashboard comparing canonical and diagnostic bandwidths."""
    estimated_records = [record for record in records if record.estimated_occupied_bandwidth is not None and record.class_name != "tone"]
    classes = Counter(record.class_name for record in records)
    canonical_failures = sum(record.canonical_out_of_range for record in records)
    legacy_proxy_failures = sum(record.legacy_estimate_out_of_range is True for record in records)

    figure, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].bar(
        ["Before MR2\n(documented)", "After MR2\n(this run)"],
        [DOCUMENTED_LEGACY_OUT_OF_RANGE, canonical_failures],
        color=["#d95f02", "#1b9e77"],
    )
    axes[0, 0].set_ylabel("Out-of-range components")
    axes[0, 0].set_title("Canonical bandwidth violations")
    for index, value in enumerate([DOCUMENTED_LEGACY_OUT_OF_RANGE, canonical_failures]):
        axes[0, 0].text(index, value, str(value), ha="center", va="bottom")

    canonical_ratios = np.asarray([record.canonical_bandwidth / record.bandwidth_max for record in estimated_records])
    estimated_ratios = np.asarray([record.estimated_occupied_bandwidth / record.bandwidth_max for record in estimated_records])
    estimate_is_out = np.asarray([record.legacy_estimate_out_of_range for record in estimated_records])
    axes[0, 1].scatter(
        canonical_ratios[~estimate_is_out],
        estimated_ratios[~estimate_is_out],
        s=18,
        alpha=0.6,
        label="diagnostic in range",
        color="#1b9e77",
    )
    axes[0, 1].scatter(
        canonical_ratios[estimate_is_out],
        estimated_ratios[estimate_is_out],
        s=22,
        alpha=0.7,
        label="diagnostic out of range",
        color="#d95f02",
    )
    axes[0, 1].axvline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0, 1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0, 1].set_xlabel("Canonical bandwidth / configured maximum")
    axes[0, 1].set_ylabel("Diagnostic bandwidth / configured maximum")
    axes[0, 1].set_title("Why the diagnostic must not become canonical")
    axes[0, 1].legend()

    ratios = estimated_ratios / canonical_ratios
    axes[1, 0].hist(ratios, bins=30, color="#7570b3", edgecolor="white")
    axes[1, 0].axvline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1, 0].set_xlabel("Diagnostic / canonical bandwidth")
    axes[1, 0].set_ylabel("Components")
    axes[1, 0].set_title("Threshold estimate differs from configured width")

    common_classes = classes.most_common(15)
    labels = [name for name, _ in reversed(common_classes)]
    counts = [count for _, count in reversed(common_classes)]
    axes[1, 1].barh(labels, counts, color="#66a61e")
    axes[1, 1].set_xlabel("Generated components")
    axes[1, 1].set_title(f"Most frequent classes ({len(classes)} unique)")

    figure.suptitle(
        f"MR2 bandwidth investigation: {len(records)} components; {legacy_proxy_failures} diagnostic estimates outside bounds",
        fontsize=14,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    print(f"saved plot: {output_path}")
    if show:
        plt.show()
    plt.close(figure)


def main() -> None:
    """Run the deterministic MR2 investigation and render its dashboard."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path for component-level audit data.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the plot in addition to saving it.",
    )
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")

    dataset, seed = build_dataset(args.config)
    records = collect_records(dataset, args.samples)
    print_summary(records, args.samples, seed)
    if args.csv is not None:
        write_csv(records, args.csv)
        print(f"saved audit CSV: {args.csv}")
    plot_records(records, args.output, show=args.show)


if __name__ == "__main__":
    main()
