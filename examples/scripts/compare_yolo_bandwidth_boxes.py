"""Compare YOLO boxes made from canonical and estimated bandwidths.

The script generates deterministic wideband IQ samples, computes their
spectrograms, and draws the same components side by side using both supported
``YOLOLabel`` bandwidth modes.

Example:
    python examples/scripts/compare_yolo_bandwidth_boxes.py --examples 3
"""

# ruff: noqa: INP001

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from torchsig.datasets.datasets import TorchSigIterableDataset
from torchsig.transforms.functional import spectrogram
from torchsig.transforms.metadata_transforms import YOLOLabel
from torchsig.utils.defaults import TorchSigDefaults
from torchsig.utils.yaml import load_config_from_yaml

if TYPE_CHECKING:
    import numpy as np
    from matplotlib.axes import Axes

    from torchsig.signals.signal_types import Signal


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPOSITORY_ROOT / "torchsig/datasets/default_configs/wideband_clean_train_all.yaml"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "yolo_bandwidth_box_comparison.png"


def build_dataset(config_path: Path) -> tuple[TorchSigIterableDataset, int]:
    """Build a deterministic raw-IQ dataset from a TorchSig YAML config."""
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


def collect_examples(
    dataset: TorchSigIterableDataset,
    count: int,
    max_attempts: int,
) -> tuple[list[Signal], int]:
    """Collect samples for which every component has an occupied estimate."""
    samples: list[Signal] = []
    skipped = 0
    for _ in range(max_attempts):
        sample = next(dataset)
        if sample.component_signals and all(hasattr(component, "estimated_occupied_bandwidth") for component in sample.component_signals):
            samples.append(sample)
            if len(samples) == count:
                return samples, skipped
        else:
            skipped += 1

    raise RuntimeError(f"found only {len(samples)} comparable samples in {max_attempts} attempts; increase --max-attempts")


def labels_for(sample: Signal, bandwidth_key: str) -> list[tuple]:
    """Apply one YOLO bandwidth mode and return its component labels."""
    YOLOLabel(bandwidth_key=bandwidth_key)(sample)
    return [component.yolo_label for component in sample.component_signals]


def yolo_to_pixel_box(
    label: tuple,
    width_pixels: int,
    height_pixels: int,
) -> tuple[tuple[float, float], float, float]:
    """Convert a normalized YOLO label to spectrogram pixel coordinates."""
    _, x_center, y_center, width, height = label
    x_center_pixels = x_center * width_pixels - 0.5
    y_center_pixels = y_center * height_pixels - 1.0
    box_width = width * width_pixels
    box_height = height * height_pixels
    return (
        (
            x_center_pixels - box_width / 2.0,
            y_center_pixels - box_height / 2.0,
        ),
        box_width,
        box_height,
    )


def draw_example(
    axis: Axes,
    spectrogram: np.ndarray,
    sample: Signal,
    labels: list[tuple],
    title: str,
    color: str,
) -> None:
    """Draw a spectrogram and one bandwidth mode's YOLO boxes."""
    height_pixels, width_pixels = spectrogram.shape
    axis.imshow(
        spectrogram,
        aspect="auto",
        origin="upper",
        cmap="viridis",
        vmin=float(sample.noise_power_db),
        vmax=float(sample.noise_power_db + sample.snr_db_max),
    )
    for component, label in zip(sample.component_signals, labels, strict=True):
        lower_left, box_width, box_height = yolo_to_pixel_box(
            label,
            width_pixels,
            height_pixels,
        )
        axis.add_patch(
            Rectangle(
                lower_left,
                box_width,
                box_height,
                fill=False,
                edgecolor=color,
                linewidth=1.5,
            )
        )
        axis.text(
            lower_left[0],
            max(0.0, lower_left[1] - 2.0),
            str(component.class_name),
            color="white",
            fontsize=7,
            bbox={"facecolor": color, "alpha": 0.75, "pad": 1},
        )
    axis.set_title(title)
    axis.set_xlabel("Time bin")
    axis.set_ylabel("Frequency bin")


def plot_comparison(
    samples: list[Signal],
    fft_size: int,
    fft_stride: int,
    output_path: Path,
    *,
    show: bool,
) -> None:
    """Plot canonical and estimated-bandwidth boxes for each sample."""
    figure, axes = plt.subplots(
        len(samples),
        2,
        figsize=(15, 5 * len(samples)),
        squeeze=False,
    )
    for row, sample in enumerate(samples):
        sample_spectrogram = spectrogram(sample.data, fft_size, fft_stride)
        estimated_labels = labels_for(sample, "estimated_occupied_bandwidth")
        canonical_labels = labels_for(sample, "bandwidth")
        draw_example(
            axes[row, 0],
            sample_spectrogram,
            sample,
            estimated_labels,
            "Default: estimated occupied bandwidth",
            "#e66101",
        )
        draw_example(
            axes[row, 1],
            sample_spectrogram,
            sample,
            canonical_labels,
            "Optional: canonical bandwidth",
            "#5e3c99",
        )

    figure.suptitle("YOLO frequency-box bandwidth comparison", fontsize=15)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    print(f"saved comparison: {output_path}")
    if show:
        plt.show()
    plt.close(figure)


def main() -> None:
    """Generate visual examples comparing the two YOLO bandwidth modes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if args.examples <= 0:
        parser.error("--examples must be positive")
    if args.max_attempts < args.examples:
        parser.error("--max-attempts must be at least --examples")

    dataset, seed = build_dataset(args.config)
    samples, skipped = collect_examples(
        dataset,
        args.examples,
        args.max_attempts,
    )
    print(f"seed: {seed}; examples: {len(samples)}; skipped: {skipped}")
    plot_comparison(
        samples,
        int(dataset.fft_size),
        int(dataset.fft_stride),
        args.output,
        show=args.show,
    )


if __name__ == "__main__":
    main()
