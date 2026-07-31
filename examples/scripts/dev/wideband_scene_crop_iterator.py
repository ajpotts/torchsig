"""Display wideband scenes before and after experimental viewport cropping.

The example loads TorchSIG's default clean wideband training configuration,
expands its time-frequency canvas, and generates IQ scenes with a
``WorkerSeedingDataLoader``. Each canvas yielded by the PyTorch iterator is
then cropped back to the default configuration's sample rate and sample count.

Example:
    python examples/scripts/dev/wideband_scene_crop_iterator.py --no-show
"""

# ruff: noqa: INP001

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from torchsig.datasets.datasets import SafeTorchSigIterableDataset
from torchsig.transforms.impairments import Impairments
from torchsig.transforms.prototype_scene_crop import PrototypeSceneCrop
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from torchsig.utils.defaults import TorchSigDefaults
from torchsig.utils.dsp import compute_spectrogram
from torchsig.utils.writer import identity_collate_fn
from torchsig.utils.yaml import load_config_from_yaml

if TYPE_CHECKING:
    import numpy as np

    from torchsig.signals.signal_types import Signal

DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[3]
    / "torchsig/datasets/default_configs/wideband_clean_train_all.yaml"
)
CANVAS_SAMPLE_RATE_SCALE = 2
CANVAS_NUM_SAMPLES_SCALE = 4


def parse_args() -> argparse.Namespace:
    """Parse generation and visualization options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Wideband dataset YAML configuration.",
    )
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--save",
        type=Path,
        default=Path("wideband_scene_crop_iterator.png"),
    )
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.num_samples <= 0:
        parser.error("--num-samples must be positive")
    return args


def build_canvas_dataset(config_path: Path) -> tuple[SafeTorchSigIterableDataset, dict]:
    """Build an expanded-IQ variant of the default wideband dataset."""
    cfg = load_config_from_yaml(config_path)
    viewport_metadata = TorchSigDefaults().default_dataset_metadata
    viewport_metadata.update(cfg.dataset_metadata)
    canvas_metadata = dict(viewport_metadata)

    viewport_sample_rate = float(viewport_metadata["sample_rate"])
    viewport_num_samples = int(viewport_metadata["num_iq_samples_dataset"])
    canvas_sample_rate = viewport_sample_rate * CANVAS_SAMPLE_RATE_SCALE
    canvas_metadata.update(
        {
            "sample_rate": canvas_sample_rate,
            "num_iq_samples_dataset": (
                viewport_num_samples * CANVAS_NUM_SAMPLES_SCALE
            ),
            "frequency_min": -canvas_sample_rate / 2,
            "frequency_max": canvas_sample_rate / 2 - 1,
            "signal_center_freq_min": (
                float(viewport_metadata["signal_center_freq_min"])
                * CANVAS_SAMPLE_RATE_SCALE
            ),
            "signal_center_freq_max": (
                float(viewport_metadata["signal_center_freq_max"])
                * CANVAS_SAMPLE_RATE_SCALE
            ),
        }
    )

    impairments = Impairments(level=cfg.impairment_level)
    dataset = SafeTorchSigIterableDataset(
        signal_generators="all",
        metadata=canvas_metadata,
        transforms=[impairments.dataset_transforms],
        component_transforms=[impairments.signal_transforms],
    )
    return dataset, viewport_metadata


def spectrogram(signal: Signal, fft_size: int) -> np.ndarray:
    """Compute a display spectrogram without changing the generated signal."""
    return compute_spectrogram(signal.data, fft_size, fft_size)


def draw_components(axis, signal: Signal) -> None:
    """Overlay component metadata boxes in physical coordinates."""
    for component in signal.component_signals:
        # Derive edges from canonical metadata because generated scenes can
        # contain stale cached lower_freq/upper_freq values.
        lower_freq = component.center_freq - component.bandwidth / 2
        upper_freq = component.center_freq + component.bandwidth / 2
        rectangle = Rectangle(
            (
                component.start_in_samples,
                lower_freq / 1e6,
            ),
            component.duration_in_samples,
            component.bandwidth / 1e6,
            fill=False,
            edgecolor="white",
            linewidth=1.0,
        )
        axis.add_patch(rectangle)
        axis.text(
            component.start_in_samples,
            upper_freq / 1e6,
            component.class_name,
            color="white",
            fontsize=6,
            va="bottom",
        )


def show_signal(axis, signal: Signal, fft_size: int, title: str) -> None:
    """Show a signal spectrogram and its component metadata."""
    axis.imshow(
        spectrogram(signal, fft_size),
        # compute_spectrogram returns positive-frequency rows first.
        origin="upper",
        aspect="auto",
        extent=(
            0,
            len(signal.data),
            -signal.sample_rate / 2e6,
            signal.sample_rate / 2e6,
        ),
        cmap="viridis",
    )
    draw_components(axis, signal)
    axis.set_title(title)
    axis.set_xlabel("IQ sample")
    axis.set_ylabel("Frequency (MHz)")


def draw_selected_viewport(axis, cropped: Signal) -> None:
    """Draw the selected crop in the expanded canvas coordinates."""
    decimation = int(cropped["scene_crop_decimation"])
    rectangle = Rectangle(
        (
            cropped["scene_crop_input_time_start"],
            (
                cropped["scene_crop_input_center_freq"]
                - cropped.sample_rate / 2
            )
            / 1e6,
        ),
        len(cropped.data) * decimation,
        cropped.sample_rate / 1e6,
        fill=False,
        edgecolor="red",
        linewidth=2.0,
    )
    axis.add_patch(rectangle)


def collect_pairs(
    dataset: SafeTorchSigIterableDataset,
    viewport_metadata: dict,
    *,
    num_samples: int,
    seed: int,
) -> list[tuple[Signal, Signal]]:
    """Read canvases from a PyTorch iterator and crop each one."""
    loader = WorkerSeedingDataLoader(
        dataset,
        seed=seed,
        batch_size=1,
        num_workers=0,
        collate_fn=identity_collate_fn,
    )
    iterator = iter(loader)
    pairs = []
    for index in range(num_samples):
        canvas = next(iterator)[0]
        crop = PrototypeSceneCrop(
            num_iq_samples=int(viewport_metadata["num_iq_samples_dataset"]),
            sample_rate=float(viewport_metadata["sample_rate"]),
            allow_empty=True,
            seed=seed + index,
        )
        pairs.append((canvas, crop(canvas)))
    return pairs


def plot_pairs(
    pairs: list[tuple[Signal, Signal]],
    *,
    fft_size: int,
    save_path: Path,
) -> None:
    """Display one before/after spectrogram pair per row."""
    figure, axes = plt.subplots(
        len(pairs),
        2,
        figsize=(15, 4.5 * len(pairs)),
        squeeze=False,
        constrained_layout=True,
    )
    for index, (canvas, cropped) in enumerate(pairs):
        show_signal(
            axes[index, 0],
            canvas,
            fft_size,
            f"Sample {index}: expanded wideband canvas",
        )
        show_signal(
            axes[index, 1],
            cropped,
            fft_size,
            f"Sample {index}: receiver viewport",
        )
        draw_selected_viewport(axes[index, 0], cropped)
        axes[index, 0].text(
            0.01,
            0.02,
            f"{len(canvas.component_signals)} generated components",
            color="white",
            transform=axes[index, 0].transAxes,
        )
        axes[index, 1].text(
            0.01,
            0.02,
            f"{len(cropped.component_signals)} visible components",
            color="white",
            transform=axes[index, 1].transAxes,
        )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=150)


def main() -> None:
    """Generate, crop, and display several default wideband samples."""
    args = parse_args()
    cfg = load_config_from_yaml(args.config)
    seed = cfg.seed if args.seed is None else args.seed
    dataset, viewport_metadata = build_canvas_dataset(args.config)
    pairs = collect_pairs(
        dataset,
        viewport_metadata,
        num_samples=args.num_samples,
        seed=seed,
    )
    plot_pairs(
        pairs,
        fft_size=int(viewport_metadata["fft_size"]),
        save_path=args.save,
    )

    print(f"Configuration: {args.config.resolve()}")
    for index, (canvas, cropped) in enumerate(pairs):
        print(
            f"Sample {index}: {len(canvas.component_signals)} canvas components, "
            f"{len(cropped.component_signals)} visible after crop"
        )
    print(f"Visualization: {args.save.resolve()}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
