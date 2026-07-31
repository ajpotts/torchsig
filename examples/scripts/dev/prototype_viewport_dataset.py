"""Save default wideband scenes from the integrated viewport prototype.

Example:
    python examples/scripts/dev/prototype_viewport_dataset.py
"""

# ruff: noqa: INP001

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from torchsig.datasets.prototype_viewport_dataset import PrototypeViewportDataset
from torchsig.transforms.impairments import Impairments
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from torchsig.utils.defaults import TorchSigDefaults
from torchsig.utils.dsp import compute_spectrogram
from torchsig.utils.writer import identity_collate_fn
from torchsig.utils.yaml import load_config_from_yaml

DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[3]
    / "torchsig/datasets/default_configs/wideband_clean_train_all.yaml"
)


def parse_args() -> argparse.Namespace:
    """Parse generation and display options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument(
        "--save",
        type=Path,
        default=Path("prototype_viewport_dataset.png"),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the saved visualization in an interactive window.",
    )
    args = parser.parse_args()
    if args.num_samples <= 0:
        parser.error("--num-samples must be positive")
    return args


def build_dataset(config_path: Path) -> tuple[PrototypeViewportDataset, dict, int]:
    """Build an expanded canvas generator with default-sized output viewports."""
    cfg = load_config_from_yaml(config_path)
    viewport = TorchSigDefaults().default_dataset_metadata
    viewport.update(cfg.dataset_metadata)
    canvas = dict(viewport)
    canvas_rate = float(viewport["sample_rate"]) * 2
    canvas.update(
        {
            "sample_rate": canvas_rate,
            "num_iq_samples_dataset": int(viewport["num_iq_samples_dataset"]) * 4,
            "frequency_min": -canvas_rate / 2,
            "frequency_max": canvas_rate / 2 - 1,
            "signal_center_freq_min": (
                float(viewport["signal_center_freq_min"]) * 2
            ),
            "signal_center_freq_max": (
                float(viewport["signal_center_freq_max"]) * 2
            ),
        }
    )
    impairments = Impairments(level=cfg.impairment_level)
    dataset = PrototypeViewportDataset(
        viewport_num_iq_samples=int(viewport["num_iq_samples_dataset"]),
        viewport_sample_rate=float(viewport["sample_rate"]),
        signal_generators="all",
        metadata=canvas,
        transforms=[impairments.dataset_transforms],
        component_transforms=[impairments.signal_transforms],
        allow_empty=True,
    )
    return dataset, viewport, cfg.seed


def draw_signal(axis, signal, fft_size: int, title: str) -> None:
    """Draw a spectrogram and canonical component boxes."""
    axis.imshow(
        compute_spectrogram(signal.data, fft_size, fft_size),
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
    for component in signal.component_signals:
        lower = component.center_freq - component.bandwidth / 2
        axis.add_patch(
            Rectangle(
                (component.start_in_samples, lower / 1e6),
                component.duration_in_samples,
                component.bandwidth / 1e6,
                fill=False,
                edgecolor="white",
            )
        )
        axis.text(
            component.start_in_samples,
            (component.center_freq + component.bandwidth / 2) / 1e6,
            component.class_name,
            color="white",
            fontsize=6,
        )
    axis.set_title(title)
    axis.set_xlabel("IQ sample")
    axis.set_ylabel("Frequency (MHz)")


def main() -> None:
    """Generate integrated viewports and save their visualization."""
    args = parse_args()
    dataset, viewport, seed = build_dataset(args.config)
    loader = WorkerSeedingDataLoader(
        dataset,
        seed=seed,
        batch_size=1,
        num_workers=0,
        collate_fn=identity_collate_fn,
    )
    iterator = iter(loader)
    pairs = []
    for _ in range(args.num_samples):
        output = next(iterator)[0]
        if dataset.last_canvas is None:
            raise RuntimeError("dataset did not retain its generated canvas")
        pairs.append((dataset.last_canvas, output))

    figure, axes = plt.subplots(
        len(pairs),
        2,
        figsize=(15, 4.5 * len(pairs)),
        squeeze=False,
        constrained_layout=True,
    )
    for index, (canvas, output) in enumerate(pairs):
        draw_signal(axes[index, 0], canvas, int(viewport["fft_size"]), "Canvas")
        draw_signal(
            axes[index, 1],
            output,
            int(viewport["fft_size"]),
            "Generator-integrated viewport",
        )
        viewport_box = Rectangle(
            (
                output.viewport_input_time_start,
                (
                    output.viewport_input_center_freq
                    - output.sample_rate / 2
                )
                / 1e6,
            ),
            len(output.data) * output.viewport_decimation,
            output.sample_rate / 1e6,
            fill=False,
            edgecolor="red",
            linewidth=2,
        )
        axes[index, 0].add_patch(viewport_box)

    args.save.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.save, dpi=150)
    print(f"Configuration: {args.config.resolve()}")
    print(f"Visualization: {args.save.resolve()}")
    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
