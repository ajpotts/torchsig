"""Visualize the experimental IQ time-frequency scene-crop transform.

Example:
    python examples/scripts/dev/prototype_scene_crop.py --no-show
"""

# ruff: noqa: INP001

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from torchsig.signals.signal_types import Signal
from torchsig.transforms.prototype_scene_crop import PrototypeSceneCrop
from torchsig.utils.dsp import compute_spectrogram

CANVAS_SAMPLE_RATE = 16_000_000.0
CANVAS_NUM_SAMPLES = 131_072
OUTPUT_SAMPLE_RATE = 8_000_000.0
OUTPUT_NUM_SAMPLES = 32_768
VIEWPORT_TIME_START = 32_768
VIEWPORT_CENTER_FREQ = 3_000_000.0
FFT_SIZE = 512


def parse_args() -> argparse.Namespace:
    """Parse visualization options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save",
        type=Path,
        default=Path("prototype_scene_crop.png"),
        help="Output image path.",
    )
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def tone(frequency: float, num_samples: int) -> np.ndarray:
    """Return a complex64 tone at a canvas-relative frequency."""
    samples = np.arange(num_samples)
    return np.exp(
        2j * np.pi * frequency * samples / CANVAS_SAMPLE_RATE
    ).astype(np.complex64)


def component(
    class_name: str,
    *,
    start: int,
    duration: int,
    center_freq: float,
    bandwidth: float,
) -> Signal:
    """Construct one deterministic component for the example canvas."""
    return Signal(
        data=tone(center_freq, duration),
        class_name=class_name,
        start_in_samples=start,
        duration_in_samples=duration,
        center_freq=center_freq,
        bandwidth=bandwidth,
    )


def build_canvas() -> Signal:
    """Build a scene containing full, partial, and excluded components."""
    components = [
        component(
            "fully-visible",
            start=45_000,
            duration=20_000,
            center_freq=2_000_000,
            bandwidth=1_000_000,
        ),
        component(
            "time-truncated",
            start=20_000,
            duration=25_000,
            center_freq=4_000_000,
            bandwidth=1_000_000,
        ),
        component(
            "frequency-truncated",
            start=60_000,
            duration=22_000,
            center_freq=7_000_000,
            bandwidth=4_000_000,
        ),
        component(
            "time-frequency-truncated",
            start=85_000,
            duration=25_000,
            center_freq=-2_000_000,
            bandwidth=4_000_000,
        ),
        component(
            "excluded",
            start=100_000,
            duration=15_000,
            center_freq=-6_000_000,
            bandwidth=1_000_000,
        ),
    ]
    data = np.zeros(CANVAS_NUM_SAMPLES, dtype=np.complex64)
    for index, signal in enumerate(components):
        start = int(signal.start_in_samples)
        stop = min(len(data), start + len(signal.data))
        data[start:stop] += signal.data[: stop - start] * (1 - index * 0.12)

    return Signal(
        data=data,
        component_signals=components,
        sample_rate=CANVAS_SAMPLE_RATE,
        num_iq_samples_dataset=CANVAS_NUM_SAMPLES,
        frequency_min=-CANVAS_SAMPLE_RATE / 2,
        frequency_max=CANVAS_SAMPLE_RATE / 2,
        center_freq=0,
        bandwidth=CANVAS_SAMPLE_RATE,
    )


def spectrogram(signal: Signal) -> np.ndarray:
    """Compute a display spectrogram for one IQ signal."""
    return compute_spectrogram(signal.data, FFT_SIZE, FFT_SIZE)


def draw_components(axis, signal: Signal, *, color: str = "white") -> None:
    """Draw component time-frequency boxes on a spectrogram axis."""
    for component_signal in signal.component_signals:
        lower_freq = component_signal.center_freq - component_signal.bandwidth / 2
        upper_freq = component_signal.center_freq + component_signal.bandwidth / 2
        box = Rectangle(
            (
                component_signal.start_in_samples,
                lower_freq / 1e6,
            ),
            component_signal.duration_in_samples,
            component_signal.bandwidth / 1e6,
            fill=False,
            edgecolor=color,
            linewidth=1.5,
        )
        axis.add_patch(box)
        axis.text(
            component_signal.start_in_samples,
            upper_freq / 1e6,
            component_signal.class_name,
            color=color,
            fontsize=7,
            va="bottom",
        )


def show_spectrogram(axis, signal: Signal, title: str) -> None:
    """Display one spectrogram in physical time-frequency coordinates."""
    axis.imshow(
        spectrogram(signal),
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
    axis.set_title(title)
    axis.set_xlabel("Sample")
    axis.set_ylabel("Frequency (MHz)")


def main() -> None:
    """Create a canvas, extract a viewport, and visualize both."""
    args = parse_args()
    canvas = build_canvas()
    crop = PrototypeSceneCrop(
        num_iq_samples=OUTPUT_NUM_SAMPLES,
        sample_rate=OUTPUT_SAMPLE_RATE,
        time_start=VIEWPORT_TIME_START,
        center_freq=VIEWPORT_CENTER_FREQ,
        allow_empty=False,
    )
    output = crop(canvas)

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(15, 9),
        constrained_layout=True,
    )
    show_spectrogram(axes[0, 0], canvas, "Expanded time-frequency canvas")
    draw_components(axes[0, 0], canvas)

    show_spectrogram(axes[0, 1], canvas, "Selected receiver viewport")
    viewport_input_samples = OUTPUT_NUM_SAMPLES * int(
        CANVAS_SAMPLE_RATE / OUTPUT_SAMPLE_RATE
    )
    viewport = Rectangle(
        (
            VIEWPORT_TIME_START,
            (VIEWPORT_CENTER_FREQ - OUTPUT_SAMPLE_RATE / 2) / 1e6,
        ),
        viewport_input_samples,
        OUTPUT_SAMPLE_RATE / 1e6,
        fill=False,
        edgecolor="red",
        linewidth=3,
    )
    axes[0, 1].add_patch(viewport)
    draw_components(axes[0, 1], canvas)

    show_spectrogram(axes[1, 0], output, "Cropped IQ in viewport coordinates")
    show_spectrogram(axes[1, 1], output, "Visible component metadata")
    draw_components(axes[1, 1], output)

    args.save.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.save, dpi=150)
    print(f"Canvas components: {len(canvas.component_signals)}")
    print(f"Visible components: {len(output.component_signals)}")
    print(f"Output shape/dtype: {output.data.shape} / {output.data.dtype}")
    print(f"Visualization: {args.save.resolve()}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
