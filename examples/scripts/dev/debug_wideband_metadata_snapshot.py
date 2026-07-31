"""Log completed wideband metadata once per generated spectrogram.

Unlike event-oriented metadata debugging, this example emits no lookup, set,
or delete records. Each ``snapshot`` record contains the completed sample
metadata and a separate metadata mapping for every component signal. Sample
arrays are represented only by shape and dtype.

Example:
    python examples/scripts/dev/debug_wideband_metadata_snapshot.py
"""

# ruff: noqa: INP001

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from pprint import pformat

from torchsig.datasets.datasets import SafeTorchSigIterableDataset
from torchsig.transforms.impairments import Impairments
from torchsig.transforms.metadata_transforms import YOLOLabel
from torchsig.transforms.transforms import Spectrogram
from torchsig.utils.defaults import TorchSigDefaults
from torchsig.utils.metadata_logging import metadata_logging_context
from torchsig.utils.yaml import load_config_from_yaml

DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[3]
    / "torchsig/datasets/default_configs/wideband_clean_train_all.yaml"
)


class MetadataSnapshotFormatter(logging.Formatter):
    """Format a complete metadata snapshot for interactive inspection."""

    def format(self, record: logging.LogRecord) -> str:
        """Return a readable snapshot or debug-session summary."""
        if getattr(record, "metadata_event", None) == "summary":
            return (
                "metadata snapshot summary: "
                f"emitted={record.metadata_emitted_events} "
                f"suppressed={record.metadata_suppressed_events} "
                f"filtered={record.metadata_filtered_events}"
            )

        context = getattr(record, "metadata_correlation_fields", {})
        snapshot = {
            "session_id": getattr(record, "metadata_session_id", None),
            "dataset_id": getattr(record, "metadata_dataset_id", None),
            "sample_index": getattr(record, "metadata_sample_index", None),
            "worker_id": getattr(record, "metadata_worker_id", None),
            "stage": context.get("stage"),
            "data_shape": getattr(record, "metadata_data_shape", None),
            "data_dtype": getattr(record, "metadata_data_dtype", None),
            "metadata": getattr(record, "metadata_snapshot", None),
            "component_metadata": getattr(
                record,
                "metadata_component_snapshots",
                (),
            ),
        }
        return "completed metadata snapshot:\n" + pformat(
            snapshot,
            sort_dicts=False,
            width=100,
        )


def parse_args() -> argparse.Namespace:
    """Parse options for the snapshot-only wideband debug run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Wideband spectrogram YAML configuration.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of completed metadata snapshots to emit.",
    )
    parser.add_argument(
        "--value-repr-limit",
        type=int,
        default=160,
        help="Maximum representation length for each metadata value.",
    )
    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error("--num-samples must be positive")
    if args.value_repr_limit < 1:
        parser.error("--value-repr-limit must be positive")
    return args


def configure_snapshot_logger() -> None:
    """Configure console output for structured metadata snapshots."""
    handler = logging.StreamHandler()
    handler.setFormatter(MetadataSnapshotFormatter())

    logger = logging.getLogger("torchsig.metadata")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False


def build_wideband_dataset(config_path: Path) -> SafeTorchSigIterableDataset:
    """Build the configured wideband spectrogram generation pipeline."""
    config = load_config_from_yaml(config_path)
    if config.output_representation != "spectrogram":
        raise ValueError("the snapshot example requires a spectrogram config")

    metadata = TorchSigDefaults().default_dataset_metadata.copy()
    metadata.update(config.dataset_metadata)
    metadata["dataset_id"] = config.dataset_id
    impairments = Impairments(level=config.impairment_level)
    return SafeTorchSigIterableDataset(
        signal_generators="all",
        metadata=metadata,
        transforms=[
            impairments.dataset_transforms,
            Spectrogram(fft_size=int(metadata["fft_size"])),
            YOLOLabel(),
        ],
        component_transforms=[impairments.signal_transforms],
        target_labels=None,
        seed=config.seed,
    )


def main() -> None:
    """Generate samples and emit one completed metadata snapshot per sample."""
    args = parse_args()
    configure_snapshot_logger()
    dataset = build_wideband_dataset(args.config)

    with metadata_logging_context(fields={"example": "wideband-snapshot"}):
        dataset.enable_metadata_debug(
            events={"snapshot"},
            max_events=args.num_samples,
            include_values=True,
            value_repr_limit=args.value_repr_limit,
        )
        try:
            for _ in range(args.num_samples):
                next(dataset)
        finally:
            dataset.disable_metadata_debug()


if __name__ == "__main__":
    main()
