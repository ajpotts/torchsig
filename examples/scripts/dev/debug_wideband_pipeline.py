"""Debug metadata access while generating wideband spectrogram samples.

This example uses TorchSIG's default clean wideband training configuration and
prints structured metadata records correlated by session, sample, worker, and
pipeline stage. It is intended for diagnosing metadata inheritance and access;
it is not a general-purpose performance profiler.

Example:
    python examples/scripts/dev/debug_wideband_pipeline.py --num-samples 2
"""

# ruff: noqa: INP001

from __future__ import annotations

import argparse
import logging
from pathlib import Path

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
DEBUG_KEYS = {
    "bandwidth_max",
    "bandwidth_min",
    "cochannel_overlap_probability",
    "fft_size",
    "fft_stride",
    "frequency_max",
    "frequency_min",
    "num_iq_samples_dataset",
    "num_signals_max",
    "num_signals_min",
    "sample_rate",
    "signal_center_freq_max",
    "signal_center_freq_min",
}


class MetadataDebugFormatter(logging.Formatter):
    """Format TorchSIG metadata records with their correlation fields."""

    def format(self, record: logging.LogRecord) -> str:
        """Return a compact, human-readable representation of one record."""
        fields = getattr(record, "metadata_correlation_fields", {})
        value = getattr(record, "metadata_value", "<values disabled>")
        return (
            f"session={getattr(record, 'metadata_session_id', None)} "
            f"sample={getattr(record, 'metadata_sample_index', None)} "
            f"worker={getattr(record, 'metadata_worker_id', None)} "
            f"stage={fields.get('stage')} "
            f"event={getattr(record, 'metadata_event', None)} "
            f"key={getattr(record, 'metadata_key', None)} "
            f"source={getattr(record, 'metadata_source', None)} "
            f"depth={getattr(record, 'metadata_depth', None)} "
            f"value={value}"
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the wideband debug run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Wideband dataset YAML configuration.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2,
        help="Number of spectrogram samples to generate.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=100,
        help="Maximum number of metadata records to emit.",
    )
    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error("--num-samples must be positive")
    if args.max_events < 0:
        parser.error("--max-events cannot be negative")
    return args


def configure_metadata_logger() -> None:
    """Send structured TorchSIG metadata records to the console."""
    handler = logging.StreamHandler()
    handler.setFormatter(MetadataDebugFormatter())

    logger = logging.getLogger("torchsig.metadata")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False


def build_wideband_dataset(config_path: Path) -> SafeTorchSigIterableDataset:
    """Build a spectrogram dataset from a TorchSIG wideband YAML config."""
    config = load_config_from_yaml(config_path)
    if config.output_representation != "spectrogram":
        raise ValueError("the debug example requires a spectrogram configuration")

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
    """Generate wideband samples while reporting metadata resolution."""
    args = parse_args()
    configure_metadata_logger()
    dataset = build_wideband_dataset(args.config)

    with metadata_logging_context(fields={"example": "wideband-debug"}) as context:
        print(f"Debug session: {context.session_id}", flush=True)
        dataset.enable_metadata_debug(
            keys=DEBUG_KEYS,
            events={"lookup"},
            max_events=args.max_events,
            include_values=True,
            value_repr_limit=80,
        )
        try:
            for sample_number in range(args.num_samples):
                sample = next(dataset)
                print(
                    f"sample {sample_number}: shape={sample.data.shape}, "
                    f"dtype={sample.data.dtype}, "
                    f"components={len(sample.component_signals)}"
                )
        finally:
            dataset.disable_metadata_debug()

    statistics = dataset.metadata_debug_statistics
    print(
        "Metadata records: "
        f"emitted={statistics.emitted_events}, "
        f"suppressed={statistics.suppressed_events}, "
        f"filtered={statistics.filtered_events}"
    )


if __name__ == "__main__":
    main()
