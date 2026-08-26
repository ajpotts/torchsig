"""Train a small multi-task model on the detailed modulation labels.

The example deliberately uses a clean, constrained dataset so it demonstrates
that the labels added by the signal generators are learnable without turning
into a large model-training exercise. Run it from the repository root:

    python examples/scripts/train_modrec_metadata_targets.py

Use ``--train-samples 64 --validation-samples 32 --epochs 1`` for a smoke run.
"""

# ruff: noqa: INP001

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from torchsig.signals.builders.constellation import ConstellationSignalGenerator
from torchsig.signals.builders.ofdm import OFDMSignalGenerator

NUM_IQ_SAMPLES = 2048
NUM_SUBCARRIERS = 32
MAX_CYCLIC_PREFIX_LEN = NUM_SUBCARRIERS // 2 - 1
NUM_FEATURE_BINS = 256
MIN_DATASET_SAMPLES = 2


@dataclass(frozen=True)
class DatasetTensors:
    """Feature and target tensors used by the training example."""

    features: torch.Tensor
    waveform_type: torch.Tensor
    has_cyclic_prefix: torch.Tensor
    cyclic_prefix_len: torch.Tensor
    pulse_shape: torch.Tensor
    alpha_rolloff: torch.Tensor

    def as_dataset(self) -> TensorDataset:
        """Return tensors in the order expected by the training loop."""
        return TensorDataset(
            self.features,
            self.waveform_type,
            self.has_cyclic_prefix,
            self.cyclic_prefix_len,
            self.pulse_shape,
            self.alpha_rolloff,
        )


class MetadataTargetModel(nn.Module):
    """Small shared MLP with one output head for each metadata target."""

    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(NUM_FEATURE_BINS * 2, 192),
            nn.ReLU(),
            nn.LayerNorm(192),
            nn.Linear(192, 96),
            nn.ReLU(),
        )
        self.waveform_type = nn.Linear(96, 1)
        self.has_cyclic_prefix = nn.Linear(96, 1)
        self.cyclic_prefix_len = nn.Linear(96, 1)
        self.pulse_shape = nn.Linear(96, 1)
        self.alpha_rolloff = nn.Linear(96, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict all five targets from one feature batch."""
        hidden = self.shared(features)
        return {
            "waveform_type": self.waveform_type(hidden).squeeze(1),
            "has_cyclic_prefix": self.has_cyclic_prefix(hidden).squeeze(1),
            "cyclic_prefix_len": self.cyclic_prefix_len(hidden).squeeze(1),
            "pulse_shape": self.pulse_shape(hidden).squeeze(1),
            "alpha_rolloff": self.alpha_rolloff(hidden).squeeze(1),
        }


def generator_metadata() -> dict[str, int]:
    """Return fixed signal parameters that keep the learning problem small."""
    return {
        "sample_rate": 8192,
        "bandwidth_min": 2048,
        "bandwidth_max": 2048,
        "signal_duration_in_samples_min": NUM_IQ_SAMPLES,
        "signal_duration_in_samples_max": NUM_IQ_SAMPLES,
    }


def extract_features(iq: np.ndarray) -> np.ndarray:
    """Return normalized spectral and autocorrelation features."""
    iq = np.asarray(iq, dtype=np.complex64)
    rms = np.sqrt(np.mean(np.abs(iq) ** 2))
    iq = iq / max(float(rms), 1e-8)

    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft(iq))))
    spectrum = spectrum.reshape(NUM_FEATURE_BINS, -1).mean(axis=1)
    spectrum = (spectrum - spectrum.mean()) / max(float(spectrum.std()), 1e-8)

    fft_length = 2 * len(iq)
    iq_fft = np.fft.fft(iq, n=fft_length)
    autocorrelation = np.abs(np.fft.ifft(iq_fft * np.conj(iq_fft)))
    autocorrelation = autocorrelation[:NUM_FEATURE_BINS]
    autocorrelation /= max(float(autocorrelation[0]), 1e-8)

    return np.concatenate((spectrum, autocorrelation)).astype(np.float32)


def build_dataset(num_samples: int, seed: int) -> DatasetTensors:
    """Generate examples and read targets from the new Signal metadata."""
    if num_samples < MIN_DATASET_SAMPLES:
        raise ValueError(f"num_samples must be at least {MIN_DATASET_SAMPLES}")

    metadata = generator_metadata()
    ofdm_generator = OFDMSignalGenerator(
        **metadata,
        num_subcarriers=NUM_SUBCARRIERS,
        seed=seed,
    )
    constellation_generator = ConstellationSignalGenerator(
        **metadata,
        constellation_name="qpsk",
        seed=seed + 1,
    )

    features: list[np.ndarray] = []
    waveform_types: list[float] = []
    has_cyclic_prefixes: list[float] = []
    cyclic_prefix_lengths: list[float] = []
    pulse_shapes: list[float] = []
    alpha_rolloffs: list[float] = []

    for sample_index in range(num_samples):
        is_ofdm = sample_index % 2 == 0
        signal = ofdm_generator.generate() if is_ofdm else constellation_generator.generate()
        features.append(extract_features(signal.data))
        waveform_types.append(float(is_ofdm))

        if is_ofdm:
            has_cyclic_prefixes.append(float(signal.has_cyclic_prefix))
            cyclic_prefix_lengths.append(float(signal.cyclic_prefix_len) / MAX_CYCLIC_PREFIX_LEN)
            pulse_shapes.append(0.0)
            alpha_rolloffs.append(0.0)
        else:
            is_srrc = signal.pulse_shape_name == "srrc"
            has_cyclic_prefixes.append(0.0)
            cyclic_prefix_lengths.append(0.0)
            pulse_shapes.append(float(is_srrc))
            alpha_rolloffs.append(float(signal.alpha_rolloff) if is_srrc else 0.0)

    def tensor(values: list[float] | list[np.ndarray]) -> torch.Tensor:
        return torch.from_numpy(np.asarray(values, dtype=np.float32))

    return DatasetTensors(
        features=tensor(features),
        waveform_type=tensor(waveform_types),
        has_cyclic_prefix=tensor(has_cyclic_prefixes),
        cyclic_prefix_len=tensor(cyclic_prefix_lengths),
        pulse_shape=tensor(pulse_shapes),
        alpha_rolloff=tensor(alpha_rolloffs),
    )


def multitask_loss(
    predictions: dict[str, torch.Tensor],
    targets: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Compute losses only where a target applies to the waveform."""
    waveform_type, has_cp, cp_len, pulse_shape, alpha = targets
    ofdm_mask = waveform_type.bool()
    constellation_mask = ~ofdm_mask
    srrc_mask = constellation_mask & pulse_shape.bool()

    loss = nn.functional.binary_cross_entropy_with_logits(predictions["waveform_type"], waveform_type)
    loss += nn.functional.binary_cross_entropy_with_logits(predictions["has_cyclic_prefix"][ofdm_mask], has_cp[ofdm_mask])
    loss += nn.functional.mse_loss(predictions["cyclic_prefix_len"][ofdm_mask], cp_len[ofdm_mask])
    loss += nn.functional.binary_cross_entropy_with_logits(
        predictions["pulse_shape"][constellation_mask],
        pulse_shape[constellation_mask],
    )
    if srrc_mask.any():
        loss += nn.functional.mse_loss(predictions["alpha_rolloff"][srrc_mask], alpha[srrc_mask])
    return loss


def train_model(
    dataset: TensorDataset,
    epochs: int,
    batch_size: int,
) -> MetadataTargetModel:
    """Train and return the multi-task model."""
    torch.manual_seed(0)
    model = MetadataTargetModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for features, *targets in loader:
            predictions = model(features)
            loss = multitask_loss(predictions, tuple(targets))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().item() * len(features)
        print(f"epoch={epoch + 1:02d} loss={total_loss / len(dataset):.4f}")

    return model


def evaluate_model(model: MetadataTargetModel, dataset: TensorDataset) -> None:
    """Print validation metrics for each applicable target."""
    features, waveform_type, has_cp, cp_len, pulse_shape, alpha = dataset.tensors
    model.eval()
    with torch.no_grad():
        predictions = model(features)

    ofdm_mask = waveform_type.bool()
    constellation_mask = ~ofdm_mask
    srrc_mask = constellation_mask & pulse_shape.bool()
    type_prediction = (predictions["waveform_type"] >= 0).float()
    cp_prediction = (predictions["has_cyclic_prefix"] >= 0).float()
    pulse_prediction = (predictions["pulse_shape"] >= 0).float()
    cp_len_prediction = (predictions["cyclic_prefix_len"].clamp(0, 1) * MAX_CYCLIC_PREFIX_LEN).round()
    cp_len_target = (cp_len * MAX_CYCLIC_PREFIX_LEN).round()

    print(f"waveform type accuracy: {(type_prediction == waveform_type).float().mean():.1%}")
    print(f"cyclic prefix accuracy: {(cp_prediction[ofdm_mask] == has_cp[ofdm_mask]).float().mean():.1%}")
    print(f"cyclic prefix length MAE: {(cp_len_prediction[ofdm_mask] - cp_len_target[ofdm_mask]).abs().float().mean():.2f} samples")
    print(f"pulse shape accuracy: {(pulse_prediction[constellation_mask] == pulse_shape[constellation_mask]).float().mean():.1%}")
    if srrc_mask.any():
        print(f"alpha roll-off MAE: {(predictions['alpha_rolloff'][srrc_mask] - alpha[srrc_mask]).abs().mean():.3f}")


def parse_args() -> argparse.Namespace:
    """Parse command-line settings for dataset size and training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-samples", type=int, default=768)
    parser.add_argument("--validation-samples", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    """Generate labeled signals, train the model, and report its metrics."""
    args = parse_args()
    print("Generating training signals and reading targets from Signal metadata...")
    training = build_dataset(args.train_samples, seed=123)
    validation = build_dataset(args.validation_samples, seed=456)
    model = train_model(training.as_dataset(), args.epochs, args.batch_size)
    evaluate_model(model, validation.as_dataset())


if __name__ == "__main__":
    main()
