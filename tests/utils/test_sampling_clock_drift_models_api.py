"""Tests for public sampling-clock drift model configuration."""

from types import SimpleNamespace

import numpy as np
import pytest

from torchsig.transforms import functional as F  # noqa: N812
from torchsig.transforms.transforms import ClockDrift


@pytest.mark.parametrize("drift_model", ["constant", "linear", "random_walk"])
def test_clock_drift_functional_propagates_model(monkeypatch, drift_model: str) -> None:
    captured = {}

    def fake_sampling_clock_impairments(**kwargs):
        captured.update(kwargs)
        return kwargs["x"]

    monkeypatch.setattr(F, "_sampling_clock_impairments", fake_sampling_clock_impairments)
    data = np.ones(8, dtype=np.complex64)
    F.clock_drift(
        data,
        drift_model=drift_model,
        drift_rate_ppm_per_sec=2.0,
        drift_noise_ppm_per_sqrt_sec=3.0,
        sample_rate_hz=4.0,
    )
    assert captured["drift_model"] == drift_model
    assert captured["drift_rate_ppm_per_sec"] == 2.0
    assert captured["drift_noise_ppm_per_sqrt_sec"] == 3.0
    assert captured["sample_rate_hz"] == 4.0


def test_clock_drift_transform_propagates_model(monkeypatch) -> None:
    captured = {}

    def fake_clock_drift(**kwargs):
        captured.update(kwargs)
        return kwargs["data"]

    monkeypatch.setattr(F, "clock_drift", fake_clock_drift)
    transform = ClockDrift(
        drift_ppm=(2.0, 2.0),
        drift_model="random_walk",
        drift_noise_ppm_per_sqrt_sec=(3.0, 3.0),
        sample_rate_hz=4.0,
        seed=7,
    )
    transform.__apply__(SimpleNamespace(data=np.ones(8, dtype=np.complex64)))
    assert captured["drift_model"] == "random_walk"
    assert captured["drift_noise_ppm_per_sqrt_sec"] == 3.0
    assert captured["sample_rate_hz"] == 4.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"drift_model": "unknown"}, "drift_model"),
        ({"drift_ppm": -1_000_000.0}, "drift_ppm"),
        ({"drift_rate_ppm_per_sec": np.nan}, "drift_rate"),
        ({"drift_noise_ppm_per_sqrt_sec": -1.0}, "drift_noise"),
        ({"sample_rate_hz": 0.0}, "sample_rate_hz"),
    ],
)
def test_clock_drift_transform_rejects_invalid_configuration(kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ClockDrift(**kwargs)
