"""Tests for public clock-jitter model configuration."""

from types import SimpleNamespace

import numpy as np
import pytest

from torchsig.transforms import functional as F  # noqa: N812 - conventional functional API alias
from torchsig.transforms.transforms import ClockJitter


@pytest.mark.parametrize("jitter_model", ["independent", "period", "ar1"])
def test_clock_jitter_functional_propagates_model_configuration(monkeypatch, jitter_model: str) -> None:
    captured = {}

    def fake_sampling_clock_impairments(**kwargs):
        captured.update(kwargs)
        return kwargs["x"]

    monkeypatch.setattr(F, "_sampling_clock_impairments", fake_sampling_clock_impairments)
    data = np.ones(8, dtype=np.complex64)

    output = F.clock_jitter(
        data,
        jitter_ppm=12.0,
        jitter_model=jitter_model,
        jitter_alpha=-0.35,
        rng=np.random.default_rng(3),
    )

    assert captured["jitter_model"] == jitter_model
    assert captured["jitter_alpha"] == -0.35
    np.testing.assert_array_equal(output, data)


def test_clock_jitter_functional_defaults_to_independent(monkeypatch) -> None:
    captured = {}

    def fake_sampling_clock_impairments(**kwargs):
        captured.update(kwargs)
        return kwargs["x"]

    monkeypatch.setattr(F, "_sampling_clock_impairments", fake_sampling_clock_impairments)

    F.clock_jitter(np.ones(8, dtype=np.complex64), rng=np.random.default_rng(3))

    assert captured["jitter_model"] == "independent"
    assert captured["jitter_alpha"] == 0.9


def test_clock_jitter_transform_propagates_model_configuration(monkeypatch) -> None:
    captured = {}

    def fake_clock_jitter(**kwargs):
        captured.update(kwargs)
        return kwargs["data"]

    monkeypatch.setattr(F, "clock_jitter", fake_clock_jitter)
    transform = ClockJitter(
        jitter_ppm=(2.0, 2.0),
        jitter_model="ar1",
        jitter_alpha=0.65,
        seed=7,
    )
    signal = SimpleNamespace(data=np.ones(8, dtype=np.complex64))

    transform.__apply__(signal)

    assert captured["jitter_model"] == "ar1"
    assert captured["jitter_alpha"] == 0.65


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"jitter_model": "unknown"}, "jitter_model"),
        ({"jitter_alpha": 1.0}, "jitter_alpha"),
        ({"jitter_alpha": -1.0}, "jitter_alpha"),
        ({"jitter_alpha": np.nan}, "jitter_alpha"),
    ],
)
def test_clock_jitter_transform_rejects_invalid_model_configuration(kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ClockJitter(**kwargs)
