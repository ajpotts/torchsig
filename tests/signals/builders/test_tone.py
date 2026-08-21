"""Unit tests for the tone signal builder and modulator."""

from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from torchsig.signals.builders.tone import (
    ToneGenerationParameters,
    ToneSignalGenerator,
    tone_modulator,
)
from torchsig.utils.dsp import TorchSigComplexDataType

MODULE_PATH = "torchsig.signals.builders.tone"


@pytest.mark.parametrize("num_samples", [0, -1, -100])
def test_tone_modulator_rejects_nonpositive_num_samples(num_samples):
    """Nonpositive sample counts should raise ValueError."""
    with pytest.raises(
        ValueError,
        match="num_samples must be positive",
    ):
        tone_modulator(num_samples)


@pytest.mark.parametrize("num_samples", [1, 2, 17, 128])
def test_tone_modulator_returns_expected_shape(num_samples):
    """The tone should contain exactly the requested number of samples."""
    result = tone_modulator(num_samples)

    assert result.shape == (num_samples,)


@pytest.mark.parametrize("num_samples", [1, 8, 64])
def test_tone_modulator_returns_all_ones(num_samples):
    """A baseband tone should consist entirely of unit-valued samples."""
    result = tone_modulator(num_samples)

    expected = np.ones(
        num_samples,
        dtype=TorchSigComplexDataType,
    )

    np.testing.assert_array_equal(result, expected)


def test_tone_modulator_returns_torchsig_complex_dtype():
    """The tone should use the configured TorchSig complex dtype."""
    result = tone_modulator(16)

    assert result.dtype == np.dtype(TorchSigComplexDataType)


def test_tone_modulator_returns_complex_values():
    """Tone samples should be represented as complex numbers."""
    result = tone_modulator(8)

    assert np.iscomplexobj(result)


def test_tone_modulator_has_unit_magnitude():
    """Every tone sample should lie on the complex unit circle."""
    result = tone_modulator(32)

    np.testing.assert_array_equal(
        np.abs(result),
        np.ones(32),
    )


def test_tone_modulator_is_deterministic():
    """Identical calls should produce identical tone samples."""
    first = tone_modulator(32)
    second = tone_modulator(32)

    np.testing.assert_array_equal(first, second)


def test_tone_modulator_returns_independent_arrays():
    """Each call should return a newly allocated array."""
    first = tone_modulator(8)
    second = tone_modulator(8)

    first[0] = 5 + 2j

    assert second[0] == 1 + 0j


def test_tone_signal_generator_initialization():
    """The generator should configure required fields and class name."""
    metadata = {}

    with (
        patch(
            f"{MODULE_PATH}.BaseSignalGenerator.__init__",
            autospec=True,
        ) as base_init,
        patch.object(
            ToneSignalGenerator,
            "set_default_class_name",
        ) as set_class_name,
    ):
        generator = ToneSignalGenerator(**metadata)

    base_init.assert_called_once_with(generator)
    set_class_name.assert_called_once_with("tone")

    assert generator.required_metadata_fields == [
        "signal_duration_in_samples_min",
        "signal_duration_in_samples_max",
    ]


def test_tone_signal_generator_initialization_forwards_metadata():
    """Initialization metadata should be forwarded to the base generator."""
    metadata = {
        "signal_duration_in_samples_min": 100,
        "signal_duration_in_samples_max": 200,
    }

    with (
        patch(
            f"{MODULE_PATH}.BaseSignalGenerator.__init__",
            autospec=True,
        ) as base_init,
        patch.object(
            ToneSignalGenerator,
            "set_default_class_name",
        ),
    ):
        generator = ToneSignalGenerator(**metadata)

    base_init.assert_called_once_with(
        generator,
        **metadata,
    )


def test_tone_signal_generator_generate():
    """The generator should sample a duration and construct a tone Signal."""
    metadata = {
        "signal_duration_in_samples_min": 100,
        "signal_duration_in_samples_max": 200,
    }

    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = 150

    class GeneratorStub:
        random_generator = rng

        def __getitem__(self, key):
            return metadata[key]

    signal_data = np.ones(
        150,
        dtype=TorchSigComplexDataType,
    )
    expected_signal = MagicMock()

    with (
        patch(
            f"{MODULE_PATH}.tone_modulator",
            return_value=signal_data,
        ) as modulator,
        patch(
            f"{MODULE_PATH}.Signal",
            return_value=expected_signal,
        ) as signal_class,
    ):
        result = ToneSignalGenerator.generate(GeneratorStub())

    rng.integers.assert_called_once_with(
        low=100,
        high=201,
    )

    modulator.assert_called_once_with(150)

    signal_class.assert_called_once_with(
        data=signal_data,
        center_freq=0,
        bandwidth=1,
    )

    assert result is expected_signal


def test_tone_signal_generator_generate_allows_minimum_duration():
    """The lower duration bound should be a valid sampled value."""
    metadata = {
        "signal_duration_in_samples_min": 100,
        "signal_duration_in_samples_max": 200,
    }

    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = 100

    class GeneratorStub:
        random_generator = rng

        def __getitem__(self, key):
            return metadata[key]

    signal_data = np.ones(
        100,
        dtype=TorchSigComplexDataType,
    )

    with (
        patch(
            f"{MODULE_PATH}.tone_modulator",
            return_value=signal_data,
        ) as modulator,
        patch(
            f"{MODULE_PATH}.Signal",
        ),
    ):
        ToneSignalGenerator.generate(GeneratorStub())

    modulator.assert_called_once_with(100)


def test_tone_signal_generator_generate_allows_maximum_duration():
    """The inclusive upper duration bound should be a valid sampled value."""
    metadata = {
        "signal_duration_in_samples_min": 100,
        "signal_duration_in_samples_max": 200,
    }

    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = 200

    class GeneratorStub:
        random_generator = rng

        def __getitem__(self, key):
            return metadata[key]

    signal_data = np.ones(
        200,
        dtype=TorchSigComplexDataType,
    )

    with (
        patch(
            f"{MODULE_PATH}.tone_modulator",
            return_value=signal_data,
        ) as modulator,
        patch(
            f"{MODULE_PATH}.Signal",
        ),
    ):
        ToneSignalGenerator.generate(GeneratorStub())

    assert rng.integers.call_args_list == [
        call(low=100, high=201),
    ]
    modulator.assert_called_once_with(200)


def test_tone_segment_matches_slice_of_complete_burst():
    """Segment generation should exactly reproduce the corresponding full IQ."""
    generator = ToneSignalGenerator(
        signal_duration_in_samples_min=1_000,
        signal_duration_in_samples_max=1_000,
    )
    parameters = ToneGenerationParameters(num_samples=1_000)

    complete = generator.generate_from_parameters(parameters)
    segment = generator.generate_segment(
        parameters,
        start_sample=321,
        num_samples=127,
    )

    np.testing.assert_array_equal(segment.data, complete.data[321:448])
    assert segment.segment_start_in_samples == 321
    assert segment.original_duration_in_samples == 1_000


def test_tone_parameter_sampling_does_not_generate_iq(monkeypatch):
    """Metadata-first sampling should not allocate waveform data."""
    generator = ToneSignalGenerator(
        signal_duration_in_samples_min=100,
        signal_duration_in_samples_max=200,
        seed=42,
    )

    def fail_modulator(_num_samples):
        raise AssertionError("parameter sampling must not generate IQ")

    monkeypatch.setattr(f"{MODULE_PATH}.tone_modulator", fail_modulator)

    parameters = generator.sample_parameters()

    assert 100 <= parameters.num_samples <= 200


@pytest.mark.parametrize(
    ("start_sample", "num_samples", "message"),
    [
        (-1, 10, "nonnegative"),
        (0, 0, "positive"),
        (90, 11, "exceeds"),
    ],
)
def test_tone_segment_rejects_invalid_interval(
    start_sample: int,
    num_samples: int,
    message: str,
):
    """Segment bounds must describe a nonempty subset of the full burst."""
    generator = ToneSignalGenerator(
        signal_duration_in_samples_min=100,
        signal_duration_in_samples_max=100,
    )

    with pytest.raises(ValueError, match=message):
        generator.generate_segment(
            ToneGenerationParameters(num_samples=100),
            start_sample=start_sample,
            num_samples=num_samples,
        )
