"""Dataset Base Classes for creation and static loading."""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from torch.utils.data import Dataset, IterableDataset

from torchsig.datasets.dataset_utils import frequency_shift_signal
from torchsig.signals.builder import (
    BaseSignalGenerator,
    ConcatSignalGenerator,
)
from torchsig.signals.signal_types import Signal
from torchsig.utils.abstractions import HierarchicalMetadataObject
from torchsig.utils.coordinate_system import (
    Coordinate,
    Rectangle,
    is_rectangle_overlap,
)
from torchsig.utils.dsp import (
    compute_spectrogram,
    update_signal_snr_bandwidth,
)
from torchsig.utils.file_handlers.hdf5 import HDF5Reader
from torchsig.utils.random import Seedable
from torchsig.utils.signal_building import lookup_signal_generator_by_string

from .pipeline_failover import PipelineFailOverEnabled

if TYPE_CHECKING:
    from torchsig.transforms.base_transforms import Transform

log = logging.getLogger(__name__)

__all__ = [
    "StaticTorchSigDataset",
    "TorchSigDatasetConfig",
    "TorchSigIterableDataset",
    "apply_label_to_signal",
    "apply_transforms_and_labels_to_signal",
]

@dataclass(frozen=True)
class TorchSigDatasetConfig:
    """Configuration dataclass for TorchSig datasets.

    Attributes:
        dataset_id: A unique identifier for the dataset.
        dataset_length: The total number of samples in the dataset.
        seed: A random seed for reproducibility.
        impairment_level: The level of impairment to apply to the signals.
        output_representation: The representation of the output data (e.g., "iq" or "spectrogram").
        output_spectrogram_fft: The FFT size to use when generating spectrograms (if output_representation is "spectrogram").
        signal_sampling_mode: The mode for sampling signals, either "per_signal" or "per_family".
        dataset_metadata: A dictionary containing additional metadata about the dataset.
    """
    dataset_id: str
    dataset_length: int
    seed: int
    impairment_level: int
    output_representation: Literal["iq", "spectrogram"]
    output_spectrogram_fft: int | None
    signal_sampling_mode: Literal["per_signal", "per_family"]
    dataset_metadata: dict[str, Any]


def apply_label_to_signal(sample: Signal, target_label: str) -> list:
    """Extract a target label from a signal and its component signals.

    Target labels are resolved through the public ``Signal`` interface rather
    than directly from the underlying metadata dictionary. This allows both
    stored metadata fields (e.g. ``class_name``) and computed properties
    (e.g. ``start``, ``stop``, ``lower_freq``, and ``upper_freq``) to be
    requested uniformly.

    If the signal contains component signals, labels are extracted only from
    the components. Otherwise, the label is extracted from the signal itself.
    This avoids returning duplicate labels for both a composite signal and its
    children.

    Args:
        sample: Signal from which to extract target labels.
        target_label: Name of the target label or ``Signal`` property to
            extract.

    Returns:
        A list containing one value for each component signal, or a single
        value for the signal itself if it has no components.
    """
    values = []

    signals = sample.component_signals or [sample]

    for signal in signals:
        if target_label == "class_index":
            if hasattr(signal, "class_index"):
                values.append(int(signal.class_index))
            elif hasattr(signal, "class_name"):
                class_names = signal.get_full_metadata()["class_names"]
                values.append(int(list(class_names).index(signal.class_name)))
        elif hasattr(signal, target_label):
            values.append(getattr(signal, target_label))

    return values


def apply_transforms_and_labels_to_signal(
    sample: Signal, transforms: list[Transform | callable], target_labels: list
) -> Signal | np.ndarray | tuple:
    """Applies a series of transformations to a signal sample and retrieves specified label values.

    Args:
        sample: The signal sample to process.
        transforms: A list of function objects, each taking a Signal object and returning a transformed Signal object.
        target_labels: Labels to be retrieved from the signal sample after transformations. If None, the transformed signal is returned. If an empty list, the signal data is returned.

    Returns:
        - If target_labels is None, a Signal object with all applied transforms.
        - If target_labels is an empty list, the numpy.ndarray data of the sample.
        - If target_labels contains one label, a tuple of (sample_data, target_value).
        - If target_labels contains multiple labels, a tuple of (sample_data, [target_values]).
    """
    # apply user transforms
    for transform in transforms:
        sample = transform(sample)

    # apply metadata transforms
    # just return data if target_labels is None or empty list
    if target_labels is None:
        # return Signal object
        return sample
    if len(target_labels) < 1:
        # just return np.ndarray data
        return sample.data

    targets = {}
    for key in target_labels:
        values = apply_label_to_signal(sample, key)
        if sample["num_signals_max"] == 1 and len(values) == 1:
            values = values[0]
        targets[key] = values
    if len(target_labels) == 1:
        return sample.data, targets[target_labels[0]]

    return sample.data, [targets[key] for key in targets]


class TorchSigIterableDataset(HierarchicalMetadataObject, IterableDataset):
    """Base class for generating signals.

    The dataset will continue to generate samples infinitely.

    Attributes:
        signal_generators: The signal generators to use. Can be a string, ConcatSignalGenerator, or list.
        transforms: List of transforms to apply to the entire signal.
        component_transforms: List of transforms to apply to individual signal components.
        target_labels: Labels to extract from the signal.
        validate_init: Whether to validate metadata during initialization.
    """

    # pylint: disable=abstract-method

    def __init__(
        self,
        signal_generators: str | ConcatSignalGenerator | list = "all",
        transforms: list[Transform | callable] = None,
        component_transforms: list[Transform | callable] = None,
        target_labels: list | None = None,
        # will try to validate required metadata in this dataset; can be turned off if a dataset needs to be initialized before it's metadata is known
        validate_init: bool = True,
        **kwargs,
    ):
        """Initializes the dataset.

        Args:
            signal_generators: The signal generators to use. Can be a string, ConcatSignalGenerator, or list.
            transforms: List of transforms to apply to the entire signal.
            component_transforms: List of transforms to apply to individual signal components.
            target_labels: Labels to extract from the signal.
            validate_init: Whether to validate metadata during initialization.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        HierarchicalMetadataObject.__init__(self, **kwargs)
        self.validate_init = validate_init
        self.signal_generators = []
        self.signal_likelihoods = []
        self.signal_probabilities = np.array([], dtype=float)
        self._signal_probability_mode = "likelihood"
        self.target_labels = target_labels
        self.transforms = [] if transforms is None else transforms
        self.component_transforms = [] if component_transforms is None else component_transforms
        if not hasattr(self, "class_names"):
            self["class_names"] = []
        if "num_signals_min" not in self.keys():
            self["num_signals_min"] = 1
        if "num_signals_max" not in self.keys():
            self["num_signals_max"] = 1
        for transform in self.transforms:
            if isinstance(transform, Seedable):
                transform.add_parent(self)
        for transform in self.component_transforms:
            if isinstance(transform, Seedable):
                transform.add_parent(self)
        if isinstance(signal_generators, str):
            signal_generators = lookup_signal_generator_by_string(signal_generators)
        if isinstance(signal_generators, ConcatSignalGenerator):
            signal_generators = signal_generators.signal_generators
        for generator in signal_generators:
            self.init_signal_generator(generator)
        
        self.validate()


    @staticmethod
    def _validate_positive_weight(value: float, parameter_name: str) -> float:
        """Validate a likelihood/probability value used for class selection."""
        if not isinstance(value, (int, float, np.integer, np.floating)):
            raise TypeError(
                f"{parameter_name} must be a real number, got {type(value).__name__}"
            )
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"{parameter_name} must be finite")
        if value <= 0.0:
            raise ValueError(f"{parameter_name} must be > 0")

        return value


    def _validate_signal_sampling_configuration(self, require_complete: bool = True) -> None:
        """Validate the dataset's configured class-selection distribution."""
        if len(self.signal_generators) == 0:
            return

        if self._signal_probability_mode == "probability":
            probabilities = np.asarray(self.signal_probabilities, dtype=float)
            if probabilities.shape[0] != len(self.signal_generators):
                raise ValueError(
                    "signal probability count does not match number of generators"
                )
            if np.any(probabilities <= 0.0):
                raise ValueError("all signal probabilities must be > 0")

            probability_sum = float(np.sum(probabilities))
            if probability_sum > 1.0 + 1e-8:
                raise ValueError(
                    f"signal probabilities must sum to 1.0, found {probability_sum}"
                )
            if require_complete and not np.isclose(
                probability_sum, 1.0, atol=1e-8
            ):
                raise ValueError(
                    "signal probabilities must sum to 1.0 before sampling, "
                    f"found {probability_sum}"
                )
            return

        likelihoods = np.asarray(self.signal_likelihoods, dtype=float)
        if likelihoods.shape[0] != len(self.signal_generators):
            raise ValueError(
                "signal likelihood count does not match number of generators"
            )
        if np.any(likelihoods <= 0.0):
            raise ValueError("all signal likelihoods must be > 0")


    def _refresh_signal_probabilities(self) -> None:
        """Recompute normalized sampling probabilities from configured weights."""
        if len(self.signal_generators) == 0:
            self.signal_probabilities = np.array([], dtype=float)
            return

        self._validate_signal_sampling_configuration(require_complete=False)

        if self._signal_probability_mode == "probability":
            self.signal_probabilities = np.asarray(
                self.signal_probabilities,
                dtype=float,
            )
            return

        likelihoods = np.asarray(self.signal_likelihoods, dtype=float)
        self.signal_probabilities = likelihoods / np.sum(likelihoods)

    def validate(self) -> None:
        """Validate the dataset configuration."""

        self.validate_metadata_fields()
        self._validate_signal_sampling_configuration()

        if self["num_signals_min"] > self["num_signals_max"]:
            raise ValueError(...)

        if (
            self["signal_duration_in_samples_max"]
            > self["num_iq_samples_dataset"]
        ):
            warnings.warn(
                "signal_duration_in_samples_max exceeds "
                "num_iq_samples_dataset. Signals may be truncated.",
                UserWarning,
                stacklevel=2,
            )

        if self["fft_size"] > self["num_iq_samples_dataset"]:
            raise ValueError(...)


    def init_signal_generator(self, signal_generator: str | callable) -> None:
        """Initializes the signal generator.

        Args:
            signal_generator: The signal generator to be initialized. If a string, it is first looked up to retrieve the corresponding signal generator function.

        Raises:
            TypeError: If the signal_generator is neither a string nor a callable.
        """
        if isinstance(signal_generator, str):
            self.add_signal_generator(
                lookup_signal_generator_by_string(signal_generator)
            )
        else:
            self.add_signal_generator(signal_generator)

    def add_signal_generator(
        self,
        signal_generator: callable,
        class_name: str | None = None,
        class_index: int | None = None,
        likelihood: float | None = None,
        probability: float | None = None,
    ) -> None:
        """Adds a signal generator to this dataset.

        Args:
            signal_generator: A callable object which takes no arguments and returns a Signal.
            class_name: (optional) A name for this signal class in the dataset. 
                If None, the signal will be generated and added to the data, 
                but no labels will be made for the signal.
            likelihood: (optional) Relative sampling weight for this signal class. 
                If no explicit probabilities are provided anywhere, omitted
                likelihoods default to 1.0 which yields uniform class sampling.
            probability: (optional) Explicit probability for this signal class.
                When any generator is added with ``probability=``, every
                generator added to the dataset must also use explicit
                probabilities, and the final probabilities must sum to 1.0
                before sampling.
        """
        # validate sampling configuration
        if probability is not None and likelihood is not None:
            raise ValueError(
                "Specify only one of likelihood or probability for a signal generator"
            )
        using_probability_mode = self._signal_probability_mode == "probability"
        using_likelihood_mode = (
            self._signal_probability_mode == "likelihood"
            and len(self.signal_generators) > 0
        )

        if using_probability_mode and probability is None:
            raise ValueError(
                "All signal generators must specify probability once probability mode is used"
            )

        if using_likelihood_mode and probability is not None:
            raise ValueError(
                "Cannot mix explicit probability with likelihood/default likelihood generators"
            )
        if probability is not None:
            self._signal_probability_mode = "probability"
            probability = self._validate_positive_weight(probability, "probability")

            candidate_probabilities = np.append(
                np.asarray(self.signal_probabilities, dtype=float),
                probability,
            )
            probability_sum = float(np.sum(candidate_probabilities))
            if probability_sum > (1.0 + 1e-8):
                raise ValueError(
                    "signal probabilities must sum to 1.0 or less while "
                    f"configuring the dataset, found {probability_sum}"
                )
        else:
            if probability is not None and likelihood is not None:
                raise ValueError("Specify only one of likelihood or probability")

            if self._signal_probability_mode == "probability" and probability is None:
                raise ValueError(
                    "All signal generators must specify probability once probability mode is used"
                )

            if probability is not None and len(self.signal_generators) > 0 and self._signal_probability_mode == "likelihood":
                raise ValueError(
                    "Cannot mix explicit probability with likelihood/default likelihood generators"
                )
            if likelihood is None:
                likelihood = 1.0
            likelihood = self._validate_positive_weight(likelihood, "likelihood")

        if isinstance(signal_generator, Seedable):
            signal_generator.add_parent(self)

        if self.validate_init:
            signal_generator.validate_metadata_fields()
        if self.validate_init:
            self.validate_metadata_fields()
        signal_generator["class_index"] = len(self.signal_generators)
        if class_index is None:
            signal_generator["class_index"] = len(self.signal_generators)
        else:
            signal_generator["class_index"] = class_index
        self.signal_generators += [signal_generator]
        if class_name is not None:
            signal_generator["class_name"] = class_name
        if (
            hasattr(signal_generator, "class_name")
            and signal_generator["class_name"] is not None
        ):
            self["class_names"] += [signal_generator["class_name"]]

        if self._signal_probability_mode == "probability":
            self.signal_probabilities = np.append(
                np.asarray(self.signal_probabilities, dtype=float),
                probability,
            )
        else:
            self.signal_likelihoods += [likelihood]
        self._refresh_signal_probabilities()


    def validate_metadata_fields(self) -> bool:
        """Validate metadata for all signal generators."""

        for generator in self.signal_generators:
            try:
                generator.validate_metadata_fields()
            except AttributeError:
                # Generators are not required to implement validation.
                pass

        return True

    def __iter__(self):
        """Returns an iterator object for the dataset.

        Returns:
            An iterator object that yields samples from the dataset.
        """
        return self

    def __next__(self) -> Signal | np.ndarray | tuple:
        """Returns a dataset sample and (optionally) corresponding targets for a given index.

        Returns:
            The sample data and the target values.

        Raises:
            IndexError: If the index is out of bounds of the generated samples.
        """
        # user requesting another sample at index +1 larger than current list of generates samples
        # generate new sample
        sample = self.__generate_new_signal__()
        return apply_transforms_and_labels_to_signal(
            sample, self.transforms, self.target_labels
        )

    def __call__(self) -> Signal | np.ndarray | tuple:
        """Same as next(); returns the next item in the dataset.

        Allows datasets to be treated as signal generators for other datasets.
        """
        return next(self)

    def __repr__(self) -> str:
        """Returns a string representation of the dataset.

        Returns:
            String representation of the dataset.
        """
        repr_str = f"{self.__class__.__name__}("
        if self.metadata is not None:
            repr_str += "metadata="
            repr_str += str(self.metadata)
            repr_str += ", "
        if self.transforms is not None:
            repr_str += "transforms="
            repr_str += str(self.transforms)
            repr_str += ", "
        if self.signal_generators is not None:
            repr_str += "signal_generators="
            repr_str += str(self.signal_generators)
            repr_str += ", "
        repr_str += ")"
        return repr_str

    def _build_noise_floor(self) -> np.ndarray:
        """Generates the noise floor for the dataset by creating an IQ sample and applying a frequency-domain noise estimation.

        Returns:
            The generated IQ samples representing the noise floor.
        """
        real_samples = self.random_generator.normal(
            0, 1, self["num_iq_samples_dataset"]
        )
        imag_samples = self.random_generator.normal(
            0, 1, self["num_iq_samples_dataset"]
        )
        # combine real and imaginary portions of noise
        iq_samples = real_samples + 1j * imag_samples
        # compute an estimate of the noise floor in the frequency domain. use a large stride to process a subset
        # of the data since not many FFTs are needed to be averaged for the noise
        noise_spectrogram_db = compute_spectrogram(
            iq_samples, self["fft_size"], self["fft_stride"] * 16
        )
        # average over time
        noise_fft_db = np.mean(noise_spectrogram_db, axis=1)
        # estimate the average noise value in dB in the frequency domain
        noise_avg_db = np.mean(noise_fft_db)
        # compute the correction factor as the distance from the desired level
        correction_db = self["noise_power_db"] - noise_avg_db
        # apply the correction
        correction = 10 ** (correction_db / 10)
        iq_samples = np.sqrt(correction) * iq_samples
        return iq_samples.astype(np.complex64)

    def __generate_new_signal__(self) -> Signal:
        """Generate a new synthetic dataset sample.

        The generated sample consists of a noise floor plus zero or more component
        signals placed into the IQ buffer. Component signals are generated at
        complex baseband, optionally transformed, updated with SNR/bandwidth
        metadata, frequency shifted, checked for overlap, and then inserted into
        the dataset sample.

        Returns:
            A generated ``Signal`` containing IQ data and component signal metadata.
        """
        iq_samples = self._build_noise_floor()
        signals = []
        signal_rectangle_list = []

        num_signals_to_generate = self.random_generator.integers(
            low=self["num_signals_min"],
            high=self["num_signals_max"] + 1,
        )

        max_attempts = 10 * num_signals_to_generate
        num_signals_created = 0

        for _ in range(max_attempts):
            if num_signals_created >= num_signals_to_generate:
                break

            new_signal = self._generate_component_signal()
            start_sample = self._choose_start_sample(iq_samples, new_signal)
            new_rectangle = self._map_to_coordinates(new_signal, start_sample)

            has_overlap = self._check_if_overlap(
                new_rectangle,
                signal_rectangle_list,
            )
            allow_overlap = (
                self.random_generator.uniform(0, 1)
                < self["cochannel_overlap_probability"]
            )

            if has_overlap and not allow_overlap:
                continue

            signal_rectangle_list.append(new_rectangle)
            self._insert_component_signal(iq_samples, new_signal, start_sample)
            signals.append(new_signal)
            num_signals_created += 1

        sample = Signal(
            data=iq_samples,
            component_signals=signals,
            center_freq=0,
            bandwidth=max([0] + [signal.bandwidth for signal in signals]),
        )

        if hasattr(self, "class_name"):
            sample.class_name = self.class_name

        if sample.parent is None:
            sample.add_parent(self, register=False)

        return sample

    def _generate_component_signal(self) -> Signal:
        """Generate, transform, update, and frequency-shift one component signal."""
        generator = self._random_signal_generator()
        new_signal = generator()

        for component_transform in self.component_transforms:
            new_signal = component_transform(new_signal)

        update_signal_snr_bandwidth(self, new_signal)

        return frequency_shift_signal(
            new_signal,
            center_freq_min=self["signal_center_freq_min"],
            center_freq_max=self["signal_center_freq_max"],
            sample_rate=self["sample_rate"],
            frequency_max=self["frequency_max"],
            frequency_min=self["frequency_min"],
            random_generator=self.random_generator,
        )


    def _choose_start_sample(self, iq_samples: np.ndarray, signal: Signal) -> int:
        """Choose a valid start sample for placing a component signal."""
        num_available_samples = len(iq_samples)
        num_signal_samples = len(signal.data)

        if num_signal_samples > num_available_samples:
            warnings.warn(
                "generated signal is too large to fit in the dataset sample; "
                "it will be cut off",
                UserWarning,
                stacklevel=2,
            )

        max_start = max(num_available_samples - num_signal_samples, 1)

        return int(
            self.random_generator.integers(
                low=0,
                high=max_start,
            )
        )


    def _insert_component_signal(
        self,
        iq_samples: np.ndarray,
        signal: Signal,
        start_sample: int,
    ) -> None:
        """Insert a component signal into the dataset IQ buffer."""
        stop_sample = min(start_sample + len(signal.data), len(iq_samples))
        num_samples_to_add = stop_sample - start_sample

        if num_samples_to_add < len(signal.data):
            signal["duration_in_samples"] = num_samples_to_add

        iq_samples[start_sample:stop_sample] += signal.data[:num_samples_to_add]
        signal["start_in_samples"] = start_sample

    def _map_to_coordinates(self, new_signal: Signal, start_sample: int) -> Rectangle:
        """Maps a new signal to coordinates based on the start sample and signal characteristics.

        Args:
            new_signal: The new signal to map.
            start_sample: The starting sample index of the new signal.

        Returns:
            A rectangle object representing the mapped coordinates of the new signal in the frequency domain.

        Notes:
            This function computes the start and stop times in terms of Fast Fourier Transform (FFT) length using the provided
            start sample and the length of the new signal's data. It also calculates the bin positions in the FFT based on
            the signal's center frequency, bandwidth, and the sample rate. Finally, it maps these positions into rectangle
            coordinates, which it returns as a `Rectangle` object.
        """
        # calculate start and stop time in terms of FFT number
        fft_start_time = np.round(start_sample / self["fft_size"])
        fft_stop_time = np.round(
            (start_sample + len(new_signal.data)) / self["fft_size"]
        )
        # calculate bin position in FFT
        fs = self["sample_rate"]
        fft_start_bin_norm = (
            (new_signal.center_freq - new_signal.bandwidth) + (fs / 2)
        ) / (fs / 2)
        fft_stop_bin_norm = (
            (new_signal.center_freq + new_signal.bandwidth) + (fs / 2)
        ) / (fs / 2)
        fft_start_bin_index = np.round(fft_start_bin_norm * self["fft_size"])
        fft_stop_bin_index = np.round(fft_stop_bin_norm * self["fft_size"])
        # map the position into retangle coordinates
        lower_left_coord = Coordinate(fft_start_time, fft_start_bin_index)
        upper_right_coord = Coordinate(fft_stop_time, fft_stop_bin_index)
        # turn into a rectangle
        return Rectangle(lower_left_coord, upper_right_coord)

    def _check_if_overlap(
        self, new_rectangle: Rectangle, signal_rectangle_list: list
    ) -> bool:
        """Determines if a new rectangle overlaps with any of the rectangles in a list.

        Args:
            new_rectangle: The new rectangle to check for overlap.
            signal_rectangle_list: A list of rectangles to check against for overlap.

        Returns:
            True if the new rectangle overlaps with any rectangle in the list, otherwise False.
        """
        # initialize the boolean value which determines if there is overlap or not
        has_overlap = False
        # determine if overlap
        if len(signal_rectangle_list) > 0:
            # check to see if the current rectangle overlaps with any signals currently
            # in the spectrogram
            for reference_box in signal_rectangle_list:
                # check for invidivual overlap
                individual_overlap = is_rectangle_overlap(new_rectangle, reference_box)
                # combine with previous potential overlap checks
                has_overlap = has_overlap or individual_overlap
        return has_overlap

    def _random_signal_generator(self) -> BaseSignalGenerator:
        """Randomly selects which signal generator to use next."""
        if len(self.signal_generators) == 0:
            raise ValueError("cannot sample from a dataset with no signal generators")

        self._validate_signal_sampling_configuration(require_complete=True)
        self._refresh_signal_probabilities()
        return self.random_generator.choice(
            self.signal_generators, p=self.signal_probabilities
        )

class SafeTorchSigIterableDataset(
    PipelineFailOverEnabled, TorchSigIterableDataset
):
    """A fault-tolerant version of TorchSigIterableDataset with automatic error recovery.

    This class behaves exactly like :class:`TorchSigIterableDataset` but adds
    built-in recovery mechanisms for when transforms fail during data generation.
    It's designed to prevent dataset generation from stopping due to transform errors
    by either retrying failed operations or falling back to safe outputs.

    The class maintains full compatibility with the parent class API while adding
    configurable error handling through the ``pipeline_fallback`` and
    ``pipeline_max_retries`` attributes.

    Example:
        >>> ds = SafeTorchSigIterableDataset(
        ...     signal_generators="all",
        ...     transforms=[MyTransform()],
        ...     target_labels=["class_index"]
        ... )
        >>> # Configure fallback behavior
        >>> ds.pipeline_fallback = "retry"
        >>> ds.pipeline_max_retries = 3
        >>> # Dataset will now retry failed transforms up to 3 times
        >>> sample = next(ds)
    """

    def __next__(self) -> Any:
        """Generate the next dataset sample with pipeline fault tolerance.

        Sample creation is performed in two stages:

        1. Generate a raw signal, including any component-level transforms.
        2. Apply whole-signal transforms and target label generation.

        Each stage is executed through the configured failover mechanism. If a
        stage raises an exception, the behavior is controlled by
        ``pipeline_fallback``:

        - ``"original"``: Return the original raw signal when available.
        - ``"zero"``: Return a zero-filled signal with matching shape.
        - ``"retry"``: Retry the failed stage up to
        ``pipeline_max_retries`` times before falling back.

        If failure occurs during raw signal generation, no original signal exists
        and only retry or zero fallbacks are possible. If failure occurs during
        whole-signal transforms or label generation, the generated raw signal can
        be used as the fallback result.

        Returns:
            A successfully generated sample, or a fallback sample if recovery
            logic is triggered.

        Note:
            All pipeline failures are logged to aid debugging and monitoring.
        """
        # Stage 1: generation + component_transforms
        # If this fails, no raw/original sample exists yet.
        def generate_raw_signal():
            return self.__generate_new_signal__()

        raw_signal = self._run_with_fallback(
            generate_raw_signal,
            fallback_raw_signal=None,
        )

        # Stage 2: whole-signal transforms + labels
        # If this fails, raw_signal exists, so original fallback is possible.
        def transform_raw_signal():
            return apply_transforms_and_labels_to_signal(
                raw_signal,
                self.transforms,
                self.target_labels,
            )

        return self._run_with_fallback(
            transform_raw_signal,
            fallback_raw_signal=raw_signal,
        )


    def set_fallback_policy(
        self,
        fallback: Literal["original", "zero", "retry"] = "original",
        max_retries: int | None = None,
    ) -> None:
        """Configure the dataset's error recovery behavior.

        Args:
            fallback: The recovery strategy to use when transforms fail:
                - "original": Return the untransformed signal
                - "zero": Return a zero-filled array of matching shape
                - "retry": Attempt the transform again (requires max_retries)

            max_retries: Maximum number of retry attempts when fallback="retry".
                Must be a positive integer. Ignored for other fallback modes.

        Raises:
            ValueError: If max_retries is provided with a fallback mode other than "retry"

        Example:
            >>> ds = SafeTorchSigIterableDataset(...)
            >>> # Configure to retry failed transforms up to 5 times
            >>> ds.set_fallback_policy(fallback="retry", max_retries=5)
        """
        self.pipeline_fallback = fallback
        if max_retries is not None:
            if fallback != "retry":
                raise ValueError("max_retries is only allowed with fallback='retry'")
            self.pipeline_max_retries = max_retries


class StaticTorchSigDataset(Dataset, Seedable):
    """Static Dataset class, which loads pre-generated data from a directory.

    Args:
        root: The root directory where the dataset is stored.
        transforms: Transforms to apply to the data (default: []).
        file_handler_class: Class used for reading the dataset (default: HDF5FileHandler).
    """

    def __init__(
        self,
        root: str,
        file_handler_class=HDF5Reader,
        transforms: list = [],
        target_labels: list | None = None,
        **kwargs,
    ):
        """Initializes the dataset.

        Args:
            root: The root directory where the dataset is stored.
            file_handler_class: Class used for reading the dataset.
            transforms: Transforms to apply to the data.
            target_labels: Labels to extract from the signal.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        self.root = Path(root)
        self.reader = file_handler_class(root=self.root)

        Seedable.__init__(self, **kwargs)
        self.transforms = transforms
        for transform in self.transforms:
            transform.add_parent(self)
        self.target_labels = target_labels

        # dataset size
        self.dataset_length = len(self.reader)

        self._verify()

    def _verify(self) -> None:
        """Checks if root exists

        Raises:
            ValueError: Root does not exist.
        """
        # check root

        if not self.root.exists():
            raise ValueError(f"root does not exist: {self.root}")

    def __len__(self) -> int:
        """Returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return self.dataset_length

    def __getitem__(self, idx: int) -> tuple[np.ndarray, tuple]:
        """Retrieves a sample from the dataset by index.

        Args:
            idx: The index of the sample to retrieve.

        Returns:
            The data and targets for the sample.

        Raises:
            IndexError: If the index is out of bounds.
        """
        if 0 <= idx < len(self):
            sample = self.reader.read(idx=idx)
            return apply_transforms_and_labels_to_signal(
                sample, self.transforms, self.target_labels
            )

        raise IndexError(
            f"Index {idx} is out of bounds. Must be [0, {self.__len__() - 1}]"
        )

    def __getitems__(
        self,
        indices: list[int],
    ) -> list[Signal | np.ndarray | tuple]:
        """Retrieve a DataLoader batch, using native contiguous reads when available.

        Readers without ``read_signals_batch`` and non-contiguous index lists use
        the existing single-item path.
        """
        if not indices:
            return []
        if any(idx < 0 or idx >= len(self) for idx in indices):
            invalid_idx = next(
                idx for idx in indices if idx < 0 or idx >= len(self)
            )
            raise IndexError(
                f"Index {invalid_idx} is out of bounds. "
                f"Must be [0, {self.__len__() - 1}]"
            )

        read_signals_batch = getattr(self.reader, "read_signals_batch", None)
        contiguous = all(
            idx == indices[0] + offset
            for offset, idx in enumerate(indices)
        )
        if read_signals_batch is None or not contiguous:
            return [self[idx] for idx in indices]

        samples = read_signals_batch(indices[0], indices[-1] + 1)
        return [
            apply_transforms_and_labels_to_signal(
                sample,
                self.transforms,
                self.target_labels,
            )
            for sample in samples
        ]

    def __str__(self) -> str:
        """Returns a string representation of the dataset.

        Returns:
            A string representation of the dataset.
        """
        return f"{self.__class__.__name__}: {self.root}"

    def __repr__(self) -> str:
        """Returns a detailed string representation of the dataset.

        Returns:
            A detailed string representation of the dataset.
        """
        return (
            f"{self.__class__.__name__}"
            f"(root={self.root}, "
            f"file_handler_class={self.reader})"
        )
