"""Metadata Transforms"""

import numpy as np

from torchsig.signals.signal_types import Signal
from torchsig.transforms.base_transforms import Transform
from torchsig.utils.printing import generate_repr_str

__all__ = ["MetadataTransform", "MultiHotLabel", "YOLOLabel"]


## Base/Helper Classes
class MetadataTransform(Transform):
    """Base class for metadata transforms.

    This class defines the basic structure of a metadata transform, which includes:
    - The ability to validate metadata before applying the transform.
    - A method for applying the transform on signal metadata.
    - A callable interface to apply the transform to a list of signal metadata.

    Attributes:
        required_metadata: List of metadata fields required for applying the target transform.

    Methods:
        __validate(metadata): Validates the signal metadata before applying the transform.
        __apply(metadata): Applies the target transform to the metadata. Should be overridden by subclasses.
        __call__(signal): Applies the transform to a list of signal metadata dictionaries.
        __str__(): Returns the string representation of the transform.
        __repr__(): Returns a detailed string representation of the transform object.
    """

    def __init__(self, required_metadata: list[str] = [], **kwargs) -> None:
        """Initialize the MetadataTransform.

        Args:
            required_metadata: List of metadata fields required for applying the target transform.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(required_metadata=required_metadata, **kwargs)

    def __validate__(self, signal):
        """Validate signal metadata before applying target transforms.

        Makes sure a signal has all required metadata for a transform;
        returns the original signal if it is valid; raises an exception otherwise.

        Args:
            signal: The signal to validate.

        Raises:
            ValueError: If metadata is missing required metadata fields or if input is not a Signal object.
        """
        if not isinstance(signal, Signal):
            raise TypeError(f"input ({type(signal)}) is not a Signal object.")
        for required_metadatum in self.required_metadata:
            if not hasattr(signal, required_metadatum):
                raise ValueError(
                    f"key: {required_metadatum} is missing from signal metadata, but is required by {self.__class__.__name__}"
                )
        return signal

    def __call__(self, signal: Signal) -> Signal:
        """Applies the target transform to a list of signal metadata.

        Args:
            signal: The signal to transform.

        Returns:
            The transformed signal.
        """
        for component_signal in signal.component_signals:
            self.__apply__(component_signal)
        return signal

    def __apply__(self, signal):
        """Applies the target transform to a single signal metadata.

        Args:
            signal: The signal to transform.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        """Returns a detailed string representation of the transform object.

        Returns:
            A string representation of the transform object.
        """
        return generate_repr_str(self, exclude_params=["required_metadata"])


class MultiHotLabel(MetadataTransform):
    """Add a sample-level multi-hot classification label.

    Each class present in a signal's components is represented by a ``1`` in
    the output vector. Repeated instances of the same class still produce a
    single ``1``. For a signal without components, its own ``class_index`` is
    used. An empty composite signal produces an all-zero vector.

    This transform is intended for multilabel classification of composite
    samples such as wideband signals. The number of classes can be supplied
    explicitly or inferred from the signal's ``class_names`` metadata.

    Args:
        num_classes: Length of the output vector. If ``None``, infer the
            length from the signal's ``class_names`` metadata.
        output_key: Metadata key under which to store the vector.
        **kwargs: Additional keyword arguments passed to the parent class.

    Attributes:
        targets_metadata: Metadata fields added by the transform.
    """

    def __init__(
        self,
        num_classes: int | None = None,
        output_key: str = "multi_hot_label",
        **kwargs,
    ) -> None:
        if num_classes is not None and (not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes < 1):
            raise ValueError("num_classes must be a positive integer or None")
        if not isinstance(output_key, str) or not output_key:
            raise ValueError("output_key must be a non-empty string")

        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.output_key = output_key
        self.targets_metadata = [output_key]

    def __call__(self, signal: Signal) -> Signal:
        """Add a multi-hot vector representing all classes in ``signal``.

        Args:
            signal: Signal whose component class indices should be encoded.

        Returns:
            The input signal with the sample-level label added.

        Raises:
            TypeError: If ``signal`` is not a ``Signal`` or a class index is
                not an integer.
            ValueError: If the class count cannot be inferred or a class
                index is outside the output vector.
        """
        self.__validate__(signal)

        num_classes = self.num_classes
        if num_classes is None:
            metadata = signal.get_full_metadata()
            if "class_names" not in metadata:
                raise ValueError("num_classes was not provided and class_names is missing from signal metadata")
            num_classes = len(metadata["class_names"])
            if num_classes < 1:
                raise ValueError("class_names must contain at least one class")

        label = np.zeros(num_classes, dtype=np.float32)
        if signal.component_signals:
            signals = signal.component_signals
        elif "class_index" in signal.metadata:
            signals = [signal]
        else:
            signals = []

        for component in signals:
            if not hasattr(component, "class_index"):
                raise ValueError("class_index is missing from signal metadata")
            class_index = component.class_index
            if not isinstance(class_index, (int, np.integer)) or isinstance(class_index, (bool, np.bool_)):
                raise TypeError("class_index must be an integer")
            if class_index < 0 or class_index >= num_classes:
                raise ValueError(f"class_index {class_index} is outside [0, {num_classes})")
            label[int(class_index)] = 1.0

        signal[self.output_key] = label
        return signal

    def __apply__(self, signal: Signal) -> Signal:
        """Apply the transform to a single signal.

        ``MultiHotLabel`` aggregates a complete sample in ``__call__`` and
        therefore does not apply labels independently to components.

        Args:
            signal: Signal to transform.

        Returns:
            The transformed signal.
        """
        return self(signal)


class YOLOLabel(MetadataTransform):
    """Adds a YOLO_label to a signal.

    This transform adds a YOLO_label to a signal in the form of a list of tuples (cid, cx, cy, width, height).

    Attributes:
        required_metadata: List of metadata fields required for applying the transform.
        targets_metadata: List of metadata fields that will be added by the transform.
    """

    def __init__(self, **kwargs):
        """Initialize the YOLOLabel transform.

        Args:
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(
            required_metadata=[
                "class_index",
                "start",
                "bandwidth",
                "center_freq",
                "dataset_metadata",
            ],
            **kwargs,
        )
        self.targets_metadata = ["yolo_label"]

    def __apply__(self, signal: Signal) -> Signal:
        """Applies the YOLOLabel transform to a single signal.

        Args:
            signal: The signal to transform.

        Returns:
            The transformed signal with YOLO_label added.
        """
        class_index = signal.class_index
        # normalized to width of sample
        width = signal.duration
        # normalize bandwidth with sample rate
        height = signal.bandwidth / signal.sample_rate
        x_center = signal.start + (width / 2.0)
        # normalize center frequency with sample rate
        # subtract from 1 since (0,0) for YOLO is upper left, but we define (0,0) lower left
        y_center = (
            1 - ((signal.sample_rate / 2.0) + signal.center_freq) / signal.sample_rate
        )
        yolo_label = (class_index, x_center, y_center, width, height)
        signal["yolo_label"] = yolo_label
        return signal


__all__ = ["MetadataTransform", "MultiHotLabel", "YOLOLabel"]
