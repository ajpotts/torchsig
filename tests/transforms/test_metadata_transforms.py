"""Tests for metadata transforms."""

import numpy as np
import pytest

from torchsig.signals.signal_types import Signal
from torchsig.transforms.metadata_transforms import (
    MetadataTransform,
    MultiHotLabel,
    YOLOLabel,
)


class DummyMetadataTransform(MetadataTransform):
    """Concrete metadata transform for testing base behavior."""

    def __init__(self, required_metadata=None):
        super().__init__(required_metadata=required_metadata or [])
        self.applied_signals = []

    def __apply__(self, signal):
        self.applied_signals.append(signal)
        signal["was_applied"] = True
        return signal


@pytest.fixture
def component_signal():
    return Signal(
        data=np.zeros(1000, dtype=np.complex64),
        class_index=3,
        start_in_samples=250,
        duration_in_samples=500,
        num_iq_samples_dataset=1000,
        bandwidth=200.0,
        center_freq=100.0,
        sample_rate=1000.0,
        dataset_metadata={"sample_rate": 1000.0},
    )


@pytest.fixture
def parent_signal(component_signal):
    """Create a parent Signal containing one component signal."""
    return Signal(
        data=None,
        component_signals=[component_signal],
        sample_rate=1000.0,
        dataset_metadata={"sample_rate": 1000.0},
    )


def test_metadata_transform_validate_accepts_signal_with_required_metadata(component_signal):
    transform = DummyMetadataTransform(required_metadata=["class_index", "start"])

    validated_signal = transform.__validate__(component_signal)

    assert validated_signal is component_signal


def test_metadata_transform_validate_rejects_non_signal():
    transform = DummyMetadataTransform()

    with pytest.raises(TypeError, match="is not a Signal object"):
        transform.__validate__({"class_index": 1})


def test_metadata_transform_validate_rejects_missing_required_metadata(component_signal):
    transform = DummyMetadataTransform(required_metadata=["missing_field"])

    with pytest.raises(
        ValueError,
        match="key: missing_field is missing from signal metadata",
    ):
        transform.__validate__(component_signal)


def test_metadata_transform_call_applies_to_each_component_signal(parent_signal):
    transform = DummyMetadataTransform()

    transformed_signal = transform(parent_signal)

    assert transformed_signal is parent_signal
    assert len(transform.applied_signals) == 1
    assert transform.applied_signals[0] is parent_signal.component_signals[0]
    assert parent_signal.component_signals[0].was_applied is True


def test_metadata_transform_base_apply_raises_not_implemented(component_signal):
    transform = MetadataTransform()

    with pytest.raises(NotImplementedError):
        transform.__apply__(component_signal)


def test_metadata_transform_repr_excludes_required_metadata():
    transform = DummyMetadataTransform(required_metadata=["class_index"])

    repr_str = repr(transform)

    assert "DummyMetadataTransform" in repr_str
    assert "required_metadata" not in repr_str


def test_multi_hot_label_encodes_unique_component_classes():
    signal = Signal(
        data=np.zeros(100, dtype=np.complex64),
        component_signals=[
            Signal(class_index=1),
            Signal(class_index=3),
            Signal(class_index=1),
        ],
        class_names=["a", "b", "c", "d"],
    )

    transformed = MultiHotLabel()(signal)

    np.testing.assert_array_equal(
        transformed.multi_hot_label,
        np.array([0, 1, 0, 1], dtype=np.float32),
    )


def test_multi_hot_label_supports_leaf_and_explicit_class_count():
    signal = Signal(class_index=np.int64(2))

    transformed = MultiHotLabel(num_classes=4, output_key="classes")(signal)

    np.testing.assert_array_equal(
        transformed.classes,
        np.array([0, 0, 1, 0], dtype=np.float32),
    )


def test_multi_hot_label_encodes_empty_composite_as_all_zero():
    signal = Signal(class_names=["a", "b", "c"])

    transformed = MultiHotLabel()(signal)

    np.testing.assert_array_equal(
        transformed.multi_hot_label,
        np.zeros(3, dtype=np.float32),
    )


@pytest.mark.parametrize("num_classes", [0, -1, 2.5, True])
def test_multi_hot_label_rejects_invalid_class_count(num_classes):
    with pytest.raises(ValueError, match="positive integer"):
        MultiHotLabel(num_classes=num_classes)


def test_multi_hot_label_requires_class_names_when_count_is_not_given():
    with pytest.raises(ValueError, match="class_names is missing"):
        MultiHotLabel()(Signal(class_index=0))


@pytest.mark.parametrize("class_index", [-1, 3])
def test_multi_hot_label_rejects_out_of_range_class_index(class_index):
    signal = Signal(class_index=class_index)

    with pytest.raises(ValueError, match="outside"):
        MultiHotLabel(num_classes=3)(signal)


def test_multi_hot_label_rejects_non_integer_class_index():
    signal = Signal(class_index=1.5)

    with pytest.raises(TypeError, match="must be an integer"):
        MultiHotLabel(num_classes=3)(signal)


def test_yolo_label_initializes_expected_metadata_fields():
    transform = YOLOLabel()

    assert transform.required_metadata == [
        "class_index",
        "start",
        "bandwidth",
        "center_freq",
        "dataset_metadata",
    ]
    assert transform.targets_metadata == ["yolo_label"]


def test_yolo_label_adds_expected_label_to_component_signal(parent_signal):
    transform = YOLOLabel()

    transformed_signal = transform(parent_signal)

    component_signal = transformed_signal.component_signals[0]
    assert component_signal.yolo_label == pytest.approx(
        (
            3,      # class_index
            0.5,    # start + duration / 2
            0.4,    # 1 - ((sample_rate / 2 + center_freq) / sample_rate)
            0.5,    # duration
            0.2,    # bandwidth / sample_rate
        )
    )


def test_yolo_label_apply_returns_component_signal(component_signal):
    transform = YOLOLabel()

    transformed_component = transform.__apply__(component_signal)

    assert transformed_component is component_signal
    assert transformed_component.yolo_label == pytest.approx((3, 0.5, 0.4, 0.5, 0.2))
