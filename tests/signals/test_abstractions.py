"""Unit tests for HierarchicalMetadataObject."""

from __future__ import annotations

import pickle

import pytest

# Update this import if the module has a different name.
from torchsig.utils.abstractions import (
    HierarchicalMetadataObject,
    MetadataAttributeError,
    MetadataResolution,
)


class ExampleMetadataObject(HierarchicalMetadataObject):
    """Concrete subclass used to verify subclass-specific behavior."""

    class_attribute = "class value"


class RequiredArgumentMetadataObject(HierarchicalMetadataObject):
    """Subclass illustrating the constructor constraint imposed by copy()."""

    def __init__(self, required_value: str, **kwargs) -> None:
        self.required_value = required_value
        super().__init__(**kwargs)


def test_metadata_attribute_error_is_attribute_error():
    error = MetadataAttributeError("missing metadata")

    assert isinstance(error, AttributeError)
    assert str(error) == "missing metadata"


def test_initialization_without_metadata_creates_empty_metadata():
    obj = HierarchicalMetadataObject(seed=123)

    assert dict(obj.keys()) == {}
    assert obj["metadata"] == {}
    assert obj.rng_seed == 123


def test_initialization_copies_metadata_values():
    metadata = {"class_name": "bpsk", "snr_db": 10.0}

    obj = HierarchicalMetadataObject(metadata=metadata)

    assert obj["class_name"] == "bpsk"
    assert obj["snr_db"] == 10.0


def test_initialization_does_not_alias_input_metadata_dictionary():
    metadata = {"class_name": "bpsk"}

    obj = HierarchicalMetadataObject(metadata=metadata)
    metadata["class_name"] = "qpsk"
    metadata["new_field"] = 42

    assert obj["class_name"] == "bpsk"
    assert "new_field" not in obj.keys()


def test_keyword_metadata_overrides_metadata_dictionary():
    obj = HierarchicalMetadataObject(
        metadata={
            "class_name": "bpsk",
            "snr_db": 10.0,
        },
        snr_db=20.0,
        sample_rate=1_000_000,
    )

    assert obj["class_name"] == "bpsk"
    assert obj["snr_db"] == 20.0
    assert obj["sample_rate"] == 1_000_000


def test_metadata_can_be_accessed_as_attribute():
    obj = HierarchicalMetadataObject(
        metadata={
            "class_name": "bpsk",
            "snr_db": 10.0,
        }
    )

    assert obj.class_name == "bpsk"
    assert obj.snr_db == 10.0


def test_real_attribute_takes_precedence_over_metadata():
    obj = ExampleMetadataObject(
        metadata={"class_attribute": "metadata value"}
    )

    assert obj.class_attribute == "class value"
    assert obj["class_attribute"] == "metadata value"


def test_metadata_property_returns_copy():
    obj = HierarchicalMetadataObject(metadata={"field": "original"})

    returned_metadata = obj.metadata
    returned_metadata["field"] = "modified"
    returned_metadata["new_field"] = "new value"

    assert obj["field"] == "original"
    assert "new_field" not in obj.keys()


def test_getitem_returns_local_metadata_value():
    obj = HierarchicalMetadataObject(metadata={"field": 123})

    assert obj["field"] == 123


def test_getitem_inherits_value_from_parent():
    parent = HierarchicalMetadataObject(
        metadata={
            "sample_rate": 1_000_000,
            "center_freq": 100_000_000,
        }
    )
    child = HierarchicalMetadataObject(parent=parent)

    assert child["sample_rate"] == 1_000_000
    assert child["center_freq"] == 100_000_000


def test_child_metadata_overrides_parent_metadata():
    parent = HierarchicalMetadataObject(
        metadata={
            "sample_rate": 1_000_000,
            "center_freq": 100_000_000,
        }
    )
    child = HierarchicalMetadataObject(
        parent=parent,
        metadata={"center_freq": 101_000_000},
    )

    assert child["sample_rate"] == 1_000_000
    assert child["center_freq"] == 101_000_000


def test_metadata_inheritance_works_across_multiple_levels():
    grandparent = HierarchicalMetadataObject(
        metadata={
            "grandparent_only": 1,
            "overridden": "grandparent",
        }
    )
    parent = HierarchicalMetadataObject(
        parent=grandparent,
        metadata={
            "parent_only": 2,
            "overridden": "parent",
        },
    )
    child = HierarchicalMetadataObject(
        parent=parent,
        metadata={
            "child_only": 3,
            "overridden": "child",
        },
    )

    assert child.grandparent_only == 1
    assert child.parent_only == 2
    assert child.child_only == 3
    assert child.overridden == "child"


def test_get_full_metadata_combines_parent_and_child_metadata():
    parent = HierarchicalMetadataObject(
        metadata={
            "field_1": 4,
            "field_2": 5,
        }
    )
    child = HierarchicalMetadataObject(
        parent=parent,
        metadata={
            "field_2": 6,
            "field_3": 7,
        },
    )

    assert child.get_full_metadata() == {
        "field_1": 4,
        "field_2": 6,
        "field_3": 7,
    }


def test_get_full_metadata_combines_multiple_hierarchy_levels():
    grandparent = HierarchicalMetadataObject(
        metadata={
            "a": 1,
            "shared": "grandparent",
        }
    )
    parent = HierarchicalMetadataObject(
        parent=grandparent,
        metadata={
            "b": 2,
            "shared": "parent",
        },
    )
    child = HierarchicalMetadataObject(
        parent=parent,
        metadata={
            "c": 3,
            "shared": "child",
        },
    )

    assert child.get_full_metadata() == {
        "a": 1,
        "b": 2,
        "c": 3,
        "shared": "child",
    }


def test_get_full_metadata_returns_new_dictionary():
    obj = HierarchicalMetadataObject(metadata={"field": "original"})

    full_metadata = obj.get_full_metadata()
    full_metadata["field"] = "modified"

    assert obj["field"] == "original"


def test_explain_metadata_reports_local_key():
    obj = HierarchicalMetadataObject(metadata={"field": "value"})

    resolution = obj.explain_metadata("field")

    assert resolution == MetadataResolution(
        key="field",
        found=True,
        source="local",
        depth=0,
        owner_type="HierarchicalMetadataObject",
        overrides_parent=False,
        cycle_detected=False,
        path=("HierarchicalMetadataObject",),
    )


def test_explain_metadata_reports_inherited_key_and_depth():
    grandparent = ExampleMetadataObject(metadata={"field": "value"})
    parent = HierarchicalMetadataObject(parent=grandparent)
    child = HierarchicalMetadataObject(parent=parent)

    resolution = child.explain_metadata("field")

    assert resolution.found is True
    assert resolution.source == "inherited"
    assert resolution.depth == 2
    assert resolution.owner_type == "ExampleMetadataObject"
    assert resolution.overrides_parent is False
    assert resolution.cycle_detected is False
    assert resolution.path == (
        "HierarchicalMetadataObject",
        "HierarchicalMetadataObject",
        "ExampleMetadataObject",
    )


def test_explain_metadata_reports_parent_override():
    grandparent = HierarchicalMetadataObject(metadata={"field": "grandparent"})
    parent = HierarchicalMetadataObject(
        parent=grandparent,
        metadata={"field": "parent"},
    )
    child = HierarchicalMetadataObject(parent=parent)

    resolution = child.explain_metadata("field")

    assert resolution.source == "inherited"
    assert resolution.depth == 1
    assert resolution.overrides_parent is True


def test_explain_metadata_reports_local_override():
    parent = HierarchicalMetadataObject(metadata={"field": "parent"})
    child = HierarchicalMetadataObject(
        parent=parent,
        metadata={"field": "child"},
    )

    resolution = child.explain_metadata("field")

    assert resolution.source == "local"
    assert resolution.depth == 0
    assert resolution.overrides_parent is True


def test_explain_metadata_reports_missing_key():
    parent = ExampleMetadataObject(metadata={"other": "value"})
    child = HierarchicalMetadataObject(parent=parent)

    resolution = child.explain_metadata("missing")

    assert resolution == MetadataResolution(
        key="missing",
        found=False,
        source="missing",
        depth=None,
        owner_type=None,
        overrides_parent=False,
        cycle_detected=False,
        path=("HierarchicalMetadataObject", "ExampleMetadataObject"),
    )


def test_explain_metadata_detects_parent_cycle():
    parent = ExampleMetadataObject(metadata={"field": "value"})
    child = HierarchicalMetadataObject(parent=parent)
    parent.parent = child

    resolution = child.explain_metadata("field")

    assert resolution.found is True
    assert resolution.source == "inherited"
    assert resolution.depth == 1
    assert resolution.cycle_detected is True
    assert resolution.path == (
        "HierarchicalMetadataObject",
        "ExampleMetadataObject",
    )


def test_explain_metadata_rejects_non_string_key():
    obj = HierarchicalMetadataObject()

    with pytest.raises(TypeError, match="metadata key must be a string"):
        obj.explain_metadata(123)


def test_keys_returns_only_local_metadata_keys():
    parent = HierarchicalMetadataObject(metadata={"parent_field": 1})
    child = HierarchicalMetadataObject(
        parent=parent,
        metadata={"child_field": 2},
    )

    assert set(child.keys()) == {"child_field"}
    assert "parent_field" not in child.keys()


def test_setitem_adds_metadata():
    obj = HierarchicalMetadataObject()

    obj["field"] = "value"

    assert obj["field"] == "value"
    assert obj.field == "value"


def test_setitem_overrides_inherited_metadata_locally():
    parent = HierarchicalMetadataObject(metadata={"field": "parent"})
    child = HierarchicalMetadataObject(parent=parent)

    child["field"] = "child"

    assert child["field"] == "child"
    assert parent["field"] == "parent"
    assert "field" in child.keys()


def test_delitem_removes_local_metadata():
    obj = HierarchicalMetadataObject(metadata={"field": "value"})

    del obj["field"]

    assert "field" not in obj.keys()

    with pytest.raises(MetadataAttributeError):
        _ = obj["field"]


def test_delitem_reveals_inherited_parent_value():
    parent = HierarchicalMetadataObject(metadata={"field": "parent"})
    child = HierarchicalMetadataObject(
        parent=parent,
        metadata={"field": "child"},
    )

    del child["field"]

    assert child["field"] == "parent"
    assert "field" not in child.keys()


def test_delitem_missing_local_key_raises_key_error():
    parent = HierarchicalMetadataObject(metadata={"field": "parent"})
    child = HierarchicalMetadataObject(parent=parent)

    with pytest.raises(KeyError):
        del child["field"]


def test_getitem_for_missing_key_raises_metadata_attribute_error():
    obj = HierarchicalMetadataObject()

    with pytest.raises(
        MetadataAttributeError,
        match="key: 'missing' could not be found in metadata",
    ):
        _ = obj["missing"]


def test_attribute_access_for_missing_key_raises_metadata_attribute_error():
    obj = HierarchicalMetadataObject()

    with pytest.raises(
        MetadataAttributeError,
        match="key: 'missing' could not be found in metadata",
    ):
        _ = obj.missing


def test_key_lookup_returns_metadata_value():
    obj = HierarchicalMetadataObject(metadata={"field": 123})

    assert obj.key_lookup("field") == 123


def test_key_lookup_reports_missing_key():
    obj = HierarchicalMetadataObject()

    with pytest.raises(
        MetadataAttributeError,
        match=r"key missing: 'missing'",
    ):
        obj.key_lookup("missing")


def test_attribute_lookup_reports_missing_key():
    obj = HierarchicalMetadataObject()

    with pytest.raises(
        MetadataAttributeError,
        match=r"key missing: 'missing'",
    ):
        _ = obj.missing


def test_direct_metadata_getitem_is_rejected():
    obj = HierarchicalMetadataObject(metadata={"field": 123})

    with pytest.raises(KeyError, match="check metadata field names"):
        _ = obj["_metadata"]


def test_internal_metadata_attribute_remains_accessible():
    obj = HierarchicalMetadataObject(metadata={"field": 123})

    assert obj._metadata == {"field": 123}


def test_copy_creates_distinct_object():
    obj = HierarchicalMetadataObject(
        seed=123,
        metadata={"field": "value"},
    )

    copied = obj.copy()

    assert copied is not obj
    assert copied.get_full_metadata() == obj.get_full_metadata()
    assert copied.rng_seed == obj.rng_seed


def test_copy_has_independent_metadata_dictionary():
    obj = HierarchicalMetadataObject(metadata={"field": "original"})

    copied = obj.copy()
    copied["field"] = "modified"
    copied["new_field"] = "new value"

    assert obj["field"] == "original"
    assert "new_field" not in obj.keys()
    assert copied["field"] == "modified"


def test_copy_is_shallow():
    nested_value = {"items": [1, 2, 3]}
    obj = HierarchicalMetadataObject(metadata={"nested": nested_value})

    copied = obj.copy()

    assert copied["nested"] is obj["nested"]


def test_copy_preserves_parent_by_default():
    parent = HierarchicalMetadataObject(metadata={"parent_field": 1})
    obj = HierarchicalMetadataObject(
        parent=parent,
        metadata={"child_field": 2},
    )

    copied = obj.copy()

    assert copied.parent is parent
    assert copied.get_full_metadata() == {
        "parent_field": 1,
        "child_field": 2,
    }


def test_copy_can_detach_from_parent():
    parent = HierarchicalMetadataObject(metadata={"parent_field": 1})
    obj = HierarchicalMetadataObject(
        parent=parent,
        metadata={"child_field": 2},
    )

    copied = obj.copy(preserve_parent=False)

    assert copied.parent is None
    assert copied.get_full_metadata() == {"child_field": 2}

    with pytest.raises(MetadataAttributeError):
        _ = copied["parent_field"]


def test_copy_preserves_runtime_subclass():
    obj = ExampleMetadataObject(
        seed=123,
        metadata={"field": "value"},
    )

    copied = obj.copy()

    assert type(copied) is ExampleMetadataObject
    assert copied["field"] == "value"


def test_copy_requires_subclasses_to_support_base_constructor_contract():
    obj = RequiredArgumentMetadataObject(
        required_value="required",
        metadata={"field": "value"},
    )

    with pytest.raises(TypeError):
        obj.copy()


def test_setstate_updates_instance_dictionary():
    obj = HierarchicalMetadataObject(metadata={"old": "value"})

    obj.__setstate__(
        {
            "_metadata": {"new": "value"},
            "additional_attribute": 123,
        }
    )

    assert obj["new"] == "value"
    assert obj.additional_attribute == 123

    with pytest.raises(MetadataAttributeError):
        _ = obj["old"]


def test_object_can_be_pickled_and_unpickled():
    parent = HierarchicalMetadataObject(
        seed=100,
        metadata={"parent_field": 1},
    )
    child = HierarchicalMetadataObject(
        seed=200,
        parent=parent,
        metadata={"child_field": 2},
    )

    restored = pickle.loads(pickle.dumps(child))

    assert restored is not child
    assert restored.parent is not parent
    assert restored.rng_seed == child.rng_seed
    assert restored.get_full_metadata() == {
        "parent_field": 1,
        "child_field": 2,
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (False, False),
        (0, 0),
        ("", ""),
        ([], []),
        ({}, {}),
    ],
)
def test_metadata_supports_falsy_values(value, expected):
    obj = HierarchicalMetadataObject(metadata={"field": value})

    assert obj["field"] == expected


def test_key_lookup_missing_key_has_descriptive_error():
    obj = HierarchicalMetadataObject()

    with pytest.raises(
        MetadataAttributeError,
        match=r"key missing: 'missing'",
    ):
        obj.key_lookup("missing")
