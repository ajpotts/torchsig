"""Tests for metadata logging correlation context."""

from __future__ import annotations

import asyncio
import logging
import os
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from torchsig.utils.abstractions import HierarchicalMetadataObject
from torchsig.utils.metadata_logging import (
    MetadataLoggingContext,
    get_metadata_logging_context,
    metadata_logging_context,
)


def test_metadata_logging_context_is_empty_by_default():
    assert get_metadata_logging_context() == MetadataLoggingContext()


def test_metadata_logging_context_sets_named_and_custom_fields():
    with metadata_logging_context(
        session_id="session-1",
        dataset_id="wideband-train",
        sample_index=42,
        worker_id=2,
        fields={"split": "train", "attempt": 3},
    ) as context:
        assert context.session_id == "session-1"
        assert context.dataset_id == "wideband-train"
        assert context.sample_index == 42
        assert context.worker_id == 2
        assert dict(context.fields) == {"attempt": 3, "split": "train"}
        assert get_metadata_logging_context() is context

    assert get_metadata_logging_context() == MetadataLoggingContext()


def test_metadata_logging_context_generates_session_id():
    with metadata_logging_context() as context:
        assert isinstance(context.session_id, str)
        assert context.session_id


def test_nested_metadata_logging_context_inherits_and_overrides():
    with metadata_logging_context(
        session_id="outer-session",
        dataset_id="dataset",
        sample_index=1,
        fields={"split": "train", "attempt": 1},
    ) as outer:
        with metadata_logging_context(
            sample_index=2,
            worker_id=4,
            fields={"attempt": 2},
        ) as inner:
            assert inner.session_id == "outer-session"
            assert inner.dataset_id == "dataset"
            assert inner.sample_index == 2
            assert inner.worker_id == 4
            assert dict(inner.fields) == {"attempt": 2, "split": "train"}

        assert get_metadata_logging_context() is outer


def test_metadata_logging_context_restores_after_exception():
    with (
        pytest.raises(
            RuntimeError,
            match="test error",
        ),
        metadata_logging_context(session_id="session"),
    ):
        raise RuntimeError("test error")

    assert get_metadata_logging_context() == MetadataLoggingContext()


def test_metadata_logging_context_is_isolated_from_new_thread():
    with metadata_logging_context(session_id="main-thread"):
        with ThreadPoolExecutor(max_workers=1) as executor:
            child_context = executor.submit(get_metadata_logging_context).result()

        assert child_context == MetadataLoggingContext()
        assert get_metadata_logging_context().session_id == "main-thread"


def test_metadata_logging_context_is_isolated_between_async_tasks():
    async def observe_context(sample_index: int) -> int:
        with metadata_logging_context(sample_index=sample_index):
            await asyncio.sleep(0)
            return get_metadata_logging_context().sample_index

    async def run_tasks() -> list[int]:
        return await asyncio.gather(observe_context(1), observe_context(2))

    assert asyncio.run(run_tasks()) == [1, 2]
    assert get_metadata_logging_context() == MetadataLoggingContext()


def test_metadata_logging_context_is_pickle_safe():
    context = MetadataLoggingContext(
        session_id="session",
        dataset_id="dataset",
        sample_index=1,
        worker_id=2,
        correlation_fields=(("split", "train"),),
    )

    restored = pickle.loads(pickle.dumps(context))  # noqa: S301

    assert restored == context


def test_metadata_logging_context_bounds_string_values():
    long_value = "x" * 300

    with metadata_logging_context(
        session_id=long_value,
        dataset_id=long_value,
        fields={"long": long_value},
    ) as context:
        assert len(context.session_id) == 200
        assert context.session_id.endswith("...")
        assert len(context.dataset_id) == 200
        assert len(context.fields["long"]) == 200


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"session_id": 1}, "session_id must be"),
        ({"dataset_id": 1}, "dataset_id must be"),
        ({"sample_index": []}, "sample_index.*must be a scalar"),
        ({"worker_id": {}}, "worker_id.*must be a scalar"),
        ({"fields": []}, "fields must be a mapping"),
        ({"fields": {1: "value"}}, "field names must be"),
        ({"fields": {"": "value"}}, "field names must be"),
        ({"fields": {"field": []}}, "field.*must be a scalar"),
    ],
)
def test_metadata_logging_context_rejects_invalid_values(kwargs, match):
    with pytest.raises(TypeError, match=match), metadata_logging_context(**kwargs):
        pass


def test_metadata_records_include_correlation_and_execution_fields(caplog):
    caplog.set_level(logging.DEBUG, logger="torchsig.metadata")
    obj = HierarchicalMetadataObject(metadata={"field": "value"})
    obj.enable_metadata_debug()

    with metadata_logging_context(
        session_id="session",
        dataset_id="dataset",
        sample_index=7,
        worker_id=3,
        fields={"split": "validation"},
    ):
        assert obj["field"] == "value"

    record = caplog.records[-1]
    assert record.metadata_session_id == "session"
    assert record.metadata_dataset_id == "dataset"
    assert record.metadata_sample_index == 7
    assert record.metadata_worker_id == 3
    assert record.metadata_process_id == os.getpid()
    assert record.metadata_thread_id == threading.get_ident()
    assert record.metadata_correlation_fields == {"split": "validation"}


def test_metadata_summary_uses_context_active_when_disabled(caplog):
    caplog.set_level(logging.DEBUG, logger="torchsig.metadata")
    obj = HierarchicalMetadataObject(metadata={"field": "value"})
    obj.enable_metadata_debug()

    with metadata_logging_context(session_id="summary-session"):
        assert obj["field"] == "value"
        obj.disable_metadata_debug()

    summary = caplog.records[-1]
    assert summary.metadata_event == "summary"
    assert summary.metadata_session_id == "summary-session"
