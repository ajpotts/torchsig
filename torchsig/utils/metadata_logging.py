"""Correlation context for structured TorchSIG metadata logging."""

from __future__ import annotations

import os
import threading
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

MetadataContextValue: TypeAlias = str | int | float | bool | None
_CONTEXT_VALUE_LIMIT = 200


@dataclass(frozen=True)
class MetadataLoggingContext:
    """Correlation information attached to metadata log records.

    Attributes:
        session_id: Identifier shared by records from one debug session.
        dataset_id: Optional dataset identifier.
        sample_index: Optional generated or loaded sample index.
        worker_id: Optional DataLoader or application worker identifier.
        correlation_fields: Additional normalized user-defined fields.
    """

    session_id: str | None = None
    dataset_id: str | None = None
    sample_index: MetadataContextValue = None
    worker_id: MetadataContextValue = None
    correlation_fields: tuple[tuple[str, MetadataContextValue], ...] = ()

    @property
    def fields(self) -> Mapping[str, MetadataContextValue]:
        """Return user-defined correlation fields as a read-only mapping."""
        return MappingProxyType(dict(self.correlation_fields))


_metadata_logging_context: ContextVar[MetadataLoggingContext | None] = ContextVar(
    "torchsig_metadata_logging_context",
    default=None,
)


def _normalize_context_value(
    name: str,
    value: MetadataContextValue,
) -> MetadataContextValue:
    """Validate and bound a correlation value."""
    if not isinstance(value, (str, int, float, bool, type(None))):
        raise TypeError(f"metadata logging context value {name!r} must be a scalar or None")
    if isinstance(value, str) and len(value) > _CONTEXT_VALUE_LIMIT:
        return value[: _CONTEXT_VALUE_LIMIT - 3] + "..."
    return value


def get_metadata_logging_context() -> MetadataLoggingContext:
    """Return the correlation context active in the current execution context."""
    return _metadata_logging_context.get() or MetadataLoggingContext()


@contextmanager
def metadata_logging_context(
    *,
    session_id: str | None = None,
    dataset_id: str | None = None,
    sample_index: MetadataContextValue = None,
    worker_id: MetadataContextValue = None,
    fields: Mapping[str, MetadataContextValue] | None = None,
) -> Iterator[MetadataLoggingContext]:
    """Temporarily attach correlation information to metadata log records.

    Unspecified named values inherit from the surrounding context. When no
    session identifier exists, a new UUID is generated. User-defined fields are
    merged with surrounding fields, with inner values taking precedence.

    Args:
        session_id: Debug-session identifier, or ``None`` to inherit or create.
        dataset_id: Dataset identifier, or ``None`` to inherit.
        sample_index: Sample index, or ``None`` to inherit.
        worker_id: Worker identifier, or ``None`` to inherit.
        fields: Additional scalar correlation fields.

    Yields:
        The effective context for the duration of the block.

    Raises:
        TypeError: If identifiers, field names, or field values are invalid.
    """
    current = get_metadata_logging_context()
    if session_id is not None and not isinstance(session_id, str):
        raise TypeError("session_id must be a string or None")
    if dataset_id is not None and not isinstance(dataset_id, str):
        raise TypeError("dataset_id must be a string or None")
    if fields is not None and not hasattr(fields, "items"):
        raise TypeError("fields must be a mapping or None")

    merged_fields = dict(current.correlation_fields)
    if fields is not None:
        for key, value in fields.items():
            if not isinstance(key, str) or not key:
                raise TypeError("metadata logging context field names must be non-empty strings")
            merged_fields[key] = _normalize_context_value(key, value)

    effective_session_id = session_id or current.session_id or str(uuid.uuid4())
    effective_dataset_id = dataset_id if dataset_id is not None else current.dataset_id
    effective_sample_index = sample_index if sample_index is not None else current.sample_index
    effective_worker_id = worker_id if worker_id is not None else current.worker_id
    context = MetadataLoggingContext(
        session_id=_normalize_context_value("session_id", effective_session_id),
        dataset_id=_normalize_context_value("dataset_id", effective_dataset_id),
        sample_index=_normalize_context_value("sample_index", effective_sample_index),
        worker_id=_normalize_context_value("worker_id", effective_worker_id),
        correlation_fields=tuple(sorted(merged_fields.items())),
    )
    token = _metadata_logging_context.set(context)
    try:
        yield context
    finally:
        _metadata_logging_context.reset(token)


def _metadata_logging_extra() -> dict[str, object]:
    """Build standard correlation fields for a metadata log record."""
    context = get_metadata_logging_context()
    return {
        "metadata_session_id": context.session_id,
        "metadata_dataset_id": context.dataset_id,
        "metadata_sample_index": context.sample_index,
        "metadata_worker_id": context.worker_id,
        "metadata_process_id": os.getpid(),
        "metadata_thread_id": threading.get_ident(),
        "metadata_correlation_fields": dict(context.correlation_fields),
    }


__all__ = [
    "MetadataContextValue",
    "MetadataLoggingContext",
    "get_metadata_logging_context",
    "metadata_logging_context",
]
