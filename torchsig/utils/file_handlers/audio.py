"""Shared record indexing for audio-backed IQ datasets."""

from __future__ import annotations

import os
from collections import OrderedDict
from contextlib import suppress
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any

import soundfile as sf

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy as np

__all__ = ["AudioHandleCache", "AudioRecord", "AudioRecordLayout"]

_MANIFEST_REQUIRED_COLUMNS = ("file_path", "start_frame", "num_frames")


class AudioHandleCache:
    """Manage a bounded set of process-local ``SoundFile`` handles.

    A size of zero disables caching. The cache notices process changes before
    every access, which prevents handles inherited by forked DataLoader workers
    from being reused. Live handles are also removed from pickle state.
    """

    def __init__(self, max_size: int = 8) -> None:
        if isinstance(max_size, bool) or not isinstance(max_size, int):
            raise TypeError("audio_handle_cache_size must be an integer")
        if max_size < 0:
            raise ValueError("audio_handle_cache_size must be nonnegative")
        self.max_size = max_size
        self._handles: OrderedDict[Path, sf.SoundFile] = OrderedDict()
        self._pid = os.getpid()

    def _reset_after_process_change(self) -> None:
        """Discard inherited handles when the current process changes."""
        pid = os.getpid()
        if pid != self._pid:
            self.close()
            self._pid = pid

    def setup(self) -> None:
        """Prepare the cache in the current process."""
        self._reset_after_process_change()

    def _get(self, path: Path) -> sf.SoundFile:
        """Return an open handle and update its LRU position."""
        self._reset_after_process_change()
        handle = self._handles.pop(path, None)
        if handle is None:
            handle = sf.SoundFile(path, mode="r")
        self._handles[path] = handle
        if len(self._handles) > self.max_size:
            _, evicted = self._handles.popitem(last=False)
            evicted.close()
        return handle

    def read(self, path: Path, start_frame: int, num_frames: int) -> np.ndarray:
        """Read exactly the requested frame range from ``path``."""
        if self.max_size == 0:
            self._reset_after_process_change()
            with sf.SoundFile(path, mode="r") as handle:
                handle.seek(start_frame)
                return handle.read(frames=num_frames, dtype="float32", always_2d=True)
        handle = self._get(path)
        handle.seek(start_frame)
        return handle.read(frames=num_frames, dtype="float32", always_2d=True)

    def close(self) -> None:
        """Close and remove every cached handle."""
        while self._handles:
            _, handle = self._handles.popitem(last=False)
            handle.close()

    def __getstate__(self) -> dict[str, Any]:
        """Return pickle state without live process-local handles."""
        return {"max_size": self.max_size, "_handles": OrderedDict(), "_pid": None}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore an empty cache in the receiving process."""
        self.max_size = state["max_size"]
        self._handles = OrderedDict()
        self._pid = os.getpid()

    def __del__(self) -> None:
        """Best-effort cleanup for readers not explicitly torn down."""
        with suppress(Exception):
            self.close()

    def __len__(self) -> int:
        """Return the number of cached handles."""
        return len(self._handles)


@dataclass(frozen=True)
class AudioRecord:
    """Describe one IQ record stored as a segment of an audio file."""

    path: Path
    start_frame: int
    num_frames: int
    expected_sample_rate: float | None = None
    channel_count: int | None = None


class AudioRecordLayout:
    """Build and validate a global index of audio record segments.

    Header-based metadata containing ``file_path``, ``start_frame``, and
    ``num_frames`` uses the explicit manifest format. Otherwise, the legacy
    fixed-length ``num_iq_samples``/``elements_per_file`` layout is adapted
    into equivalent descriptors.
    """

    def __init__(
        self,
        root: Path,
        metadata_rows: Iterable[dict[str, Any]],
        audio_files: Iterable[Path],
        *,
        num_iq_samples: int,
        elements_per_file: int,
    ) -> None:
        self.root = root.resolve()
        self.audio_files = sorted((path.resolve() for path in audio_files), key=str)
        rows = list(metadata_rows)
        all_columns = {key for row in rows for key in row}
        manifest_columns = set(_MANIFEST_REQUIRED_COLUMNS).intersection(all_columns)
        if manifest_columns:
            missing = sorted(set(_MANIFEST_REQUIRED_COLUMNS).difference(all_columns))
            if missing:
                raise ValueError(f"Audio record manifest is missing required columns: {missing}")
            self.records = [self._record_from_manifest(row, position) for position, row in enumerate(rows)]
            self.is_manifest = True
        else:
            self.records = self._legacy_records(
                len(rows),
                num_iq_samples=num_iq_samples,
                elements_per_file=elements_per_file,
            )
            self.is_manifest = False
        self._validate_segments()

    @staticmethod
    def _parse_nonnegative_int(value: Any, name: str, position: int) -> int:
        """Parse a manifest integer without accepting booleans or fractions."""
        if isinstance(value, bool):
            raise TypeError(f"Audio record {position} has invalid {name} {value!r}")
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Audio record {position} has invalid {name} {value!r}") from exc
        if str(value).strip() not in {str(parsed), f"{parsed}.0"}:
            raise ValueError(f"Audio record {position} has invalid {name} {value!r}")
        if parsed < 0:
            raise ValueError(f"Audio record {position} has negative {name} {parsed}")
        return parsed

    def _resolve_path(self, value: Any, position: int) -> Path:
        """Resolve a relative manifest path while containing it in the root."""
        raw_path = Path(str(value))
        if raw_path.is_absolute():
            raise ValueError(f"Audio record {position} path must be relative: {value!r}")
        path = (self.root / raw_path).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"Audio record {position} path escapes dataset root: {value!r}") from exc
        if path not in self.audio_files or not path.is_file():
            raise ValueError(f"Audio record {position} does not resolve to an audio file: {value!r}")
        return path

    def _record_from_manifest(self, row: dict[str, Any], position: int) -> AudioRecord:
        """Convert one metadata row into an explicit record descriptor."""
        missing = [name for name in _MANIFEST_REQUIRED_COLUMNS if not str(row.get(name, "")).strip()]
        if missing:
            raise ValueError(f"Audio record {position} is missing manifest values: {missing}")
        sample_rate = row.get("expected_sample_rate")
        channel_count = row.get("channel_count")
        try:
            expected_sample_rate = None if sample_rate in (None, "") else float(sample_rate)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Audio record {position} has invalid expected_sample_rate {sample_rate!r}") from exc
        if expected_sample_rate is not None and expected_sample_rate < 0:
            raise ValueError(f"Audio record {position} has negative expected_sample_rate {expected_sample_rate}")
        return AudioRecord(
            path=self._resolve_path(row["file_path"], position),
            start_frame=self._parse_nonnegative_int(row["start_frame"], "start_frame", position),
            num_frames=self._parse_nonnegative_int(row["num_frames"], "num_frames", position),
            expected_sample_rate=expected_sample_rate,
            channel_count=(None if channel_count in (None, "") else self._parse_nonnegative_int(channel_count, "channel_count", position)),
        )

    def _legacy_records(
        self,
        dataset_size: int,
        *,
        num_iq_samples: int,
        elements_per_file: int,
    ) -> list[AudioRecord]:
        """Adapt the historical uniform file layout to record descriptors."""
        if num_iq_samples <= 0 or elements_per_file <= 0:
            raise ValueError("Legacy audio datasets require positive num_iq_samples and elements_per_file")
        capacity = len(self.audio_files) * elements_per_file
        if dataset_size != capacity:
            raise ValueError(f"Metadata contains {dataset_size} elements, but audio layout implies {capacity}.")
        return [
            AudioRecord(
                path=self.audio_files[index // elements_per_file],
                start_frame=(index % elements_per_file) * num_iq_samples,
                num_frames=num_iq_samples,
            )
            for index in range(dataset_size)
        ]

    def _validate_segments(self) -> None:
        """Ensure all segments fit their files and do not overlap."""
        frames_by_path: dict[Path, int] = {}
        intervals_by_path: dict[Path, list[tuple[int, int, int]]] = {}
        for position, record in enumerate(self.records):
            frames = frames_by_path.setdefault(record.path, int(sf.info(record.path).frames))
            end_frame = record.start_frame + record.num_frames
            if end_frame > frames:
                raise ValueError(f"Audio record {position} segment [{record.start_frame}, {end_frame}) exceeds {record.path} length {frames}")
            intervals_by_path.setdefault(record.path, []).append((record.start_frame, end_frame, position))
        for path, intervals in intervals_by_path.items():
            intervals.sort()
            for previous, current in pairwise(intervals):
                if current[0] < previous[1]:
                    raise ValueError(f"Audio records {previous[2]} and {current[2]} overlap in {path}")

    def __len__(self) -> int:
        """Return the number of indexed audio records."""
        return len(self.records)

    def __getitem__(self, index: int) -> AudioRecord:
        """Return the descriptor at a global record index."""
        return self.records[index]
