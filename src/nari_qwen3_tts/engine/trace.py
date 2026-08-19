"""Engine-owned behavior-neutral observation for scheduling and commits.

The traffic path owns only compact immutable tuples in a preallocated overwrite
ring.  Public dictionaries are reconstructed only when the trace is drained.
"""

from __future__ import annotations

import ctypes
import gc
import pickle
import time
from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True, slots=True)
class TraceEvent:
    sequence: int
    kind: str
    fields: tuple[tuple[str, object], ...]

    def normalized(self) -> dict[str, object]:
        return {"sequence": self.sequence, "kind": self.kind, **dict(self.fields)}


_MAPPING_EVENT = 0
_PACKED_FIELDS = 1
_PACKED_DISPATCH = 2
_PACKED_DECISION = 3


class _FrozenMarker(Enum):
    MAPPING = 1


_FROZEN_MAPPING = _FrozenMarker.MAPPING
_PYOBJECT_GC_UNTRACK = ctypes.pythonapi.PyObject_GC_UnTrack
_PYOBJECT_GC_UNTRACK.argtypes = [ctypes.py_object]
_PYOBJECT_GC_UNTRACK.restype = None

_WORK_FIELD_NAMES = (
    "request_id",
    "version",
    "lane",
    "stage",
    "logical_step",
    "admission_sequence",
    "ready_sequence",
    "startup",
    "deadline_s",
    "reserve_s",
)
_ROW_FIELD_NAMES = (
    "physical_row",
    "request_id",
    "version",
    "lane",
    "stage",
    "logical_step",
)


class TraceRecorder:
    """Fixed-capacity single-writer trace with after-drain normalization."""

    def __init__(
        self,
        *,
        enabled: bool,
        max_events: int = 10_000,
        direct_object_ring: bool = False,
        timestamps: bool = False,
    ) -> None:
        if isinstance(max_events, bool) or not isinstance(max_events, int):
            raise TypeError("trace max_events must be an integer")
        if max_events < 1:
            raise ValueError("trace max_events must be positive")
        if enabled and direct_object_ring and gc.isenabled():
            raise RuntimeError("direct-object trace requires cyclic GC to be disabled")
        self.enabled = enabled
        self.max_events = max_events
        self._direct_object_ring = direct_object_ring
        self._timestamps = timestamps
        capacity = max_events if enabled else 0
        self._sequence_ring = [0] * capacity
        self._timestamp_ring = [0] * capacity
        self._kind_ring: list[str | None] = [None] * capacity
        self._codec_ring = [0] * capacity
        self._payload_ring: list[bytes | tuple[object, ...] | None] = [None] * capacity
        self._sequence = 0
        if capacity:
            # These fixed rings contain only primitives.  In GC-safe mode the
            # payload ring contains only bytes; direct objects remain tracked
            # and are allowed only while cyclic GC is disabled.
            _PYOBJECT_GC_UNTRACK(self._sequence_ring)
            _PYOBJECT_GC_UNTRACK(self._timestamp_ring)
            _PYOBJECT_GC_UNTRACK(self._kind_ring)
            _PYOBJECT_GC_UNTRACK(self._codec_ring)
            if not direct_object_ring:
                _PYOBJECT_GC_UNTRACK(self._payload_ring)

    @classmethod
    def _freeze(cls, value: object) -> object:
        if isinstance(value, dict):
            return (
                _FROZEN_MAPPING,
                tuple((key, cls._freeze(item)) for key, item in value.items()),
            )
        if isinstance(value, (list, tuple)):
            return tuple(cls._freeze(item) for item in value)
        if isinstance(value, (set, frozenset)):
            return tuple(cls._freeze(item) for item in value)
        return value

    @classmethod
    def _thaw(cls, value: object) -> object:
        if (
            isinstance(value, tuple)
            and len(value) == 2
            and value[0] is _FROZEN_MAPPING
        ):
            items = value[1]
            assert isinstance(items, tuple)
            return {key: cls._thaw(item) for key, item in items}
        if isinstance(value, tuple):
            return tuple(cls._thaw(item) for item in value)
        return value

    def _store(self, kind: str, codec: int, payload: tuple[object, ...]) -> None:
        self._sequence += 1
        slot = (self._sequence - 1) % self.max_events
        self._timestamp_ring[slot] = time.perf_counter_ns() if self._timestamps else 0
        self._payload_ring[slot] = (
            payload
            if self._direct_object_ring
            else pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        )
        self._kind_ring[slot] = kind
        self._codec_ring[slot] = codec
        # Publish sequence last so a later snapshot never pairs a wrapped
        # sequence with the previous occupant's payload.
        self._sequence_ring[slot] = self._sequence

    def record(self, kind: str, **fields: object) -> None:
        """Record an infrequent mapping event with an owned immutable snapshot."""

        if not self.enabled:
            return
        payload = tuple((name, self._freeze(value)) for name, value in fields.items())
        self._store(kind, _MAPPING_EVENT, payload)

    def record_packed_fields(
        self,
        kind: str,
        field_names: tuple[str, ...],
        values: tuple[object, ...],
    ) -> None:
        """Record a trusted acyclic tuple without constructing a kwargs dict."""

        if not self.enabled:
            return
        if len(field_names) != len(values):
            raise ValueError("packed trace field names and values must have equal length")
        self._store(kind, _PACKED_FIELDS, (field_names, values))

    def record_dispatch(self, payload: tuple[object, ...]) -> None:
        """Record a compact dispatch payload for after-drain row expansion."""

        if self.enabled:
            self._store("dispatch", _PACKED_DISPATCH, payload)

    def record_decision(self, payload: tuple[object, ...]) -> None:
        """Record a compact scheduling decision for after-drain expansion."""

        if self.enabled:
            self._store("decision", _PACKED_DECISION, payload)

    @staticmethod
    def _compatibility_normalized(packed: object) -> dict[str, object]:
        schema, values = packed
        type_name, field_names = schema
        return {"type": type_name, **dict(zip(field_names, values, strict=True))}

    @classmethod
    def _work_normalized(cls, packed: object) -> dict[str, object]:
        values = packed
        return {
            **dict(zip(_WORK_FIELD_NAMES, values[:-1], strict=True)),
            "compatibility": cls._compatibility_normalized(values[-1]),
        }

    @classmethod
    def _row_normalized(cls, packed: object) -> dict[str, object]:
        values = packed
        return {
            **dict(zip(_ROW_FIELD_NAMES, values[:6], strict=True)),
            "compatibility": cls._compatibility_normalized(values[6]),
            "padding": values[7],
        }

    @classmethod
    def _dispatch_normalized(cls, payload: tuple[object, ...]) -> dict[str, object]:
        decision_id, plan_id, stage, request_ids, row_manifest = payload
        return {
            "decision_id": decision_id,
            "plan_id": plan_id,
            "stage": stage,
            "request_ids": request_ids,
            "row_manifest": tuple(cls._row_normalized(row) for row in row_manifest),
        }

    @classmethod
    def _decision_normalized(cls, payload: tuple[object, ...]) -> dict[str, object]:
        (
            decision_id,
            snapshot_sequence,
            ready,
            eligible,
            selected,
            compatibility_partition,
            split_pad,
            wait_reasons,
            policy,
            policy_inputs,
            rr_counterfactual,
            row_manifest,
        ) = payload
        return {
            "decision_id": decision_id,
            "snapshot_sequence": snapshot_sequence,
            "ready": tuple(cls._work_normalized(work) for work in ready),
            "eligible": tuple(cls._work_normalized(work) for work in eligible),
            "selected": tuple(cls._work_normalized(work) for work in selected),
            "compatibility_partition": tuple(
                {
                    "stage": stage,
                    "compatibility": cls._compatibility_normalized(compatibility),
                    "request_ids": request_ids,
                }
                for stage, compatibility, request_ids in compatibility_partition
            ),
            "split_pad": split_pad,
            "wait_reasons": wait_reasons,
            "policy": policy,
            "policy_inputs": policy_inputs,
            "rr_counterfactual": rr_counterfactual,
            "row_manifest": tuple(
                tuple(cls._row_normalized(row) for row in batch) for batch in row_manifest
            ),
        }

    @classmethod
    def _normalize_payload(
        cls,
        codec: int,
        payload: tuple[object, ...],
    ) -> dict[str, object]:
        if codec == _MAPPING_EVENT:
            return {name: cls._thaw(value) for name, value in payload}
        if codec == _PACKED_FIELDS:
            field_names, values = payload
            return dict(zip(field_names, values, strict=True))
        if codec == _PACKED_DISPATCH:
            return cls._dispatch_normalized(payload)
        if codec == _PACKED_DECISION:
            return cls._decision_normalized(payload)
        raise RuntimeError(f"unknown trace codec {codec}")

    def normalized(self) -> tuple[dict[str, object], ...]:
        if not self.enabled or self._sequence == 0:
            return ()
        retained = min(self._sequence, self.max_events)
        first_sequence = self._sequence - retained + 1
        events: list[dict[str, object]] = []
        for sequence in range(first_sequence, self._sequence + 1):
            slot = (sequence - 1) % self.max_events
            if self._sequence_ring[slot] != sequence:
                raise RuntimeError("trace ring contains an incoherent slot")
            kind = self._kind_ring[slot]
            stored_payload = self._payload_ring[slot]
            assert kind is not None and stored_payload is not None
            payload = (
                stored_payload
                if self._direct_object_ring
                else pickle.loads(stored_payload)
            )
            assert isinstance(payload, tuple)
            events.append(
                {
                    "sequence": sequence,
                    "kind": kind,
                    **(
                        {"monotonic_ns": self._timestamp_ring[slot]}
                        if self._timestamps
                        else {}
                    ),
                    **self._normalize_payload(self._codec_ring[slot], payload),
                }
            )
        return tuple(events)

    @property
    def total_events(self) -> int:
        return self._sequence

    @property
    def dropped_events(self) -> int:
        return max(0, self._sequence - self.max_events) if self.enabled else 0


__all__ = ["TraceEvent", "TraceRecorder"]
