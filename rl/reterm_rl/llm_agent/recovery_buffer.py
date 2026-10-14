"""Node-local failure-state queue for ReTerm-GRPO.

State snapshots stay with the worker that produced them. Entries are sampled
newest-first, and the oldest entry is evicted when the queue is full.
"""

from __future__ import annotations

import shutil
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from reterm_core.types import AttemptRecord


@dataclass(slots=True)
class RecoveryBufferEntry:
    entry_id: str
    task_id: str
    initial_attempt: AttemptRecord
    snapshot_dir: Path
    created_step: int
    created_at: float
    source_group_id: int | str
    # Monotonic node-local insertion order.  A cancelled lease retains this
    # value so it can return to its original relative position while newer
    # Initial failures remain ahead of it.
    queue_order: int = 0

    def public_metadata(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "task_id": self.task_id,
            "created_step": self.created_step,
            "created_at": self.created_at,
            "source_group_id": self.source_group_id,
            "queue_order": self.queue_order,
        }


class NodeLocalRecoveryBuffer:
    """Thread-safe bounded LIFO-consumption/FIFO-eviction state store."""

    def __init__(self, capacity: int = 12, storage_root: Path | None = None) -> None:
        if capacity < 1:
            raise ValueError("recovery buffer capacity must be positive")
        self.capacity = int(capacity)
        self.storage_root = storage_root.expanduser().resolve() if storage_root else None
        if self.storage_root is not None:
            self.storage_root.mkdir(parents=True, exist_ok=True)
        self._entries: deque[RecoveryBufferEntry] = deque()
        self._leased: dict[str, RecoveryBufferEntry] = {}
        # key -> (number of failures seen, one uniform reservoir sample)
        self._provisional: dict[str, tuple[int, RecoveryBufferEntry]] = {}
        self._next_queue_order = 0
        self._lock = threading.RLock()

    def _insert_available_locked(
        self,
        entry: RecoveryBufferEntry,
        *,
        assign_new_order: bool,
    ) -> int:
        if assign_new_order:
            self._next_queue_order += 1
            entry.queue_order = self._next_queue_order
        elif entry.queue_order <= 0:
            raise ValueError("requeued recovery state has no original queue order")

        index = len(self._entries)
        for candidate_index, candidate in enumerate(self._entries):
            if candidate.queue_order < entry.queue_order:
                index = candidate_index
                break
        self._entries.insert(index, entry)
        return index

    def _evict_over_capacity_locked(self) -> list[RecoveryBufferEntry]:
        evicted: list[RecoveryBufferEntry] = []
        # Leases remain part of the bounded buffer.  New Initial states may
        # evict only currently available old states; an in-flight Recovery is
        # never deleted underneath its worker.
        while len(self._entries) + len(self._leased) > self.capacity and self._entries:
            evicted.append(self._entries.pop())
        return evicted

    @staticmethod
    def _snapshot_dir(attempt: AttemptRecord) -> Path:
        if attempt.state_after is None:
            raise ValueError("buffered failure attempt has no state_after snapshot")
        value = attempt.state_after.payload.get("state_path")
        if not value:
            raise ValueError("buffered failure snapshot has no state_path")
        path = Path(str(value)).resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"buffered failure snapshot is missing: {path}")
        return path

    def put(
        self,
        *,
        task_id: str,
        initial_attempt: AttemptRecord,
        created_step: int,
        source_group_id: int | str,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        entry = RecoveryBufferEntry(
            entry_id=uuid.uuid4().hex,
            task_id=str(task_id),
            initial_attempt=initial_attempt,
            snapshot_dir=self._snapshot_dir(initial_attempt),
            created_step=int(created_step),
            created_at=time.time(),
            source_group_id=source_group_id,
        )
        self._adopt(entry)
        evicted: list[RecoveryBufferEntry] = []
        with self._lock:
            self._insert_available_locked(entry, assign_new_order=True)
            evicted = self._evict_over_capacity_locked()
        for stale in evicted:
            self._delete(stale)
        return entry.public_metadata(), evicted[0].public_metadata() if evicted else None

    def offer_provisional(
        self,
        *,
        rollout_id: str,
        task_id: str,
        initial_attempt: AttemptRecord,
        created_step: int,
        source_group_id: int | str,
    ) -> dict[str, Any]:
        entry = RecoveryBufferEntry(
            entry_id=uuid.uuid4().hex,
            task_id=str(task_id),
            initial_attempt=initial_attempt,
            snapshot_dir=self._snapshot_dir(initial_attempt),
            created_step=int(created_step),
            created_at=time.time(),
            source_group_id=source_group_id,
        )
        key = f"{rollout_id}:{source_group_id}"
        discarded: RecoveryBufferEntry | None = None
        with self._lock:
            seen, selected = self._provisional.get(key, (0, entry))
            seen += 1
            # UUIDs are uniform; modulo implements the 1/seen replacement
            # probability without sharing mutable RNG state across threads.
            replace = seen == 1 or int(entry.entry_id, 16) % seen == 0
            if replace:
                if seen > 1:
                    discarded = selected
                self._adopt(entry)
                selected = entry
            else:
                discarded = entry
            self._provisional[key] = (seen, selected)
        if discarded is not None:
            self._delete(discarded)
        return selected.public_metadata()

    def resolve_provisional(
        self,
        *,
        rollout_id: str,
        source_group_id: int | str,
        promote: bool,
    ) -> dict[str, Any] | None:
        key = f"{rollout_id}:{source_group_id}"
        with self._lock:
            provisional = self._provisional.pop(key, None)
        if provisional is None:
            return None
        _seen, selected = provisional
        if not promote:
            self._delete(selected)
            return None
        self._adopt(selected)
        evicted: list[RecoveryBufferEntry] = []
        with self._lock:
            self._insert_available_locked(selected, assign_new_order=True)
            evicted = self._evict_over_capacity_locked()
        for stale in evicted:
            self._delete(stale)
        return selected.public_metadata()

    def discard_rollout_provisionals(self, rollout_id: str) -> int:
        prefix = f"{rollout_id}:"
        with self._lock:
            keys = [key for key in self._provisional if key.startswith(prefix)]
            entries = [self._provisional.pop(key)[1] for key in keys]
        for entry in entries:
            self._delete(entry)
        return len(entries)

    def lease_newest(
        self,
        *,
        max_created_step: int | None = None,
        excluded_entry_ids: set[str] | None = None,
    ) -> tuple[str, RecoveryBufferEntry] | None:
        excluded = excluded_entry_ids or set()
        with self._lock:
            selected_index = None
            for index, entry in enumerate(self._entries):
                if entry.entry_id in excluded:
                    continue
                if max_created_step is None or entry.created_step <= int(max_created_step):
                    selected_index = index
                    break
            if selected_index is None:
                return None
            entry = self._entries[selected_index]
            del self._entries[selected_index]
            lease_id = uuid.uuid4().hex
            self._leased[lease_id] = entry
            return lease_id, entry

    def finish(self, lease_id: str, *, consume: bool) -> dict[str, Any]:
        evicted: list[RecoveryBufferEntry] = []
        requeue_position: int | None = None
        with self._lock:
            entry = self._leased.pop(str(lease_id))
            if not consume:
                requeue_position = self._insert_available_locked(
                    entry,
                    assign_new_order=False,
                )
                evicted = self._evict_over_capacity_locked()
        metadata = entry.public_metadata()
        metadata.update(
            {
                "consumed": bool(consume),
                "requeued": bool(not consume and entry not in evicted),
                "requeue_position": requeue_position,
                "evicted_on_requeue": bool(entry in evicted),
            }
        )
        if consume:
            self._delete(entry)
        for stale in evicted:
            self._delete(stale)
        return metadata

    def get_lease(self, lease_id: str) -> RecoveryBufferEntry:
        with self._lock:
            return self._leased[str(lease_id)]

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "capacity": self.capacity,
                "available": len(self._entries),
                "leased": len(self._leased),
                "provisional": len(self._provisional),
                "entries": [entry.public_metadata() for entry in self._entries],
            }

    def close(self) -> None:
        with self._lock:
            entries = [
                *self._entries,
                *self._leased.values(),
                *(entry for _seen, entry in self._provisional.values()),
            ]
            self._entries.clear()
            self._leased.clear()
            self._provisional.clear()
        for entry in entries:
            self._delete(entry)

    @staticmethod
    def _delete(entry: RecoveryBufferEntry) -> None:
        # snapshot_dir identifies one saved environment state.
        shutil.rmtree(entry.snapshot_dir, ignore_errors=True)

    def _adopt(self, entry: RecoveryBufferEntry) -> None:
        """Move a promoted state out of the transient snapshot cleanup tree."""

        if self.storage_root is None or entry.snapshot_dir.parent == self.storage_root:
            return
        destination = self.storage_root / entry.entry_id
        if destination.exists():
            raise FileExistsError(f"recovery buffer destination already exists: {destination}")
        entry.snapshot_dir.rename(destination)
        state_after = entry.initial_attempt.state_after
        if state_after is None:
            raise ValueError("recovery buffer entry lost state_after during adoption")
        snapshot = replace(
            state_after,
            payload={**dict(state_after.payload), "state_path": str(destination)},
        )
        entry.initial_attempt = replace(entry.initial_attempt, state_after=snapshot)
        entry.snapshot_dir = destination
