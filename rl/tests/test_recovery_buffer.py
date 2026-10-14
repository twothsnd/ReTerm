from pathlib import Path

from reterm_core.types import AgentRunResult, AttemptRecord, AttemptStatus, StateSnapshot
from reterm_rl.llm_agent.recovery_buffer import NodeLocalRecoveryBuffer


def _attempt(root: Path, name: str) -> AttemptRecord:
    snapshot = root / name
    snapshot.mkdir()
    (snapshot / "manifest.json").write_text("{}", encoding="utf-8")
    return AttemptRecord(
        attempt_index=1,
        task_id=name,
        prompt=name,
        status=AttemptStatus.FAILED,
        agent_result=AgentRunResult(),
        state_after=StateSnapshot(payload={"state_path": str(snapshot)}),
    )


def test_newest_first_and_oldest_eviction(tmp_path: Path) -> None:
    buffer = NodeLocalRecoveryBuffer(capacity=2)
    a = _attempt(tmp_path, "a")
    b = _attempt(tmp_path, "b")
    c = _attempt(tmp_path, "c")
    buffer.put(task_id="a", initial_attempt=a, created_step=1, source_group_id=1)
    buffer.put(task_id="b", initial_attempt=b, created_step=1, source_group_id=2)
    buffer.put(task_id="c", initial_attempt=c, created_step=2, source_group_id=3)
    assert not (tmp_path / "a").exists()
    lease_id, entry = buffer.lease_newest()
    assert entry.task_id == "c"
    buffer.finish(lease_id, consume=False)
    _lease_id, entry = buffer.lease_newest()
    assert entry.task_id == "c"


def test_provisional_promotes_exactly_one_failure(tmp_path: Path) -> None:
    buffer = NodeLocalRecoveryBuffer(capacity=12)
    for index in range(3):
        buffer.offer_provisional(
            rollout_id="r1",
            task_id=f"t{index}",
            initial_attempt=_attempt(tmp_path, f"p{index}"),
            created_step=3,
            source_group_id=7,
        )
    selected = buffer.resolve_provisional(
        rollout_id="r1", source_group_id=7, promote=True
    )
    assert selected is not None
    assert buffer.status()["available"] == 1
    assert buffer.status()["provisional"] == 0
    assert sum((tmp_path / f"p{index}").exists() for index in range(3)) == 1


def test_cancelled_lease_returns_to_original_position_behind_new_states(tmp_path: Path) -> None:
    buffer = NodeLocalRecoveryBuffer(capacity=12)
    for index, name in enumerate(("oldest", "middle", "newest"), start=1):
        buffer.put(
            task_id=name,
            initial_attempt=_attempt(tmp_path, name),
            created_step=index,
            source_group_id=index,
        )

    lease_id, leased = buffer.lease_newest()
    assert leased.task_id == "newest"
    buffer.put(
        task_id="fresh",
        initial_attempt=_attempt(tmp_path, "fresh"),
        created_step=4,
        source_group_id=4,
    )
    resolution = buffer.finish(lease_id, consume=False)

    assert resolution["requeued"] is True
    assert resolution["requeue_position"] == 1
    observed = []
    while buffer.status()["available"]:
        next_lease, entry = buffer.lease_newest()
        observed.append(entry.task_id)
        buffer.finish(next_lease, consume=True)
    assert observed == ["fresh", "newest", "middle", "oldest"]


def test_same_step_states_are_not_recovery_eligible(tmp_path: Path) -> None:
    buffer = NodeLocalRecoveryBuffer(capacity=12)
    buffer.put(
        task_id="previous",
        initial_attempt=_attempt(tmp_path, "previous"),
        created_step=6,
        source_group_id=1,
    )
    buffer.put(
        task_id="current",
        initial_attempt=_attempt(tmp_path, "current"),
        created_step=7,
        source_group_id=2,
    )

    lease_id, entry = buffer.lease_newest(max_created_step=6)
    assert entry.task_id == "previous"
    buffer.finish(lease_id, consume=True)
    lease_id, entry = buffer.lease_newest(max_created_step=7)
    assert entry.task_id == "current"
    buffer.finish(lease_id, consume=True)


def test_excluded_entry_is_skipped_without_changing_queue_order(tmp_path: Path) -> None:
    buffer = NodeLocalRecoveryBuffer(capacity=12)
    for index, name in enumerate(("oldest", "middle", "newest"), start=1):
        buffer.put(
            task_id=name,
            initial_attempt=_attempt(tmp_path, name),
            created_step=index,
            source_group_id=index,
        )

    newest_id = buffer.status()["entries"][0]["entry_id"]
    lease_id, entry = buffer.lease_newest(excluded_entry_ids={newest_id})
    assert entry.task_id == "middle"
    buffer.finish(lease_id, consume=True)

    lease_id, entry = buffer.lease_newest()
    assert entry.task_id == "newest"
    buffer.finish(lease_id, consume=True)


def test_leased_entries_count_toward_capacity(tmp_path: Path) -> None:
    buffer = NodeLocalRecoveryBuffer(capacity=2)
    buffer.put(task_id="a", initial_attempt=_attempt(tmp_path, "a"), created_step=1, source_group_id=1)
    buffer.put(task_id="b", initial_attempt=_attempt(tmp_path, "b"), created_step=1, source_group_id=2)
    lease_id, leased = buffer.lease_newest()
    buffer.put(task_id="c", initial_attempt=_attempt(tmp_path, "c"), created_step=2, source_group_id=3)

    # The leased state and the fresh state fill the two slots; the oldest
    # available state is evicted rather than letting a later requeue overflow.
    assert buffer.status()["available"] == 1
    assert not (tmp_path / "a").exists()
    resolution = buffer.finish(lease_id, consume=False)
    assert resolution["requeued"] is True
    assert buffer.status()["available"] == 2
    assert leased.task_id == "b"


def test_promoted_state_moves_out_of_transient_snapshot_tree(tmp_path: Path) -> None:
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    buffer_root = tmp_path / "recovery_buffer"
    attempt = _attempt(snapshots, "transient")
    buffer = NodeLocalRecoveryBuffer(capacity=12, storage_root=buffer_root)
    buffer.offer_provisional(
        rollout_id="rollout",
        task_id="task",
        initial_attempt=attempt,
        created_step=1,
        source_group_id=3,
    )
    metadata = buffer.resolve_provisional(
        rollout_id="rollout", source_group_id=3, promote=True
    )
    assert metadata is not None
    assert not (snapshots / "transient").exists()
    lease_id, entry = buffer.lease_newest()
    assert entry.snapshot_dir.parent == buffer_root.resolve()
    assert entry.initial_attempt.state_after.payload["state_path"] == str(entry.snapshot_dir)
    buffer.finish(lease_id, consume=True)
