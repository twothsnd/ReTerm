from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class Task:
    """A benchmark task description."""

    task_id: str
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TaskOutcome:
    """Outcome of evaluating a single attempt."""

    success: bool
    score: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    """A benchmark state snapshot or opaque state handle."""

    payload: Any
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ActionRecord:
    """One agent action and optional observation."""

    action: Any
    observation: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AgentContext:
    """Context passed to an agent for one attempt."""

    benchmark: str
    task_id: str
    attempt_index: int
    state_before: StateSnapshot | None = None
    previous_attempts: tuple["AttemptRecord", ...] = ()
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    """The result of one agent attempt."""

    actions: tuple[ActionRecord, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class AttemptStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    """One initial or recovery attempt."""

    attempt_index: int
    task_id: str
    prompt: str
    status: AttemptStatus
    agent_result: AgentRunResult
    outcome: TaskOutcome | None = None
    state_before: StateSnapshot | None = None
    state_after: StateSnapshot | None = None
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """The result of one initial attempt or one recovery continuation."""

    task_id: str
    attempts: tuple[AttemptRecord, ...]
    success: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class AgentAdapter(Protocol):
    """Adapter for model-backed agents."""

    name: str

    def run(
        self,
        task: Task,
        prompt: str,
        environment: Any,
        context: AgentContext,
    ) -> AgentRunResult:
        """Run one attempt against the current benchmark environment."""


@runtime_checkable
class BenchmarkAdapter(Protocol):
    """Adapter for stateful benchmark environments."""

    name: str

    def load_task(self, task_id: str) -> Task:
        ...

    def reset(self, task: Task) -> StateSnapshot:
        ...

    def restore(self, snapshot: StateSnapshot) -> StateSnapshot:
        ...

    def snapshot(self, *, label: str | None = None) -> StateSnapshot:
        ...

    def agent_environment(self) -> Any:
        ...

    def evaluate(self, task: Task) -> TaskOutcome:
        ...

    def list_tasks(self) -> list[str]:
        ...
