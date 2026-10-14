"""State and history must carry across one failed attempt and its recovery."""

from __future__ import annotations

import pytest

from reterm_core.attempt import AttemptRunner
from reterm_core.errors import AttemptAborted, InfrastructureError
from reterm_core.types import AgentRunResult, StateSnapshot, Task, TaskOutcome


class TaskEnvironment:
    name = "example"

    def __init__(self) -> None:
        self.state: set[str] = set()
        self.events: list[str] = []

    def load_task(self, task_id: str) -> Task:
        return Task(task_id, "Complete the task")

    def reset(self, task: Task) -> StateSnapshot:
        self.events.append("reset")
        self.state = set()
        return StateSnapshot(payload=set())

    def restore(self, snapshot: StateSnapshot) -> StateSnapshot:
        self.events.append("restore")
        self.state = set(snapshot.payload)
        return snapshot

    def snapshot(self, *, label: str | None = None) -> StateSnapshot:
        self.events.append(f"snapshot:{label}")
        return StateSnapshot(payload=set(self.state), label=label)

    def agent_environment(self) -> TaskEnvironment:
        return self

    def evaluate(self, task: Task) -> TaskOutcome:
        self.events.append("evaluate")
        success = "repaired" in self.state
        self.state.add("verifier-side-effect")
        return TaskOutcome(success=success)


class Agent:
    name = "example-agent"

    def run(self, task, prompt, environment, context) -> AgentRunResult:
        if context.previous_attempts:
            assert context.attempt_index == 2
            assert context.previous_attempts[0].state_after.payload == {"useful-work"}
            assert environment.state == {"useful-work"}
            assert "previous attempt failed" in prompt
            environment.state.add("repaired")
        else:
            assert context.attempt_index == 1
            assert "Original task" in prompt
            environment.state.add("useful-work")
        return AgentRunResult(metadata={"messages": [{"role": "assistant", "content": "done"}]})


def test_initial_failure_and_recovery_preserve_pre_verifier_state() -> None:
    environment = TaskEnvironment()
    runner = AttemptRunner(environment, Agent(), "example-model")

    initial = runner.run_initial("task")
    assert not initial.success
    failed_attempt = initial.attempts[0]
    assert failed_attempt.state_after.payload == {"useful-work"}
    assert environment.state == {"useful-work", "verifier-side-effect"}

    recovery = runner.run_recovery("task", failed_attempt)
    assert recovery.success
    assert recovery.attempts == (failed_attempt, recovery.attempts[1])
    assert recovery.attempts[1].state_after.payload == {"useful-work", "repaired"}
    assert environment.events == [
        "reset", "snapshot:attempt-1-before", "snapshot:attempt-1-after", "evaluate",
        "reset", "restore", "snapshot:attempt-2-before", "snapshot:attempt-2-after", "evaluate",
    ]


@pytest.mark.parametrize("error_type", [InfrastructureError, AttemptAborted])
def test_agent_operational_failures_are_not_scored_as_task_failures(error_type: type[Exception]) -> None:
    class FailingAgent(Agent):
        def run(self, task, prompt, environment, context) -> AgentRunResult:
            raise error_type("agent unavailable")

    environment = TaskEnvironment()
    with pytest.raises(error_type):
        AttemptRunner(environment, FailingAgent(), "model").run_initial("task")
    assert "evaluate" not in environment.events


def test_verifier_infrastructure_failure_is_not_scored_as_task_failure() -> None:
    class FailingVerifier(TaskEnvironment):
        def evaluate(self, task: Task) -> TaskOutcome:
            raise InfrastructureError("verifier unavailable")

    with pytest.raises(InfrastructureError):
        AttemptRunner(FailingVerifier(), Agent(), "model").run_initial("task")
