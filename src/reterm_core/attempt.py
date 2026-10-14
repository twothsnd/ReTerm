"""Execute one initial or recovery attempt in a stateful task environment."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import AttemptAborted, InfrastructureError, TaskSkip, raise_if_fatal_api_error
from .prompts import make_recovery_prompt, make_task_prompt
from .types import (
    AgentAdapter,
    AgentContext,
    AgentRunResult,
    AttemptRecord,
    AttemptStatus,
    BenchmarkAdapter,
    BenchmarkResult,
    Task,
    TaskOutcome,
)


@dataclass(slots=True)
class AttemptRunner:
    benchmark: BenchmarkAdapter
    agent: AgentAdapter
    model_name: str

    def run_initial(self, task_id: str) -> BenchmarkResult:
        task = self.benchmark.load_task(task_id)
        self.benchmark.reset(task)
        attempt = self._execute(task, role="initial", previous_attempt=None)
        return self._result(task_id, (attempt,))

    def run_recovery(self, task_id: str, previous_attempt: AttemptRecord) -> BenchmarkResult:
        if previous_attempt.task_id != task_id or previous_attempt.state_after is None:
            raise ValueError("recovery requires a saved state from the same task")
        task = self.benchmark.load_task(task_id)
        self.benchmark.reset(task)
        self.benchmark.restore(previous_attempt.state_after)
        attempt = self._execute(task, role="recovery", previous_attempt=previous_attempt)
        return self._result(task_id, (previous_attempt, attempt))

    def _execute(
        self,
        task: Task,
        *,
        role: str,
        previous_attempt: AttemptRecord | None,
    ) -> AttemptRecord:
        is_recovery = previous_attempt is not None
        attempt_index = 2 if is_recovery else 1
        prompt = make_recovery_prompt(task) if is_recovery else make_task_prompt(task)
        state_before = self.benchmark.snapshot(label=f"attempt-{attempt_index}-before")
        context = AgentContext(
            benchmark=self.benchmark.name,
            task_id=task.task_id,
            attempt_index=attempt_index,
            state_before=state_before,
            previous_attempts=(previous_attempt,) if previous_attempt is not None else (),
        )
        try:
            agent_result = self.agent.run(task, prompt, self.benchmark.agent_environment(), context)
        except (TaskSkip, InfrastructureError, AttemptAborted):
            raise
        except Exception as exc:
            raise_if_fatal_api_error(exc)
            agent_result = AgentRunResult(error=f"{type(exc).__name__}: {exc}")

        state_after = None
        outcome = TaskOutcome(success=False, details={"skipped_evaluation": "agent_error"})
        if agent_result.error is None:
            # The verifier may change the environment. Recovery starts from
            # this snapshot, taken before evaluation.
            state_after = self.benchmark.snapshot(label=f"attempt-{attempt_index}-after")
            try:
                outcome = self.benchmark.evaluate(task)
            except (TaskSkip, InfrastructureError, AttemptAborted):
                raise
            except Exception as exc:
                raise_if_fatal_api_error(exc)
                outcome = TaskOutcome(
                    success=False,
                    details={"evaluation_error": f"{type(exc).__name__}: {exc}"},
                )
        status = (
            AttemptStatus.ERROR if agent_result.error is not None
            else AttemptStatus.SUCCESS if outcome.success
            else AttemptStatus.FAILED
        )
        return AttemptRecord(
            attempt_index=attempt_index,
            task_id=task.task_id,
            prompt=prompt,
            status=status,
            agent_result=agent_result,
            outcome=outcome,
            state_before=state_before,
            state_after=state_after,
            notes={"role": role, "state_after_timing": "after_agent_before_evaluate"},
        )

    def _result(
        self,
        task_id: str,
        attempts: tuple[AttemptRecord, ...],
    ) -> BenchmarkResult:
        return BenchmarkResult(
            task_id=task_id,
            attempts=attempts,
            success=bool(attempts[-1].outcome and attempts[-1].outcome.success),
            metadata={
                "benchmark": self.benchmark.name,
                "model": self.model_name,
                "agent": self.agent.name,
            },
        )
