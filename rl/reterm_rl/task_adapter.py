"""Load a task environment supplied by the training application."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Protocol

from reterm_core.errors import InfrastructureError
from reterm_core.types import BenchmarkAdapter, StateSnapshot


class RecoveryTaskAdapter(BenchmarkAdapter, Protocol):
    """Task environment interface used by ReTerm-GRPO rollouts."""

    rollout_cancel_event: Any

    def detach_snapshot(self, snapshot: StateSnapshot) -> None: ...

    def cancel_active_environment(self) -> None: ...

    def close(self) -> None: ...


class TaskInfrastructureError(InfrastructureError):
    """A task adapter failure that should be excluded from policy rewards."""

    def __init__(self, message: str, *, category: str = "environment") -> None:
        super().__init__(message)
        self.category = category


def create_task_dataset(config: dict[str, Any]) -> RecoveryTaskAdapter:
    """Instantiate a task dataset from ``module:factory`` and its settings.

    The factory receives a dictionary of task-specific settings. It returns a
    dataset compatible with ``reterm_core.attempt.AttemptRunner``.
    """

    settings = dict(config)
    reference = settings.pop("factory", None)
    if not isinstance(reference, str) or ":" not in reference:
        raise ValueError("recovery_env.dataset.factory must be module:factory")
    module_name, function_name = reference.rsplit(":", 1)
    if not module_name or not function_name:
        raise ValueError("recovery_env.dataset.factory must be module:factory")
    factory = getattr(import_module(module_name), function_name)
    return factory(settings)
