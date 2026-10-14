"""Task instructions for the two ReTerm-GRPO attempt types."""

from __future__ import annotations

from .types import Task


def make_task_prompt(task: Task) -> str:
    return f"Complete the original task.\n\nOriginal task:\n{task.prompt.strip()}\n"


def make_recovery_prompt(task: Task) -> str:
    return (
        "Your previous attempt failed.\n\n"
        "The environment has NOT been reset.\n"
        "You are continuing from the current state left by your previous attempt.\n\n"
        "Some previous actions may have changed the environment.\n"
        "Repair any incorrect changes if needed.\n"
        "Preserve any useful progress.\n"
        "Complete the original task.\n\n"
        f"Original task:\n{task.prompt.strip()}\n"
    )
