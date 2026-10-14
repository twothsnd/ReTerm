"""Terminal tool schemas shared by SFT and ReTerm-GRPO."""

from __future__ import annotations

from typing import Any


TERMINAL_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute exactly one non-interactive shell command in the task environment.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Submit the current task state for official evaluation.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
]
