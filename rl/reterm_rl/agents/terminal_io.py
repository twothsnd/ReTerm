"""Terminal command results and bounded observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class CommandResult:
    success: bool
    output: str
    exit_code: int | None
    timed_out: bool = False
    method: str | None = None
    reward: int | None = None


class TerminalEnvironment(Protocol):
    def exec(self, command: str, timeout: float | None = None) -> CommandResult: ...


def limit_output_length(output: str, max_bytes: int) -> tuple[str, dict[str, int | bool]]:
    original_bytes = len(output.encode("utf-8"))
    if max_bytes <= 0 or original_bytes <= max_bytes:
        return output, {
            "limited": False,
            "original_bytes": original_bytes,
            "model_bytes": original_bytes,
            "omitted_bytes": 0,
        }
    portion_size = max(1, max_bytes // 2)
    output_bytes = output.encode("utf-8")
    first = output_bytes[:portion_size].decode("utf-8", errors="ignore")
    last = output_bytes[-portion_size:].decode("utf-8", errors="ignore")
    omitted = original_bytes - len(first.encode("utf-8")) - len(last.encode("utf-8"))
    limited = f"{first}\n[... output limited to {max_bytes} bytes; {omitted} interior bytes omitted ...]\n{last}"
    return limited, {
        "limited": True,
        "original_bytes": original_bytes,
        "model_bytes": len(limited.encode("utf-8")),
        "omitted_bytes": omitted,
    }
