from __future__ import annotations

import re
from typing import Any


class FatalRunError(RuntimeError):
    """Fatal condition that should stop the whole benchmark run."""


class InfrastructureError(RuntimeError):
    """Environment or verifier failure that is not a policy outcome."""


class AttemptAborted(RuntimeError):
    """An attempt stopped before it could produce a policy outcome."""


class TaskSkip(RuntimeError):
    """Recoverable task-level condition that should not stop the full run."""

    def __init__(self, task_id: str, reason: str, *, details: dict[str, Any] | None = None):
        super().__init__(reason)
        self.task_id = task_id
        self.reason = reason
        self.details = details or {}


FATAL_STATUS_CODES = {401, 403}
FATAL_MESSAGE_MARKERS = (
    "insufficient_quota",
    "insufficient quota",
    "quota exceeded",
    "quota_exceeded",
    "credit balance",
    "credits exhausted",
    "billing",
    "out of credits",
    "permission denied",
    "access denied",
    "insufficient permissions",
)


def fatal_api_error_message(exc: BaseException) -> str | None:
    status_code = _status_code(exc)
    message = str(exc)
    lowered = message.lower()
    message_status = _status_from_message(message)
    if status_code in FATAL_STATUS_CODES:
        return f"fatal API status {status_code}: {message}"
    if message_status in FATAL_STATUS_CODES:
        return f"fatal API status {message_status}: {message}"
    if any(marker in lowered for marker in FATAL_MESSAGE_MARKERS):
        return f"fatal API quota/permission error: {message}"
    return None


def raise_if_fatal_api_error(exc: BaseException) -> None:
    if isinstance(exc, FatalRunError):
        raise exc
    message = fatal_api_error_message(exc)
    if message is not None:
        raise FatalRunError(message) from exc


def _status_code(exc: BaseException) -> int | None:
    for attr in ("status_code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            pass
    response = getattr(exc, "response", None)
    if response is not None:
        value = getattr(response, "status_code", None)
        if isinstance(value, int):
            return value
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            pass
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        return _status_from_mapping(body)
    return None


def _status_from_mapping(mapping: dict[str, Any]) -> int | None:
    for key in ("status_code", "status"):
        value = mapping.get(key)
        if isinstance(value, int):
            return value
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            pass
    error = mapping.get("error")
    if isinstance(error, dict):
        return _status_from_mapping(error)
    return None


def _status_from_message(message: str) -> int | None:
    lowered = message.lower()
    for match in re.finditer(r"\b(401|403)\b", lowered):
        start = max(0, match.start() - 32)
        end = min(len(lowered), match.end() + 32)
        context = lowered[start:end]
        if any(marker in context for marker in ("status", "code", "http", "error", "forbidden", "unauthorized")):
            return int(match.group(1))
    return None
