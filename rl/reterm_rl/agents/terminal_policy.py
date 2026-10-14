"""Terminal agent adapter for policy-driven bash-tool attempts."""

from __future__ import annotations

import copy
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from reterm_rl.bootstrap import ensure_local_paths

ensure_local_paths()

from reterm_core.errors import AttemptAborted  # noqa: E402
from reterm_core.types import (  # noqa: E402
    ActionRecord,
    AgentContext,
    AgentRunResult,
    AttemptRecord,
    Task,
)
from reterm_rl.agents.terminal_io import CommandResult, TerminalEnvironment, limit_output_length  # noqa: E402


SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

QWEN_XML_TOOL_SYSTEM_PROMPT = """You are a helpful assistant that can interact with a Linux terminal.

Goal: complete the user's terminal task by using the available tools.

Important rules:
- Use the bash tool to execute exactly one non-interactive shell command at a time.
- Directory and environment-variable changes are persistent across turns in this harness.
- For long-running commands, wrap with timeout, e.g. timeout 30 <command>.
- Output may be truncated. Use head, tail, grep, find filters, or sed -n to inspect smaller regions.
- When you are confident the task is complete, call the done tool.
- After submitting done, you cannot continue working on the task.
"""

OBSERVATION_TOO_LONG_HINT = """The output of your last command was too long.
Please try a different command that produces less output. Use head, tail,
grep, find filters, or sed -n to view smaller regions selectively."""


class MessagePolicy(Protocol):
    def __call__(self, messages: list[dict[str, Any]], context: AgentContext) -> str | "PolicyResponse":
        ...


@dataclass
class PolicyResponse:
    content: str
    token_ids: list[int] | None = None
    log_probs: list[float] | None = None
    metadata: dict[str, Any] | None = None


class RolloutTimeoutError(TimeoutError):
    """A rollout deadline expired while the policy was generating or acting."""


class RolloutCancelledError(AttemptAborted):
    """A speculative rollout was cancelled after the train queue became full."""


@dataclass
class BashAction:
    command: str
    source: str
    tool_call_id: str | None = None
    missing_thought: bool = False

    @property
    def done(self) -> bool:
        return SUBMIT_MARKER in self.command


@dataclass
class TerminalPolicyAgentConfig:
    """Per-attempt terminal agent budget and formatting controls."""

    max_turns_per_attempt: int = 8
    max_observation_bytes: int = 50000
    command_timeout: float | None = None
    max_format_errors: int = 8
    max_wall_time_seconds: float | None = None


class TerminalPolicyAgent:
    """Run one policy attempt in an interactive terminal environment."""

    name = "terminal-policy-agent"

    def __init__(
        self,
        policy: MessagePolicy | Callable[[list[dict[str, Any]], AgentContext], str | PolicyResponse],
        config: TerminalPolicyAgentConfig | None = None,
        deadline_setter: Callable[[float | None], None] | None = None,
        cancel_event: Any | None = None,
    ):
        self.policy = policy
        self.config = config or TerminalPolicyAgentConfig()
        self.deadline_setter = deadline_setter
        self.cancel_event = cancel_event

    def _raise_if_cancelled(self) -> None:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise RolloutCancelledError("speculative rollout cancelled after train queue filled")

    def run(
        self,
        task: Task,
        prompt: str,
        environment: TerminalEnvironment,
        context: AgentContext,
    ) -> AgentRunResult:
        deadline = (
            time.monotonic() + float(self.config.max_wall_time_seconds)
            if self.config.max_wall_time_seconds is not None and self.config.max_wall_time_seconds > 0
            else None
        )
        if self.deadline_setter is not None:
            self.deadline_setter(deadline)
        try:
            return self._run_with_deadline(task, prompt, environment, context, deadline)
        finally:
            if self.deadline_setter is not None:
                self.deadline_setter(None)

    def _run_with_deadline(
        self,
        task: Task,
        prompt: str,
        environment: TerminalEnvironment,
        context: AgentContext,
        deadline: float | None,
    ) -> AgentRunResult:
        messages = self._initial_messages(task, prompt, context)
        current_attempt_start_index = len(messages)
        actions: list[ActionRecord] = []
        turns: list[dict[str, Any]] = []
        format_errors = 0

        stopped_by_context_limit = False
        context_limit_error: str | None = None
        stopped_by_wall_time = False
        wall_time_error: str | None = None

        for turn_index in range(1, self.config.max_turns_per_attempt + 1):
            self._raise_if_cancelled()
            remaining_time = self._remaining_wall_time(deadline)
            if remaining_time is not None and remaining_time <= 0:
                stopped_by_wall_time = True
                wall_time_error = f"rollout exceeded wall-clock limit of {self.config.max_wall_time_seconds}s"
                turns.append(
                    {
                        "turn_index": turn_index,
                        "harness": "bash-tool",
                        "stopped_by_wall_time": True,
                        "error": wall_time_error,
                    }
                )
                break
            try:
                policy_response = self.policy(messages, context)
            except Exception as exc:
                if self._is_wall_time_error(exc):
                    stopped_by_wall_time = True
                    wall_time_error = f"{type(exc).__name__}: {exc}"
                    turns.append(
                        {
                            "turn_index": turn_index,
                            "harness": "bash-tool",
                            "stopped_by_wall_time": True,
                            "error": wall_time_error,
                        }
                    )
                    break
                if not self._is_context_limit_error(exc):
                    raise
                stopped_by_context_limit = True
                context_limit_error = f"{type(exc).__name__}: {exc}"
                turns.append(
                    {
                        "turn_index": turn_index,
                        "harness": "bash-tool",
                        "stopped_by_context_limit": True,
                        "error": context_limit_error,
                    }
                )
                break
            raw_response, assistant_meta = self._coerce_policy_response(policy_response)
            self._raise_if_cancelled()
            assistant_message: dict[str, Any] = {"role": "assistant", "content": raw_response}
            assistant_message.update(assistant_meta)
            messages.append(assistant_message)
            turn_record: dict[str, Any] = {
                "turn_index": turn_index,
                "raw_response": raw_response,
                "harness": "bash-tool",
            }

            try:
                action = self._parse_bash_action(raw_response)
            except ValueError as exc:
                assistant_message["native_message"] = self._qwen_native_assistant_message(raw_response)
                format_errors += 1
                observation = self._format_error_observation(str(exc))
                correction_message: dict[str, Any] = {
                    "role": "user",
                    "content": observation,
                    "message_type": "format_correction",
                }
                correction_message["native_message"] = {"role": "user", "content": observation}
                messages.append(correction_message)
                turn_record["format_error"] = str(exc)
                turn_record["observation"] = observation
                turns.append(turn_record)
                if format_errors >= self.config.max_format_errors:
                    break
                continue

            format_errors = 0
            action.tool_call_id = action.tool_call_id or (
                f"reterm-call-{int(context.attempt_index)}-{turn_index}"
            )
            assistant_message["native_message"] = self._qwen_native_assistant_message(
                raw_response,
                action=action,
            )
            command = action.command.strip()
            turn_record.update(
                {
                    "command": command,
                    "action_source": action.source,
                    "missing_thought": action.missing_thought,
                }
            )

            command_timeout = self._bounded_timeout(self.config.command_timeout, deadline)
            command_start = time.monotonic()
            result = environment.exec(command + ("\n" if not command.endswith("\n") else ""), timeout=command_timeout)
            command_elapsed_seconds = time.monotonic() - command_start
            output, output_meta = limit_output_length(result.output, self.config.max_observation_bytes)
            if output_meta.get("limited"):
                output = f"{output}\n\n{OBSERVATION_TOO_LONG_HINT}"

            turn_record.update(
                {
                    "exit_code": result.exit_code,
                    "timed_out": result.timed_out,
                    "command_elapsed_seconds": command_elapsed_seconds,
                    "success": result.success,
                    "output": output,
                    "output_limit": output_meta,
                }
            )
            actions.append(
                ActionRecord(
                    action=command,
                    observation=output,
                    metadata={
                        "turn_index": turn_index,
                        "exit_code": result.exit_code,
                        "timed_out": result.timed_out,
                        "command_elapsed_seconds": command_elapsed_seconds,
                        "harness": "bash-tool",
                        "done": action.done,
                    },
                )
            )

            observation = self._format_tool_result(command, result, output)
            native_tool_content = observation
            observation = f"<tool_response>\n{observation}\n</tool_response>"
            observation_message: dict[str, Any] = {"role": "user", "content": observation}
            observation_message["native_message"] = {
                "role": "tool",
                "content": native_tool_content,
                "tool_call_id": action.tool_call_id,
            }
            messages.append(observation_message)
            turn_record["observation"] = observation
            turns.append(turn_record)
            if action.done:
                break

        return AgentRunResult(
            actions=tuple(actions),
            metadata={
                "messages": messages,
                "current_attempt_start_index": current_attempt_start_index,
                "turns": turns,
                "harness": "bash-tool",
                "submit_marker": SUBMIT_MARKER,
                "max_turns_per_attempt": self.config.max_turns_per_attempt,
                "stopped_by_context_limit": stopped_by_context_limit,
                "context_limit_error": context_limit_error,
                "stopped_by_wall_time": stopped_by_wall_time,
                "wall_time_error": wall_time_error,
                "max_wall_time_seconds": self.config.max_wall_time_seconds,
            },
        )

    def _qwen_native_assistant_message(
        self,
        raw_response: str,
        *,
        action: BashAction | None = None,
    ) -> dict[str, Any]:
        reasoning, visible = self._split_qwen_sampled_response(raw_response)
        message: dict[str, Any] = {
            "role": "assistant",
            "content": "" if action is not None else visible,
            "reasoning_content": reasoning,
        }
        if action is not None:
            if not action.tool_call_id:
                raise ValueError("native Qwen tool history requires a tool_call_id")
            name = "done" if action.done else "bash"
            arguments = {} if action.done else {"command": action.command.strip()}
            message["tool_calls"] = [
                {
                    "id": action.tool_call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            ]
        return message

    @staticmethod
    def _split_qwen_sampled_response(raw_response: str) -> tuple[str, str]:
        """Split a Qwen response sampled after the template's opening ``<think>``.

        vLLM response IDs normally start inside the think block because the
        official generation prompt already contains ``<think>\n``.  Accept a
        redundant explicit opening tag as well for serving-wrapper parity.
        """

        text = raw_response.strip()
        explicit = re.match(
            r"^<think>\s*(.*?)\s*</think>\s*(.*)$",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if explicit:
            return explicit.group(1).strip(), explicit.group(2).strip()
        if "</think>" in text.lower():
            reasoning, visible = re.split(r"</think>", text, maxsplit=1, flags=re.IGNORECASE)
            return reasoning.strip(), visible.strip()
        # Without a closing tag, the whole sampled suffix is still reasoning;
        # a tool-shaped string here must never become an executed tool call.
        return text, ""

    def _initial_messages(self, task: Task, prompt: str, context: AgentContext) -> list[dict[str, Any]]:
        if context.previous_attempts:
            messages = self._latest_attempt_messages(context.previous_attempts)
            messages.append({"role": "user", "content": prompt if prompt else task.prompt})
            return messages

        task_text = prompt.strip() or task.prompt.strip()
        return [
            {"role": "system", "content": QWEN_XML_TOOL_SYSTEM_PROMPT.strip()},
            {"role": "user", "content": task_text},
        ]

    @staticmethod
    def _latest_attempt_messages(previous_attempts: tuple[AttemptRecord, ...]) -> list[dict[str, Any]]:
        raw_messages = previous_attempts[-1].agent_result.metadata.get("messages")
        if not isinstance(raw_messages, list) or not raw_messages:
            raise RuntimeError("Qwen-XML recovery requires the previous attempt's complete message history")
        if not all(isinstance(message, dict) for message in raw_messages):
            raise RuntimeError("Qwen-XML recovery message history contains a non-message entry")
        return copy.deepcopy(raw_messages)

    def _coerce_policy_response(self, response: str | PolicyResponse) -> tuple[str, dict[str, Any]]:
        if isinstance(response, PolicyResponse):
            meta: dict[str, Any] = dict(response.metadata or {})
            if response.token_ids is not None:
                meta["token_ids"] = list(response.token_ids)
            if response.log_probs is not None:
                meta["log_probs"] = [float(value) for value in response.log_probs]
            return response.content, meta
        return response, {}

    def _format_error_observation(self, error: str) -> str:
        return (
            f"Format error: {error}\n\n"
            "Please call exactly one available tool. Use bash for one shell command, "
            "or done when the task is complete."
        )

    def _is_context_limit_error(self, exc: Exception) -> bool:
        text = str(exc).lower()
        if "context_length_exceeded" in text or "context window exceeded" in text:
            return True
        return "maximum context length" in text and any(
            marker in text
            for marker in (
                "prompt length",
                "prompt contains",
                "input token",
                "requested tokens",
            )
        )

    def _is_wall_time_error(self, exc: Exception) -> bool:
        return isinstance(exc, RolloutTimeoutError) or "rollout exceeded wall-clock limit" in str(exc)

    @staticmethod
    def _remaining_wall_time(deadline: float | None) -> float | None:
        if deadline is None:
            return None
        return deadline - time.monotonic()

    def _bounded_timeout(self, configured_timeout: float | None, deadline: float | None) -> float | None:
        remaining = self._remaining_wall_time(deadline)
        if remaining is None:
            return configured_timeout
        remaining = max(1.0, remaining)
        if configured_timeout is None:
            return remaining
        return max(1.0, min(float(configured_timeout), remaining))

    def _format_tool_result(self, command: str, result: CommandResult, output: str) -> str:
        return (
            f"OBSERVATION:\n"
            f"$ {command}\n"
            f"exit_code={result.exit_code} timed_out={result.timed_out}\n"
            f"{output}"
        ).rstrip()

    def _parse_bash_action(self, response: str) -> BashAction:
        text = response.strip()
        if not text:
            raise ValueError("empty response")

        action = self._parse_qwen_xml_tool_action(text)
        if action is not None:
            return action
        raise ValueError("response did not contain exactly one bash or done tool call")

    def _parse_qwen_xml_tool_action(self, text: str) -> BashAction | None:
        # Match ReTerm SFT: a tool-shaped string inside hidden reasoning is
        # not an executable tool call. Only parse the payload after the single
        # leading, complete think block.
        think_matches = re.findall(r"<think>\s*(.*?)\s*</think>", text, flags=re.IGNORECASE | re.DOTALL)
        if len(think_matches) > 1:
            raise ValueError(f"expected at most one complete think block, found {len(think_matches)}")
        reasoning, action_text = self._split_qwen_sampled_response(text)
        has_thought = bool(reasoning)
        calls = self._extract_qwen_xml_tool_calls(action_text)
        if not calls:
            return None
        if len(calls) != 1:
            raise ValueError(f"expected exactly one tool call, found {len(calls)}")
        name, params = calls[0]
        if name == "done":
            return BashAction(
                command=f"echo {SUBMIT_MARKER}",
                source="qwen_xml_done",
                missing_thought=not has_thought,
            )
        if name != "bash":
            raise ValueError(f"unsupported tool call {name!r}; expected bash or done")
        command = params.get("command", "").strip()
        if not command:
            raise ValueError("bash tool call is missing non-empty command parameter")
        return BashAction(command=command, source="qwen_xml_bash", missing_thought=not has_thought)

    @staticmethod
    def _extract_qwen_xml_tool_calls(text: str) -> list[tuple[str, dict[str, str]]]:
        tool_bodies = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, flags=re.IGNORECASE | re.DOTALL)
        calls: list[tuple[str, dict[str, str]]] = []
        for body in tool_bodies:
            function_matches = re.findall(
                r"<function=([^>\s]+)>\s*(.*?)\s*</function>",
                body,
                flags=re.IGNORECASE | re.DOTALL,
            )
            for raw_name, function_body in function_matches:
                params: dict[str, str] = {}
                for raw_param, raw_value in re.findall(
                    r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>",
                    function_body,
                    flags=re.IGNORECASE | re.DOTALL,
                ):
                    params[raw_param.strip()] = raw_value.strip()
                calls.append((raw_name.strip(), params))
        return calls
