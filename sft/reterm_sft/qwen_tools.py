"""Terminal tool schemas and Qwen message conversion for SFT."""

from __future__ import annotations

import re
from typing import Any

from reterm_core.qwen_tools import TERMINAL_TOOLS


def canonical_messages_to_tool_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Render stored XML actions as matched Qwen assistant/tool messages."""
    converted: list[dict[str, Any]] = []
    pending_call_id: str | None = None
    for index, message in enumerate(messages):
        role = message.get("role")
        raw_content = message.get("content", "")
        content = raw_content if isinstance(raw_content, str) else ""
        if role == "assistant":
            following_tool_output = _following_tool_output(messages, index)
            reasoning, action_text = split_qwen_assistant_content(content)
            call_id = f"reterm-sft-call-{index}"
            call = qwen_xml_to_tool_call(action_text, call_id=call_id)
            if following_tool_output is not None and call is not None:
                item: dict[str, Any] = {"role": "assistant", "content": "", "tool_calls": [call]}
                if reasoning is not None:
                    item["reasoning_content"] = reasoning
                converted.append(item)
                pending_call_id = call_id
                continue

            # Unexecuted actions stay in assistant history; only actual tool
            # calls with observations become tool messages.
            item = {
                "role": "assistant",
                "content": action_text if reasoning is not None else content,
            }
            if reasoning is not None:
                item["reasoning_content"] = reasoning
            converted.append(item)
            pending_call_id = None
            continue
        if role == "user" and pending_call_id is not None:
            tool_output = unwrap_tool_response(content)
            if tool_output is not None:
                converted.append({"role": "tool", "content": tool_output, "tool_call_id": pending_call_id})
                pending_call_id = None
                continue
        if role in {"system", "user", "assistant"}:
            converted.append({"role": str(role), "content": content})
            if role != "assistant":
                pending_call_id = None
    return converted


def _following_tool_output(messages: list[dict[str, Any]], index: int) -> str | None:
    if index + 1 >= len(messages):
        return None
    following = messages[index + 1]
    if following.get("role") != "user" or following.get("message_type") == "format_correction":
        return None
    raw_content = following.get("content", "")
    if not isinstance(raw_content, str):
        return None
    return unwrap_tool_response(raw_content)


def qwen_xml_to_tool_call(
    text: str,
    *,
    call_id: str,
) -> dict[str, Any] | None:
    bodies = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, flags=re.IGNORECASE | re.DOTALL)
    if len(bodies) != 1:
        return None
    functions = re.findall(
        r"<function=([^>\s]+)>\s*(.*?)\s*</function>",
        bodies[0],
        flags=re.IGNORECASE | re.DOTALL,
    )
    if len(functions) != 1:
        return None
    name, body = functions[0]
    arguments = {
        key.strip(): value.strip()
        for key, value in re.findall(
            r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>",
            body,
            flags=re.IGNORECASE | re.DOTALL,
        )
    }
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name.strip(), "arguments": arguments},
    }


def split_qwen_assistant_content(content: str) -> tuple[str | None, str]:
    match = re.match(
        r"^\s*<think>\s*(.*?)\s*</think>\s*(.*)$",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None, content
    return match.group(1), match.group(2)


def unwrap_tool_response(content: str) -> str | None:
    match = re.fullmatch(
        r"\s*<tool_response>\s*(.*?)\s*</tool_response>\s*",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match.group(1) if match else None
