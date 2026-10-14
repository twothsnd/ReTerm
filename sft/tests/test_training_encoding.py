from __future__ import annotations

import os
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from reterm_sft.train import (
    _normalize_messages_for_template,
    encode_record,
    has_target_format_error,
)


class DictReturningTokenizer:
    chat_template = (
        '{%- elif message.role == "assistant" %}'
        " assistant branch "
        '{%- elif message.role == "tool" %}'
    )

    def __call__(self, text, **kwargs):
        if text == "<|im_start|>assistant\n":
            return {"input_ids": [30]}
        if text == "<|im_start|>assistant\n<think>\n":
            return {"input_ids": [30, 31]}
        raise AssertionError(text)

    def apply_chat_template(self, messages, **kwargs):
        ids = []
        mask = []
        role_ids = {"system": 10, "user": 20, "assistant": 30, "tool": 40}
        for message in messages:
            ids.extend([role_ids[message["role"]], len(message.get("content", "")) + 100])
            enabled = int(message["role"] == "assistant")
            mask.extend([enabled, enabled])
        return {
            "input_ids": [ids],
            "attention_mask": [[1] * len(ids)],
            "assistant_masks": [mask],
        }


def test_encode_record_accepts_transformers_batch_encoding_shape() -> None:
    tokenizer = DictReturningTokenizer()
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": (
                "<think>failed reasoning</think>"
                "<tool_call><function=bash><parameter=command>false</parameter></function></tool_call>"
            ),
        },
        {"role": "user", "content": "<tool_response>failed observation</tool_response>"},
        {"role": "user", "content": "recovery prompt"},
        {
            "role": "assistant",
            "content": (
                "<think>recovery reasoning</think>"
                "<tool_call><function=done></function></tool_call>"
            ),
        },
    ]

    sample = encode_record(
        {"messages": messages, "train_from_message_index": 5},
        tokenizer,
    )

    assert len(sample["input_ids"]) == len(messages) * 2
    assert sample["labels"][:-1] == [-100] * (len(sample["labels"]) - 1)
    assert sample["labels"][-1] == sample["input_ids"][-1]


def test_training_uses_canonical_xml_instead_of_provider_transport_content() -> None:
    messages = [
        {
            "role": "assistant",
            "content": (
                "<think>canonical reasoning</think>"
                "<tool_call><function=bash><parameter=command>pwd</parameter></function></tool_call>"
            ),
            "transport_content": "provider-visible prose must not become a training target",
            "transport_reasoning_content": "provider reasoning",
            "transport_tool_calls": [
                {
                    "id": "provider-id",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command":"wrong"}'},
                }
            ],
        },
        {"role": "user", "content": "<tool_response>/home/user</tool_response>"},
    ]

    normalized = _normalize_messages_for_template(messages)

    assert normalized[0]["reasoning_content"] == "canonical reasoning"
    assert normalized[0]["tool_calls"][0]["function"]["arguments"] == {"command": "pwd"}
    assert "provider-visible prose" not in str(normalized)
    assert normalized[1]["tool_call_id"] == "reterm-sft-call-0"


def test_target_format_error_filter_ignores_weak_attempt_corrections() -> None:
    record = {
        "train_from_message_index": 3,
        "messages": [
            {"role": "assistant", "content": "malformed"},
            {"role": "user", "content": "Format error: weak correction"},
            {"role": "user", "content": "recovery prompt"},
            {"role": "assistant", "content": "<think>recover</think>"},
        ],
    }

    assert not has_target_format_error(record)


def test_training_keeps_rejected_weak_output_like_pro_without_shifting_target_index() -> None:
    messages = [
        {
            "role": "assistant",
            "content": (
                "<think>bad reasoning with an embedded tool call"
                "<tool_call><function=bash><parameter=command>false</parameter>"
                "</function></tool_call></think>"
            ),
        },
        {
            "role": "user",
            "content": "Format error: response did not contain exactly one tool call",
            "message_type": "format_correction",
        },
        {"role": "user", "content": "recovery prompt"},
        {"role": "assistant", "content": "<think>teacher recovery</think>"},
    ]

    normalized = _normalize_messages_for_template(messages, train_from_message_index=3)

    assert len(normalized) == len(messages)
    assert normalized[0] == {
        "role": "assistant",
        "content": "",
        "reasoning_content": (
            "bad reasoning with an embedded tool call"
            "<tool_call><function=bash><parameter=command>false</parameter>"
            "</function></tool_call>"
        ),
    }
    assert normalized[1] == {"role": "user", "content": messages[1]["content"]}
    assert normalized[3] == {
        "role": "assistant",
        "content": "",
        "reasoning_content": "teacher recovery",
    }


def test_target_format_error_filter_detects_legacy_and_new_recovery_corrections() -> None:
    legacy = {
        "train_from_message_index": 1,
        "messages": [
            {"role": "user", "content": "recovery prompt"},
            {"role": "assistant", "content": "malformed"},
            {"role": "user", "content": "Format error: expected one tool call"},
        ],
    }
    tagged = {
        "train_from_message_index": 1,
        "messages": [
            {"role": "user", "content": "recovery prompt"},
            {"role": "assistant", "content": "malformed"},
            {
                "role": "user",
                "content": "<tool_response>\nFormat error: expected one tool call\n</tool_response>",
                "message_type": "format_correction",
            },
        ],
    }

    assert has_target_format_error(legacy)
    assert has_target_format_error(tagged)


def test_training_removes_rollout_only_recovery_guidance() -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "failed action"},
        {
            "role": "user",
            "content": (
                "Your previous attempt failed.\n\n"
                "Original task:\nDo the task.\n\n"
                "Recovery tool-call constraint:\n- exactly one tool call\n\n"
                "Recovery-Bench recovery workflow:\n\n1. Inherited state audit.\n"
            ),
        },
        {"role": "assistant", "content": "<think>recover</think>"},
    ]

    normalized = _normalize_messages_for_template(
        messages,
        train_from_message_index=4,
    )
    rendered = str(normalized)

    assert "Your previous attempt failed." in rendered
    assert "Original task:\\nDo the task." in rendered
    assert "Recovery tool-call constraint:" not in rendered
    assert "Recovery-Bench recovery workflow:" not in rendered
    assert "Inherited state audit" not in rendered
    assert "Recovery tool-call constraint:" in messages[3]["content"]


def test_real_qwen_template_masks_only_recovery_assistants() -> None:
    model_path = Path(os.environ.get("RETERM_TEST_MODEL", ""))
    if not os.environ.get("RETERM_TEST_MODEL"):
        pytest.skip("set RETERM_TEST_MODEL to run the tokenizer integration test")
    if not model_path.exists():
        pytest.skip("local Qwen3.5 tokenizer is unavailable")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": (
                "<think>weak reasoning</think>"
                "<tool_call><function=bash><parameter=command>false</parameter></function></tool_call>"
            ),
        },
        {"role": "user", "content": "<tool_response>failed</tool_response>"},
        {"role": "user", "content": "recovery prompt"},
        {
            "role": "assistant",
            "content": (
                "<think>teacher reasoning</think>"
                "<tool_call><function=bash><parameter=command>true</parameter></function></tool_call>"
            ),
        },
        {"role": "user", "content": "<tool_response>ok</tool_response>"},
        {
            "role": "assistant",
            "content": "<think>finish</think><tool_call><function=done></function></tool_call>",
        },
    ]

    sample = encode_record({"messages": messages, "train_from_message_index": 5}, tokenizer)
    target = tokenizer.decode([token for token in sample["labels"] if token != -100])

    assert "weak reasoning" not in target
    assert "teacher reasoning" in target
    assert "<function=done>" in target
