"""Supervised training on ReTerm-10K."""

from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers.utils.import_utils import is_causal_conv1d_available, is_flash_linear_attention_available

from reterm_sft.qwen_tools import TERMINAL_TOOLS, canonical_messages_to_tool_messages

class SFTDataset(Dataset):
    def __init__(
        self,
        path: Path,
        tokenizer: Any,
        max_length: int,
        *,
        over_length_policy: str = "error",
        exclude_target_format_errors: bool = False,
    ):
        self.samples: list[dict[str, list[int]]] = []
        self.total_records = 0
        self.excluded_target_format_errors = 0
        self.lengths: list[int] = []
        self.over_length: list[tuple[str, int]] = []
        for line_number, line in enumerate(_physical_lines(path), start=1):
            if not line.strip():
                continue
            self.total_records += 1
            record = json.loads(line)
            if exclude_target_format_errors and has_target_format_error(record):
                self.excluded_target_format_errors += 1
                continue
            sample = encode_record(record, tokenizer)
            length = len(sample["input_ids"])
            self.lengths.append(length)
            if length > max_length:
                record_id = str(record.get("id") or record.get("task_id") or f"line-{line_number}")
                self.over_length.append((record_id, length))
                continue
            self.samples.append(sample)
        if self.over_length and over_length_policy == "error":
            preview = ", ".join(f"{record_id}={length}" for record_id, length in self.over_length[:10])
            raise ValueError(
                f"{len(self.over_length)} records in {path} exceed max_length={max_length}: {preview}. "
                "Increase --max-length or explicitly use --over-length-policy drop."
            )
        if not self.samples:
            raise ValueError(f"no trainable samples in {path}; over_length={len(self.over_length)}")

    def summary(self) -> dict[str, int]:
        return {
            "records": self.total_records,
            "excluded_target_format_errors": self.excluded_target_format_errors,
            "kept": len(self.samples),
            "over_length": len(self.over_length),
            "min_tokens": min(self.lengths),
            "max_tokens": max(self.lengths),
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.samples[index]


def _physical_lines(path: Path):
    with path.open(encoding="utf-8") as source:
        yield from source


def has_target_format_error(record: dict[str, Any]) -> bool:
    """Return whether the supervised recovery segment contains parser feedback."""
    messages = record.get("messages")
    if not isinstance(messages, list):
        return False
    train_from = int(record.get("train_from_message_index", len(messages)))
    for message in messages[train_from:]:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        if message.get("message_type") == "format_correction":
            return True
        content = message.get("content")
        if not isinstance(content, str):
            continue
        stripped = content.lstrip()
        if stripped.startswith("Format error:"):
            return True
        if stripped.startswith("<tool_response>") and "\nFormat error:" in stripped:
            return True
    return False


def encode_record(record: dict[str, Any], tokenizer: Any) -> dict[str, list[int]]:
    train_from = int(record["train_from_message_index"])
    messages = _normalize_messages_for_template(
        record["messages"],
        train_from_message_index=train_from,
    )
    if not 0 <= train_from < len(messages):
        raise ValueError(f"train_from_message_index={train_from} is outside {len(messages)} messages")
    input_ids, assistant_mask = _chat_template_ids_and_assistant_mask(tokenizer, messages)
    labels = [-100] * len(input_ids)
    assistant_indexes = [index for index, message in enumerate(messages) if message.get("role") == "assistant"]
    segments = _true_segments(assistant_mask)
    if len(segments) != len(assistant_indexes):
        raise ValueError(
            f"assistant mask produced {len(segments)} spans for {len(assistant_indexes)} assistant messages"
        )
    role_prefix = _encode_plain(tokenizer, "<|im_start|>assistant\n")
    thinking_prefix = _encode_plain(tokenizer, "<|im_start|>assistant\n<think>\n")
    for message_index, (start, end) in zip(assistant_indexes, segments, strict=True):
        if message_index < train_from:
            continue
        response_start = _response_start(input_ids, start, end, role_prefix, thinking_prefix)
        labels[response_start:end] = input_ids[response_start:end]
    if all(label == -100 for label in labels):
        raise ValueError("record has no target assistant tokens")
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def _chat_template_ids_and_assistant_mask(
    tokenizer: Any,
    messages: list[dict[str, Any]],
) -> tuple[list[int], list[int]]:
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template:
        raise ValueError("tokenizer does not expose a string chat_template")
    encoded = tokenizer.apply_chat_template(
        messages,
        tools=TERMINAL_TOOLS,
        chat_template=_assistant_mask_template(template),
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    input_ids = _flatten_token_ids(encoded["input_ids"])
    raw_mask = encoded.get("assistant_masks")
    if raw_mask is None:
        raise ValueError("tokenizer did not return assistant_masks")
    assistant_mask = _flatten_token_ids(raw_mask)
    if len(input_ids) != len(assistant_mask):
        raise ValueError(
            f"input_ids and assistant_masks differ in length: {len(input_ids)} != {len(assistant_mask)}"
        )
    return input_ids, assistant_mask


@lru_cache(maxsize=4)
def _assistant_mask_template(template: str) -> str:
    assistant_branch = '{%- elif message.role == "assistant" %}'
    tool_branch = '{%- elif message.role == "tool" %}'
    if template.count(assistant_branch) != 1 or template.count(tool_branch) != 1:
        raise ValueError("unsupported Qwen chat template: assistant/tool branches are not unique")
    return template.replace(
        assistant_branch,
        assistant_branch + "\n{% generation %}",
        1,
    ).replace(
        tool_branch,
        "{% endgeneration %}\n" + tool_branch,
        1,
    )


def _true_segments(mask: list[int]) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for index, enabled in enumerate([*mask, 0]):
        if enabled and start is None:
            start = index
        elif not enabled and start is not None:
            segments.append((start, index))
            start = None
    return segments


def _response_start(
    input_ids: list[int],
    start: int,
    end: int,
    role_prefix: list[int],
    thinking_prefix: list[int],
) -> int:
    segment = input_ids[start:end]
    for prefix in (thinking_prefix, role_prefix):
        if segment[: len(prefix)] == prefix:
            return start + len(prefix)
    raise ValueError("assistant span does not start with the expected Qwen assistant prefix")


def _encode_plain(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    if isinstance(encoded, dict) or (hasattr(encoded, "__contains__") and "input_ids" in encoded):
        encoded = encoded["input_ids"]
    return _flatten_token_ids(encoded)


def _flatten_token_ids(encoded: Any) -> list[int]:
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError(f"unexpected batched token output with batch size {len(encoded)}")
        encoded = encoded[0]
    return [int(token_id) for token_id in encoded]


def _normalize_messages_for_template(
    messages: list[dict[str, Any]],
    *,
    train_from_message_index: int | None = None,
) -> list[dict[str, Any]]:
    # Training follows the canonical Qwen XML trajectory. Provider-native
    # transport fields exist only for exact API replay and must not alter the
    # supervised target rendered by Qwen's official chat template.
    if train_from_message_index is not None:
        messages = _remove_rollout_only_recovery_guidance(messages, train_from_message_index)
    return canonical_messages_to_tool_messages(messages)


def _remove_rollout_only_recovery_guidance(
    messages: list[dict[str, Any]],
    train_from_message_index: int,
) -> list[dict[str, Any]]:
    """Remove Teacher rollout guidance from the student's recovery context."""
    copied = [dict(message) for message in messages]
    upper_bound = min(max(train_from_message_index, 0), len(copied))
    for index in range(upper_bound - 1, -1, -1):
        message = copied[index]
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            break
        marker_positions = [
            position
            for marker in (
                "\n\nRecovery tool-call constraint:",
                "\n\nRecovery-Bench recovery workflow:",
            )
            if (position := content.find(marker)) >= 0
        ]
        if marker_positions:
            message["content"] = content[: min(marker_positions)].rstrip() + "\n"
        break
    return copied


@dataclass
class SFTCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        width = max(len(feature["input_ids"]) for feature in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            padding = width - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [self.pad_token_id] * padding)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * padding)
            batch["labels"].append(feature["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


class SFTTrainer(Trainer):
    """Keep Liger's fused loss active during both training and validation."""

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ):
        inputs = {**inputs, "skip_logits": True}
        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ):
        return super().prediction_step(model, inputs, True, ignore_keys=ignore_keys)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--deepspeed", default=str(Path(__file__).resolve().parents[1] / "configs/deepspeed_zero3_optimizer_offload.json"))
    parser.add_argument("--max-length", type=int, default=65536)
    parser.add_argument(
        "--validation-max-length",
        type=int,
        default=None,
        help="validation-only length limit; defaults to --max-length",
    )
    parser.add_argument("--over-length-policy", choices=("error", "drop"), default="error")
    parser.add_argument(
        "--exclude-target-format-errors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="exclude trajectories with parser corrections in the supervised recovery segment",
    )
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--attention", choices=("auto", "sdpa", "flash_attention_2"), default="auto")
    parser.add_argument("--use-liger-kernel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-fast-linear-attention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-strategy", choices=("no", "steps", "epoch"), default="steps")
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--eval-strategy", choices=("no", "steps", "epoch"), default="steps")
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument(
        "--load-best-model-at-end",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="reload the checkpoint with the lowest validation loss after training",
    )
    parser.add_argument("--skip-final-save", action="store_true")
    args = parser.parse_args()

    if args.load_best_model_at_end and args.validation_data is None:
        parser.error("--load-best-model-at-end requires --validation-data")

    if args.require_fast_linear_attention and not (
        is_flash_linear_attention_available() and is_causal_conv1d_available()
    ):
        raise RuntimeError(
            "Qwen3.5 fast linear-attention kernels are unavailable. Install flash-linear-attention and "
            "causal-conv1d, or explicitly use --no-require-fast-linear-attention for a diagnostic run."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = SFTDataset(
        args.data,
        tokenizer,
        args.max_length,
        over_length_policy=args.over_length_policy,
        exclude_target_format_errors=args.exclude_target_format_errors,
    )
    validation_dataset = None
    if args.validation_data is not None:
        validation_max_length = args.validation_max_length or args.max_length
        validation_dataset = SFTDataset(
            args.validation_data,
            tokenizer,
            validation_max_length,
            over_length_policy=args.over_length_policy,
            exclude_target_format_errors=args.exclude_target_format_errors,
        )
    print(json.dumps({"train": dataset.summary()}, sort_keys=True))
    if validation_dataset is not None:
        print(json.dumps({"validation": validation_dataset.summary()}, sort_keys=True))
    attention = args.attention
    if attention == "auto":
        attention = "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=attention,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        # Transformers 5.x accepts a fractional warmup_steps value as a ratio.
        warmup_steps=args.warmup_ratio,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=True,
        logging_steps=1,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy=args.eval_strategy if validation_dataset is not None else "no",
        eval_steps=args.eval_steps,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model="eval_loss" if args.load_best_model_at_end else None,
        greater_is_better=False if args.load_best_model_at_end else None,
        deepspeed=args.deepspeed,
        remove_unused_columns=False,
        report_to=["tensorboard"],
        use_liger_kernel=args.use_liger_kernel,
        average_tokens_across_devices=True,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=validation_dataset,
        data_collator=SFTCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=True if list(args.output_dir.glob("checkpoint-*")) else None)
    if not args.skip_final_save:
        trainer.save_model(str(args.output_dir / "final"))
        tokenizer.save_pretrained(str(args.output_dir / "final"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
