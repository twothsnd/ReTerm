"""Optional one-GPU smoke test for local Qwen3.5 weights and the bundled Bridge."""

import gc
import os

import pytest
import torch


def test_qwen35_forward_and_megatron_provider():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    model_path = os.environ.get("RETERM_TEST_MODEL_PATH")
    if not model_path:
        pytest.skip("set RETERM_TEST_MODEL_PATH to a local Qwen3.5 checkpoint")

    from megatron.bridge.models.conversion.auto_bridge import AutoBridge
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
    )
    model.eval()
    inputs = tokenizer("Hello.", return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        logits = model(**inputs).logits
    assert logits.shape[0] == 1
    assert torch.isfinite(logits).all()
    del logits, inputs, model
    gc.collect()
    torch.cuda.empty_cache()

    bridge = AutoBridge.from_hf_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="cpu",
    )
    provider = bridge.to_megatron_provider(load_weights=False)
    assert provider is not None
