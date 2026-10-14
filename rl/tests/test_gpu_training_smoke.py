"""Small CUDA checks for the published ReTerm-GRPO training path."""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
from verl.workers.engine_workers import _materialize_weights_for_shm
from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import (
    BucketedWeightReceiver,
    BucketedWeightSender,
)

from reterm_rl.trainer.agent_trainer import compute_official_attempt_grpo_advantage


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def test_sparse_assistant_mask_and_attempt_loss_backward_on_gpu():
    # Each task group has two attempts. Internal zeros represent inherited
    # history/tool observations, not padding or trainable assistant tokens.
    mask = torch.tensor(
        [[1, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 1], [0, 1, 0, 0]],
        dtype=torch.bool,
        device="cuda",
    )
    batch = DataProto()
    batch.batch = TensorDict({"response_mask": mask}, batch_size=4)
    batch.non_tensor_batch = {
        "attempt_ids": np.array(["initial-a", "initial-b", "recovery-a", "recovery-b"], dtype=object),
        "attempt_rewards": np.array([0.0, 1.0, 1.0, 0.0], dtype=object),
        "group_ids": np.array(["initial", "initial", "recovery", "recovery"], dtype=object),
    }
    compute_official_attempt_grpo_advantage(batch, normalize_by_std=False)

    log_prob = torch.zeros(mask.shape, device="cuda", requires_grad=True)
    old_log_prob = torch.zeros_like(log_prob)
    config = OmegaConf.create({
        "clip_ratio": 0.2,
        "clip_ratio_low": 0.2,
        "clip_ratio_high": 0.28,
        "global_batch_info": {"batch_num_tokens": int(mask.sum()), "dp_size": 1},
    })
    loss, _ = compute_policy_loss_vanilla(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=batch.batch["advantages"],
        response_mask=mask,
        loss_agg_mode="token-mean",
        config=config,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert log_prob.grad is not None
    assert torch.count_nonzero(log_prob.grad[~mask]) == 0
    assert torch.count_nonzero(log_prob.grad[mask]) == int(mask.sum())
    assert batch.batch["advantages"][0].sum() < 0
    assert batch.batch["advantages"][2].sum() > 0


def test_cuda_weights_are_materialized_for_shared_memory(monkeypatch):
    monkeypatch.setenv("VERL_WEIGHT_TRANSFER_USE_SHM", "1")
    weight = torch.arange(24, dtype=torch.bfloat16, device="cuda").reshape(4, 6)[:, ::2]

    [(name, materialized)] = _materialize_weights_for_shm([("weight", weight)])
    assert name == "weight"
    assert materialized.device.type == "cpu"
    assert materialized.is_contiguous()
    torch.testing.assert_close(materialized, weight.cpu())

    sender = object.__new__(BucketedWeightSender)
    sender.buffer = torch.empty(128, dtype=torch.uint8, device="cpu")
    payload = sender._prepare_weight_payload("weight", weight)
    assert payload.device.type == "cpu"
    assert payload.is_contiguous()
    torch.testing.assert_close(payload, materialized)


def test_shared_memory_weight_sender_receiver_on_gpu(tmp_path):
    address = f"ipc://{tmp_path / 'weights.sock'}"
    sender = BucketedWeightSender(address, bucket_size_mb=1, use_shm=True)
    receiver = BucketedWeightReceiver(address, device=torch.device("cuda:0"), use_shm=True)
    weight = torch.arange(24, dtype=torch.bfloat16, device="cuda").reshape(4, 6)[:, ::2]
    received = []

    def on_bucket(weights):
        received.extend((name, tensor.clone()) for name, tensor in weights)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(receiver.receive_weights, on_bucket)
        asyncio.run(sender.async_send_weights([("weight", weight)]))
        future.result(timeout=30)

    assert len(received) == 1
    assert received[0][0] == "weight"
    assert received[0][1].device.type == "cuda"
    torch.testing.assert_close(received[0][1], weight.contiguous())
