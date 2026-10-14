from __future__ import annotations

import pytest
import torch
import torch.nn.functional as functional

from liger_kernel.transformers.model.loss_utils import LigerForCausalLMLoss


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Liger fused loss requires CUDA")
def test_liger_loss_and_gradients_match_reference_cross_entropy() -> None:
    torch.manual_seed(7)
    labels = torch.tensor(
        [
            [-100, -100, -100, 3, 4, 5, -100, 7, 8],
            [-100, 9, 10, -100, -100, 11, 12, 13, -100],
        ],
        device="cuda",
    )
    hidden = torch.randn(2, 9, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(64, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_(True)
    reference_weight = weight.detach().clone().requires_grad_(True)

    fused_loss = LigerForCausalLMLoss(hidden, weight, labels, hidden_size=32)
    shifted_labels = functional.pad(labels, (0, 1), value=-100)[..., 1:]
    logits = functional.linear(reference_hidden, reference_weight).float()
    reference_loss = functional.cross_entropy(
        logits.reshape(-1, 64),
        shifted_labels.reshape(-1),
        ignore_index=-100,
    )
    fused_loss.backward()
    reference_loss.backward()

    torch.testing.assert_close(fused_loss, reference_loss, rtol=0, atol=0)
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=0, atol=0)
    torch.testing.assert_close(weight.grad, reference_weight.grad, rtol=0, atol=0)

    active_targets = int(shifted_labels.ne(-100).sum())
    globally_normalized = LigerForCausalLMLoss(
        hidden.detach(),
        weight.detach(),
        labels,
        hidden_size=32,
        num_items_in_batch=active_targets,
    )
    torch.testing.assert_close(globally_normalized, reference_loss, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Liger fused loss requires CUDA")
def test_per_rank_losses_match_global_token_normalization() -> None:
    torch.manual_seed(11)
    labels_by_rank = [
        torch.tensor([[-100, 3, 4, -100, -100, 5]], device="cuda"),
        torch.tensor([[-100, 6, 7, 8, 9, 10]], device="cuda"),
    ]
    hidden_by_rank = [
        torch.randn(1, 6, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in labels_by_rank
    ]
    weight = torch.randn(64, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    reference_hidden = [hidden.detach().clone().requires_grad_(True) for hidden in hidden_by_rank]
    reference_weight = weight.detach().clone().requires_grad_(True)
    shifted = [functional.pad(labels, (0, 1), value=-100)[..., 1:] for labels in labels_by_rank]
    global_targets = sum(int(labels.ne(-100).sum()) for labels in shifted)

    # DDP averages gradients after Trainer multiplies each rank's globally
    # normalized loss by world size. The result is the sum below.
    distributed_loss = sum(
        LigerForCausalLMLoss(
            hidden,
            weight,
            labels,
            hidden_size=32,
            num_items_in_batch=global_targets,
        )
        for hidden, labels in zip(hidden_by_rank, labels_by_rank, strict=True)
    )
    reference_logits = torch.cat(
        [functional.linear(hidden, reference_weight).float() for hidden in reference_hidden],
        dim=0,
    )
    reference_labels = torch.cat(shifted, dim=0)
    reference_loss = functional.cross_entropy(
        reference_logits.reshape(-1, 64),
        reference_labels.reshape(-1),
        ignore_index=-100,
    )
    distributed_loss.backward()
    reference_loss.backward()

    torch.testing.assert_close(distributed_loss, reference_loss, rtol=1e-7, atol=2e-6)
    torch.testing.assert_close(weight.grad, reference_weight.grad, rtol=0.03, atol=1e-3)
    for hidden, reference in zip(hidden_by_rank, reference_hidden, strict=True):
        torch.testing.assert_close(hidden.grad, reference.grad, rtol=0.03, atol=1e-3)
