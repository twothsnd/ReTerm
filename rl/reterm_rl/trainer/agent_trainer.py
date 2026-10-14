"""Distributed trainer for ReTerm-GRPO.

This module uses veRL as the distributed PPO/model engine substrate. The
ReTerm-specific part is the rollout loop:

1. ask ``ReTermAgentProxy`` to produce terminal task and recovery rollouts;
2. use verifier outcomes as token-level rewards;
3. run policy updates through veRL workers.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from pprint import pprint
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm
from verl import DataProto
from verl.trainer.ppo.metric_utils import reduce_metrics
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    apply_kl_penalty,
)
from verl.utils import tensordict_utils as tu
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding
from verl.utils.tracking import Tracking

from reterm_rl.bootstrap import ensure_local_paths

ensure_local_paths()

from reterm_rl.llm_agent import ReTermAgentProxy  # noqa: E402


def _sync_node_local_checkpoint_tree(
    checkpoint_dir: str,
    *,
    checkpoint_root: str,
    canonical_host: str,
    identity_file: str,
    direction: str,
    delete_after_push: bool = False,
) -> dict[str, Any]:
    """Move a node-local checkpoint tree to/from the canonical driver host.

    Megatron distributed checkpointing assumes a shared filesystem. When
    training nodes use separate filesystems, each non-driver node pushes its
    rank shards after save and pulls the complete tree before load. A Ray
    task pinned to that node executes this helper.
    """

    if direction not in {"push", "pull"}:
        raise ValueError(f"unsupported checkpoint sync direction: {direction!r}")
    root = Path(checkpoint_dir).expanduser().resolve()
    identity = Path(identity_file).expanduser().resolve()
    if not identity.is_file():
        raise FileNotFoundError(f"checkpoint sync SSH identity is missing: {identity}")
    ssh = [
        "ssh",
        "-i", str(identity),
        "-o", "BatchMode=yes",
        "-o", "IdentitiesOnly=yes",
        "-o", "StrictHostKeyChecking=yes",
        "-o", "ConnectTimeout=15",
    ]
    ssh_transport = " ".join(ssh)
    remote = f"{canonical_host}:{root}/"
    if direction == "push":
        if not root.is_dir():
            raise FileNotFoundError(f"node-local checkpoint tree is missing: {root}")
        mkdir = subprocess.run(
            [*ssh, canonical_host, "mkdir", "-p", str(root)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=120,
        )
        if mkdir.returncode != 0:
            raise RuntimeError(f"canonical checkpoint mkdir failed: {mkdir.stdout[-4000:]}")
        source, destination = f"{root}/", remote
    else:
        root.mkdir(parents=True, exist_ok=True)
        source, destination = remote, f"{root}/"

    process = subprocess.run(
        [
            "rsync", "-a", "--partial", "--protect-args",
            "-e", ssh_transport,
            source, destination,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=4 * 60 * 60,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"checkpoint {direction} rsync failed with rc={process.returncode}: "
            f"{process.stdout[-8000:]}"
        )
    files = [path for path in root.rglob("*") if path.is_file()]
    result = {
        "direction": direction,
        "node": os.uname().nodename,
        "checkpoint_dir": str(root),
        "files": len(files),
        "bytes": sum(path.stat().st_size for path in files),
    }
    if direction == "push" and delete_after_push:
        _remove_ephemeral_checkpoint_tree(str(root), checkpoint_root=checkpoint_root)
        result["deleted_after_push"] = True
    return result


def _remove_ephemeral_checkpoint_tree(checkpoint_dir: str, *, checkpoint_root: str) -> dict[str, Any]:
    """Delete only an exact non-canonical ``global_step_N/actor`` tree."""

    root = Path(checkpoint_dir).expanduser().resolve()
    configured_root = Path(checkpoint_root).expanduser().resolve()
    if (
        root.name != "actor"
        or not root.parent.name.startswith("global_step_")
        or not root.parent.name.removeprefix("global_step_").isdigit()
        or root.parent.parent != configured_root
    ):
        raise ValueError(f"refusing unsafe ephemeral checkpoint cleanup target: {root}")
    existed = root.exists()
    if existed:
        shutil.rmtree(root)
    parent_removed = False
    try:
        root.parent.rmdir()
        parent_removed = True
    except OSError:
        # The step directory can legitimately contain other node-local state.
        pass
    return {
        "node": os.uname().nodename,
        "checkpoint_dir": str(root),
        "existed": existed,
        "removed": not root.exists(),
        "empty_step_dir_removed": parent_removed,
    }


def compute_official_attempt_grpo_advantage(
    batch: DataProto,
    *,
    normalize_by_std: bool,
    epsilon: float = 1e-6,
) -> dict[str, float]:
    """Broadcast one same-task attempt advantage across its official-template segments.

    Advantages are additionally scaled so veRL's token-mean policy loss gives
    each complete attempt its configured weight, normalized by the sum of
    logical-attempt weights, independent of segment count or sampled tokens.
    """

    required = ("attempt_ids", "attempt_rewards", "group_ids")
    missing = [name for name in required if name not in batch.non_tensor_batch]
    if missing:
        raise ValueError(f"official attempt GRPO batch is missing metadata: {missing}")
    attempt_ids = np.asarray(batch.non_tensor_batch["attempt_ids"], dtype=object)
    attempt_rewards = np.asarray(batch.non_tensor_batch["attempt_rewards"], dtype=float)
    attempt_loss_weights = np.asarray(
        batch.non_tensor_batch.get("attempt_loss_weights", np.ones(len(attempt_ids))), dtype=float
    )
    group_ids = np.asarray(batch.non_tensor_batch["group_ids"], dtype=object)
    training_padding = np.asarray(
        batch.non_tensor_batch.get("training_padding", np.zeros(len(attempt_ids), dtype=bool)),
        dtype=bool,
    )
    response_mask = batch.batch["response_mask"].to(dtype=torch.float32)
    if not (
        len(attempt_ids)
        == len(attempt_rewards)
        == len(attempt_loss_weights)
        == len(group_ids)
        == len(training_padding)
        == response_mask.shape[0]
    ):
        raise ValueError("official attempt GRPO row metadata does not match tensor batch size")

    attempt_rows: dict[str, list[int]] = {}
    attempt_reward: dict[str, float] = {}
    attempt_loss_weight: dict[str, float] = {}
    attempt_group: dict[str, object] = {}
    for row, raw_attempt_id in enumerate(attempt_ids):
        if training_padding[row]:
            if bool(response_mask[row].any()):
                raise ValueError(f"official attempt GRPO padding row {row} contains response tokens")
            continue
        attempt_id = str(raw_attempt_id)
        attempt_rows.setdefault(attempt_id, []).append(row)
        reward = float(attempt_rewards[row])
        loss_weight = float(attempt_loss_weights[row])
        if not np.isfinite(loss_weight) or loss_weight < 0.0:
            raise ValueError(f"attempt {attempt_id} has an invalid loss weight: {loss_weight}")
        group_id = group_ids[row]
        if attempt_id in attempt_reward and not np.isclose(attempt_reward[attempt_id], reward):
            raise ValueError(f"attempt {attempt_id} has inconsistent rewards across segments")
        if attempt_id in attempt_group and attempt_group[attempt_id] != group_id:
            raise ValueError(f"attempt {attempt_id} crosses task groups")
        if attempt_id in attempt_loss_weight and not np.isclose(attempt_loss_weight[attempt_id], loss_weight):
            raise ValueError(f"attempt {attempt_id} has inconsistent loss weights across segments")
        attempt_reward[attempt_id] = reward
        attempt_loss_weight[attempt_id] = loss_weight
        attempt_group[attempt_id] = group_id

    group_attempts: dict[object, list[str]] = {}
    for attempt_id, group_id in attempt_group.items():
        group_attempts.setdefault(group_id, []).append(attempt_id)

    attempt_advantage: dict[str, float] = {}
    for group_id, ids in group_attempts.items():
        rewards = torch.tensor([attempt_reward[attempt_id] for attempt_id in ids], dtype=torch.float32)
        if len(ids) == 1:
            normalized = rewards
        else:
            centered = rewards - rewards.mean()
            normalized = centered / (rewards.std() + epsilon) if normalize_by_std else centered
        for attempt_id, value in zip(ids, normalized.tolist(), strict=True):
            attempt_advantage[attempt_id] = float(value)

    action_tokens_by_attempt: dict[str, float] = {}
    for attempt_id, rows in attempt_rows.items():
        count = float(response_mask[rows].sum().item())
        if count <= 0:
            raise ValueError(f"official attempt {attempt_id} has no response tokens")
        action_tokens_by_attempt[attempt_id] = count
    total_action_tokens = float(response_mask.sum().item())
    attempt_count = len(attempt_rows)
    if attempt_count == 0 or total_action_tokens <= 0:
        raise ValueError("official attempt GRPO batch contains no trainable attempts")
    total_attempt_weight = sum(attempt_loss_weight.values())
    if not np.isfinite(total_attempt_weight) or total_attempt_weight <= 0:
        raise ValueError("official attempt GRPO batch must have positive total loss weight")

    advantages = torch.zeros_like(response_mask)
    for attempt_id, rows in attempt_rows.items():
        equal_attempt_scale = total_action_tokens / (total_attempt_weight * action_tokens_by_attempt[attempt_id])
        advantages[rows] = (
            response_mask[rows]
            * float(attempt_advantage[attempt_id])
            * float(equal_attempt_scale)
            * attempt_loss_weight[attempt_id]
        )
    batch.batch["advantages"] = advantages
    batch.batch["returns"] = advantages.clone()
    return {
        "train/official_attempt_grpo_attempts": float(attempt_count),
        "train/official_attempt_grpo_weight_sum": float(total_attempt_weight),
        "train/official_attempt_grpo_segments": float(len(attempt_ids)),
        "train/official_attempt_grpo_padding_rows": float(training_padding.sum()),
        "train/official_attempt_grpo_action_tokens": total_action_tokens,
        "train/official_attempt_grpo_min_attempt_tokens": min(action_tokens_by_attempt.values()),
        "train/official_attempt_grpo_max_attempt_tokens": max(action_tokens_by_attempt.values()),
    }


class _NullStatefulDataloader:
    """Minimal checkpoint-compatible dataloader placeholder.

    ReTerm samples tasks through ReTermAgentProxy rather than from a
    static prompt dataset. veRL's checkpoint code still expects a dataloader
    object with ``len()``, ``state_dict()``, and ``load_state_dict()``.
    """

    def __init__(self, total_steps: int):
        self.total_steps = max(1, int(total_steps))

    def __len__(self) -> int:
        return self.total_steps

    def __iter__(self):
        for _ in range(self.total_steps):
            yield {}

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, _state: dict[str, Any]) -> None:
        return None


class ReTermTrainer(RayPPOTrainer):
    """veRL trainer with ReTerm task and recovery sampling."""

    agent_state_filename = "agent_state.json"

    def _create_dataloader(self, train_dataset=None, val_dataset=None, collate_fn=None, train_sampler=None):
        total_training_steps = int(self.config.trainer.total_training_steps)
        if total_training_steps < 0:
            raise ValueError("trainer.total_training_steps must be non-negative")
        self.total_training_steps = total_training_steps
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_dataloader = _NullStatefulDataloader(max(1, total_training_steps))
        self.val_dataloader = _NullStatefulDataloader(max(1, int(self.config.trainer.get("validation_steps", 1))))
        print(f"rl total training steps: {self.total_training_steps}")

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            if OmegaConf.select(self.config, "critic.optim"):
                self.config.critic.optim.total_training_steps = total_training_steps

    def init_agent_proxy(self):
        llm_client = getattr(self, "llm_server_manager", None)
        llm_client = llm_client.get_client() if llm_client is not None else None
        self.agent_proxy = ReTermAgentProxy(
            config=self.config,
            actor_rollout_wg=getattr(self, "actor_rollout_wg", None),
            tokenizer=self.tokenizer,
            llm_client=llm_client,
        )

    def _checkpoint_step_dir(self) -> Path:
        checkpoint_dir = Path(str(self.config.trainer.default_local_dir))
        if not checkpoint_dir.is_absolute():
            checkpoint_dir = Path.cwd() / checkpoint_dir
        return checkpoint_dir / f"global_step_{self.global_steps}"

    def _save_agent_proxy_checkpoint(self) -> None:
        if not hasattr(self, "agent_proxy"):
            return
        checkpoint_dir = self._checkpoint_step_dir()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        state_path = checkpoint_dir / self.agent_state_filename
        tmp_path = checkpoint_dir / f".{self.agent_state_filename}.tmp"
        payload = {
            "global_step": int(self.global_steps),
            "agent_proxy": self.agent_proxy.state_dict(),
        }
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(state_path)

    def _agent_proxy_checkpoint_to_load(self) -> Path:
        if self.config.trainer.resume_mode == "resume_path":
            resume_from_path = self.config.trainer.resume_from_path
            if isinstance(resume_from_path, str) and resume_from_path:
                resume_dir = Path(resume_from_path)
                if not resume_dir.is_absolute():
                    resume_dir = Path.cwd() / resume_dir
                resume_state_path = resume_dir / self.agent_state_filename
                if resume_state_path.exists():
                    return resume_state_path
        return self._checkpoint_step_dir() / self.agent_state_filename

    def _load_agent_proxy_checkpoint(self) -> None:
        if not hasattr(self, "agent_proxy") or self.global_steps <= 0:
            return
        state_path = self._agent_proxy_checkpoint_to_load()
        if not state_path.exists():
            print(f"[rl] no task sampler state found at {state_path}; task stream starts from scratch")
            return
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        state_step = int(payload.get("global_step", self.global_steps))
        if state_step != int(self.global_steps):
            raise ValueError(f"agent sampler state step {state_step} != checkpoint step {self.global_steps}")
        self.agent_proxy.load_state_dict(payload.get("agent_proxy", {}))
        print(f"[rl] restored task sampler state from {state_path}", flush=True)

    def _multi_node_checkpoint_sync_config(self) -> tuple[str, str] | None:
        if int(self.config.trainer.get("nnodes", 1)) <= 1:
            return None
        host = os.environ.get("RETERM_RL_CHECKPOINT_CANONICAL_HOST", "").strip()
        identity = os.environ.get("RETERM_RL_CHECKPOINT_SSH_IDENTITY", "").strip()
        if not host or not identity:
            raise RuntimeError(
                "multi-node training uses node-local filesystems but checkpoint synchronization is not configured; "
                "set RETERM_RL_CHECKPOINT_CANONICAL_HOST and RETERM_RL_CHECKPOINT_SSH_IDENTITY"
            )
        return host, identity

    def _non_driver_gpu_nodes(self) -> list[dict[str, Any]]:
        import ray

        driver_node_id = ray.get_runtime_context().get_node_id()
        return [
            node
            for node in ray.nodes()
            if node.get("Alive")
            and node.get("NodeID") != driver_node_id
            and float(node.get("Resources", {}).get("GPU", 0.0)) > 0.0
        ]

    def _sync_non_driver_checkpoint_nodes(
        self,
        checkpoint_dir: Path,
        *,
        direction: str,
        delete_after_push: bool = False,
    ) -> list[dict[str, Any]]:
        sync_config = self._multi_node_checkpoint_sync_config()
        if sync_config is None:
            return []
        canonical_host, identity = sync_config
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        remote_sync = ray.remote(num_cpus=0.01)(_sync_node_local_checkpoint_tree)
        results: list[dict[str, Any]] = []
        for node in self._non_driver_gpu_nodes():
            ref = remote_sync.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node["NodeID"], soft=False)
            ).remote(
                str(checkpoint_dir),
                checkpoint_root=str(self.config.trainer.default_local_dir),
                canonical_host=canonical_host,
                identity_file=identity,
                direction=direction,
                delete_after_push=delete_after_push,
            )
            results.append(ray.get(ref))
        return results

    def _cleanup_non_driver_checkpoint_nodes(self, checkpoint_dir: Path) -> list[dict[str, Any]]:
        sync_config = self._multi_node_checkpoint_sync_config()
        if sync_config is None:
            return []
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        remote_cleanup = ray.remote(num_cpus=0.01)(_remove_ephemeral_checkpoint_tree)
        results: list[dict[str, Any]] = []
        for node in self._non_driver_gpu_nodes():
            ref = remote_cleanup.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node["NodeID"], soft=False)
            ).remote(
                str(checkpoint_dir),
                checkpoint_root=str(self.config.trainer.default_local_dir),
            )
            results.append(ray.get(ref))
        return results

    def _prepare_multi_node_checkpoint_load(self) -> Path | None:
        sync_config = self._multi_node_checkpoint_sync_config()
        if sync_config is None or self.config.trainer.resume_mode == "disable":
            return None
        if self.config.trainer.resume_mode == "resume_path":
            checkpoint_dir = Path(str(self.config.trainer.resume_from_path)).expanduser().resolve()
        else:
            from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path

            latest = find_latest_ckpt_path(str(Path(self.config.trainer.default_local_dir).expanduser().resolve()))
            if latest is None:
                return None
            checkpoint_dir = Path(latest).resolve()
        complete = checkpoint_dir / ".node_sync_complete.json"
        if not complete.is_file():
            raise RuntimeError(
                f"refusing to resume an unconsolidated multi-node checkpoint: {checkpoint_dir}"
            )
        actor_dir = checkpoint_dir / "actor"
        results = self._sync_non_driver_checkpoint_nodes(actor_dir, direction="pull")
        print(
            "[rl] checkpoint_preload_sync_done "
            f"path={actor_dir} workers={json.dumps(results, sort_keys=True)}",
            flush=True,
        )
        return actor_dir

    def _save_checkpoint(self):
        self._save_agent_proxy_checkpoint()
        checkpoint_dir = self._checkpoint_step_dir()
        sync_config = self._multi_node_checkpoint_sync_config()
        incomplete = checkpoint_dir / ".node_sync_incomplete"
        complete = checkpoint_dir / ".node_sync_complete.json"
        if sync_config is not None:
            complete.unlink(missing_ok=True)
            incomplete.write_text("incomplete\n", encoding="utf-8")
        result = super()._save_checkpoint()
        if sync_config is not None:
            actor_dir = checkpoint_dir / "actor"
            workers = self._sync_non_driver_checkpoint_nodes(
                actor_dir,
                direction="push",
                delete_after_push=True,
            )
            payload = {
                "global_step": int(self.global_steps),
                "canonical_node": os.uname().nodename,
                "actor_dir": str(actor_dir),
                "workers": workers,
            }
            complete.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            incomplete.unlink(missing_ok=True)
            print(
                "[rl] checkpoint_node_sync_done "
                f"path={actor_dir} workers={json.dumps(workers, sort_keys=True)}",
                flush=True,
            )
        self._prune_checkpoint_history()
        return result

    def _prune_checkpoint_history(self) -> None:
        """Enforce checkpoint retention across process restarts.

        veRL's in-memory checkpoint manager only knows checkpoints written by
        the current trainer process.  A resumed process can therefore leave
        older, full checkpoints behind indefinitely.  Scan the canonical
        checkpoint directory after the current save is complete and retain
        the newest configured number of complete checkpoints globally.
        """
        keep = self.config.trainer.get("max_actor_ckpt_to_keep", None)
        if keep is None:
            return
        keep = int(keep)
        if keep <= 0:
            return

        root = Path(str(self.config.trainer.default_local_dir)).expanduser().resolve()
        current = self._checkpoint_step_dir().resolve()
        multi_node = self._multi_node_checkpoint_sync_config() is not None
        candidates: list[tuple[int, Path]] = []
        for path in root.glob("global_step_*"):
            if not path.is_dir():
                continue
            try:
                step = int(path.name.removeprefix("global_step_"))
            except ValueError:
                continue
            # Never prune partial saves.  Multi-node checkpoints become
            # canonical only after all remote actor shards are synchronized.
            if multi_node and not (path / ".node_sync_complete.json").is_file():
                continue
            if not (path / "actor").is_dir():
                continue
            candidates.append((step, path.resolve()))

        candidates.sort(key=lambda item: item[0], reverse=True)
        for step, path in candidates[keep:]:
            if path == current:
                continue
            shutil.rmtree(path)
            print(
                "[rl] checkpoint_retention_removed "
                f"step={step} path={path} keep={keep}",
                flush=True,
            )

    def _validate(self, merged: bool = False):  # noqa: ARG002 - veRL signature compatibility
        if not hasattr(self, "agent_proxy"):
            self.init_agent_proxy()
        reward_values: list[float] = []
        env_metric_dict: dict[str, list[float]] = {}
        validation_total = 0
        validation_valid = 0
        validation_steps = int(self.config.trainer.get("validation_steps", 1))
        for step in range(validation_steps):
            batch = self.agent_proxy.rollout(DataProto(), val=True, seed=10_000 + step)
            batch_rewards = batch.batch["rm_scores"].sum(dim=-1).cpu().tolist()
            metric_rewards = batch.non_tensor_batch.get("metric_rewards")
            if metric_rewards is None:
                metric_rewards = batch_rewards
            metric_rewards = np.asarray(metric_rewards, dtype=float)
            metric_valid = batch.non_tensor_batch.get("metric_valid")
            if metric_valid is None:
                metric_valid = np.ones(len(batch_rewards), dtype=bool)
            metric_valid = np.asarray(metric_valid, dtype=bool)
            metric_counted = batch.non_tensor_batch.get("metric_counted")
            if metric_counted is None:
                metric_counted = np.ones(len(batch_rewards), dtype=bool)
            metric_counted = np.asarray(metric_counted, dtype=bool)
            if (
                len(metric_valid) != len(batch_rewards)
                or len(metric_counted) != len(batch_rewards)
                or len(metric_rewards) != len(batch_rewards)
            ):
                raise ValueError("validation metric masks do not match rollout batch size")
            validation_total += int(metric_counted.sum())
            validation_valid += int((metric_counted & metric_valid).sum())
            reward_values.extend(
                float(reward)
                for reward, is_counted, is_valid in zip(
                    metric_rewards,
                    metric_counted,
                    metric_valid,
                    strict=True,
                )
                if is_counted and is_valid
            )
            for key, value in batch.meta_info.get("metrics", {}).items():
                env_metric_dict.setdefault(f"val/{key}", []).append(float(value))

        metrics = reduce_metrics(env_metric_dict) if env_metric_dict else {}
        if reward_values:
            metrics["val/recovery_reward_mean"] = float(np.mean(reward_values))
            metrics["val/recovery_reward_max"] = float(np.max(reward_values))
            metrics["val/recovery_reward_min"] = float(np.min(reward_values))
        metrics["val/evaluated_count"] = float(validation_total)
        metrics["val/valid_count"] = float(validation_valid)
        metrics["val/infrastructure_excluded_count"] = float(validation_total - validation_valid)
        return metrics

    def _update_actor(self, batch: DataProto) -> DataProto:
        """Keep one official segmented rollout batch as one optimizer mini-batch.

        A physical segment row is not an independent rollout sample. Splitting
        the flattened rows into fixed-size PPO mini-batches would make attempt
        weights depend on where a segment happened to land. Dynamic token
        micro-batching still handles memory inside this single logical update.
        """

        if str(self.config.algorithm.adv_estimator) != "official_attempt_grpo":
            raise ValueError("ReTerm-GRPO requires algorithm.adv_estimator=official_attempt_grpo")
        rollout_n = int(self.config.actor_rollout_ref.rollout.n)
        if rollout_n != 1:
            raise ValueError("official segmented GRPO currently requires actor_rollout_ref.rollout.n=1")
        logical_mini_batch_size = len(batch)
        original_mini_batch_size = int(self.config.actor_rollout_ref.actor.ppo_mini_batch_size)
        with open_dict(self.config.actor_rollout_ref.actor):
            self.config.actor_rollout_ref.actor.ppo_mini_batch_size = logical_mini_batch_size
        try:
            return super()._update_actor(batch)
        finally:
            with open_dict(self.config.actor_rollout_ref.actor):
                self.config.actor_rollout_ref.actor.ppo_mini_batch_size = original_mini_batch_size

    def _compute_old_log_prob(self, batch: DataProto):
        """Compute PPO old log-probs without the optional entropy tensor.

        veRL's default implementation forces ``calculate_entropy=True`` here.
        For long-context terminal trajectories, the entropy path materializes
        a large ``[tokens, vocab]`` computation and can OOM before the tensor is
        discarded. PPO only needs ``old_log_probs`` for the update path below.
        """

        batch_td = batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)
        calculate_sum_pi_squared = self.config.actor_rollout_ref.actor.get("calculate_sum_pi_squared", False)
        tu.assign_non_tensor(
            batch_td,
            calculate_entropy=False,
            calculate_sum_pi_squared=calculate_sum_pi_squared,
            compute_loss=False,
        )
        output = self.actor_rollout_wg.compute_log_prob(batch_td)
        log_probs = tu.get(output, "log_probs")
        routed_experts = tu.get(output, "routed_experts")
        sum_pi_squared = tu.get(output, "sum_pi_squared") if calculate_sum_pi_squared else None
        old_log_prob_mfu = tu.get(output, "metrics").get("mfu", 0.0)

        if os.environ.get("RETERM_RL_DEBUG_LOGPROB_PARITY", "0") == "1":
            raw_log_probs = log_probs.values() if log_probs.is_nested else log_probs
            raw_finite = raw_log_probs[torch.isfinite(raw_log_probs)]
            print(
                "[rl logprob raw] "
                f"shape={tuple(log_probs.shape)} nested={log_probs.is_nested} "
                f"tokens={raw_log_probs.numel()} finite={raw_finite.numel()} "
                f"nonzero={(raw_finite != 0).sum().item()} "
                f"min={raw_finite.min().item() if raw_finite.numel() else None} "
                f"max={raw_finite.max().item() if raw_finite.numel() else None} "
                f"mean={raw_finite.mean().item() if raw_finite.numel() else None}",
                flush=True,
            )

        log_probs = no_padding_2_padding(log_probs, batch_td)
        if os.environ.get("RETERM_RL_DEBUG_LOGPROB_PARITY", "0") == "1":
            response_mask = batch_td["response_mask"].bool()
            selected = log_probs[response_mask]
            print(
                "[rl logprob padded] "
                f"shape={tuple(log_probs.shape)} selected={selected.numel()} "
                f"nonzero={(selected != 0).sum().item()} "
                f"min={selected.min().item() if selected.numel() else None} "
                f"max={selected.max().item() if selected.numel() else None} "
                f"mean={selected.mean().item() if selected.numel() else None}",
                flush=True,
            )
        result = {"old_log_probs": log_probs.float()}
        if routed_experts is not None:
            result["routed_experts"] = routed_experts
        if sum_pi_squared is not None:
            result["sum_pi_squared"] = no_padding_2_padding(sum_pi_squared, batch_td).float()
        old_log_prob = DataProto.from_tensordict(tu.get_tensordict(result))
        return old_log_prob, old_log_prob_mfu

    def _should_measure_logprob_parity(self) -> bool:
        parity_config = self.config.algorithm.get("logprob_parity", {})
        if not parity_config or not bool(parity_config.get("enabled", False)):
            return False
        steps = parity_config.get("steps", [1])
        return not steps or int(self.global_steps) in {int(step) for step in steps}

    def _select_logprob_parity_batch(self, batch: DataProto) -> tuple[DataProto, dict[str, float]]:
        """Select a TP-aligned diagnostic subset without changing the PPO batch."""

        parity_config = self.config.algorithm.get("logprob_parity", {})
        megatron_config = self.config.actor_rollout_ref.actor.get("megatron", {})
        tp_size = int(megatron_config.get("tensor_model_parallel_size", 1))
        sequence_parallel = bool(megatron_config.get("sequence_parallel", False))
        alignment = tp_size if sequence_parallel else 1

        sequence_lengths = batch.batch["attention_mask"].sum(dim=-1)
        eligible_indices = torch.nonzero(sequence_lengths.remainder(alignment) == 0, as_tuple=False).flatten()

        max_samples = int(parity_config.get("max_samples", 64))
        total_gpus = int(self.config.trainer.get("n_gpus_per_node", 1)) * int(
            self.config.trainer.get("nnodes", 1)
        )
        pp_size = int(megatron_config.get("pipeline_model_parallel_size", 1))
        data_parallel_size = max(1, total_gpus // max(1, tp_size * pp_size))
        selected_count = min(max_samples, int(eligible_indices.numel()))
        selected_count -= selected_count % data_parallel_size
        if selected_count == 0:
            raise RuntimeError(
                "no TP-aligned trajectories available for logprob parity: "
                f"batch={len(batch)} alignment={alignment} dp={data_parallel_size}"
            )

        selected_indices = eligible_indices[:selected_count]
        return batch[selected_indices], {
            "train/logprob_parity/samples_total": float(len(batch)),
            "train/logprob_parity/samples_eligible": float(eligible_indices.numel()),
            "train/logprob_parity/samples_selected": float(selected_count),
            "train/logprob_parity/sequence_alignment": float(alignment),
        }

    def _logprob_parity_metrics(self, batch: DataProto, trainer_old_log_prob: DataProto) -> dict[str, float]:
        """Compare vLLM rollout log-probs with the pre-update trainer policy."""

        rollout_log_probs = batch.batch["old_log_probs"].float()
        trainer_log_probs = trainer_old_log_prob.batch["old_log_probs"].float()
        response_mask = batch.batch["response_mask"].bool()
        if rollout_log_probs.shape != trainer_log_probs.shape or response_mask.shape != rollout_log_probs.shape:
            raise ValueError(
                "logprob parity shape mismatch: "
                f"rollout={tuple(rollout_log_probs.shape)} "
                f"trainer={tuple(trainer_log_probs.shape)} "
                f"mask={tuple(response_mask.shape)}"
            )

        valid_mask = response_mask & torch.isfinite(rollout_log_probs) & torch.isfinite(trainer_log_probs)
        token_count = int(valid_mask.sum().item())
        if token_count == 0:
            return {
                "train/logprob_parity/valid": 0.0,
                "train/logprob_parity/tokens": 0.0,
            }

        logprob_diff = (trainer_log_probs - rollout_log_probs)[valid_mask]
        abs_logprob_diff = logprob_diff.abs()
        log_ratio = logprob_diff.clamp(min=-20.0, max=20.0)
        ratio = torch.exp(log_ratio)
        probability_diff = (
            torch.exp(trainer_log_probs[valid_mask]) - torch.exp(rollout_log_probs[valid_mask])
        ).abs()
        k3 = ratio - log_ratio - 1.0

        metrics = {
            "train/logprob_parity/valid": 1.0,
            "train/logprob_parity/tokens": float(token_count),
            "train/logprob_parity/logprob_diff_mean": float(logprob_diff.mean().item()),
            "train/logprob_parity/logprob_diff_abs_mean": float(abs_logprob_diff.mean().item()),
            "train/logprob_parity/logprob_diff_std": float(logprob_diff.std(unbiased=False).item()),
            "train/logprob_parity/logprob_diff_abs_max": float(abs_logprob_diff.max().item()),
            "train/logprob_parity/probability_diff_abs_mean": float(probability_diff.mean().item()),
            "train/logprob_parity/probability_diff_abs_max": float(probability_diff.max().item()),
            "train/logprob_parity/ratio_mean": float(ratio.mean().item()),
            "train/logprob_parity/k3_mean": float(k3.mean().item()),
            "train/logprob_parity/rollout_logprob_mean": float(rollout_log_probs[valid_mask].mean().item()),
            "train/logprob_parity/trainer_logprob_mean": float(trainer_log_probs[valid_mask].mean().item()),
        }

        for quantile in (0.5, 0.9, 0.95, 0.99, 0.999):
            label = str(quantile).replace("0.", "p")
            metrics[f"train/logprob_parity/logprob_diff_abs_{label}"] = float(
                torch.quantile(abs_logprob_diff, quantile).item()
            )
        for threshold in (0.01, 0.1, 1.0, 5.0):
            label = str(threshold).replace(".", "_")
            metrics[f"train/logprob_parity/logprob_diff_abs_gt_{label}"] = float(
                (abs_logprob_diff > threshold).float().mean().item()
            )

        adjacent_mask = valid_mask[:, 1:] & valid_mask[:, :-1]
        if adjacent_mask.any():
            metrics["train/logprob_parity/shift_rollout_prev_abs_mean"] = float(
                (trainer_log_probs[:, 1:] - rollout_log_probs[:, :-1])[adjacent_mask].abs().mean().item()
            )
            metrics["train/logprob_parity/shift_rollout_next_abs_mean"] = float(
                (trainer_log_probs[:, :-1] - rollout_log_probs[:, 1:])[adjacent_mask].abs().mean().item()
            )

        response_position_ids = batch.batch["position_ids"][:, 1:]
        sequence_lengths = batch.batch["attention_mask"].sum(dim=-1, keepdim=True).clamp(min=1)
        relative_positions = response_position_ids.float() / sequence_lengths.float()
        for bin_index, (lower, upper) in enumerate(((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01))):
            bin_mask = valid_mask & (relative_positions >= lower) & (relative_positions < upper)
            if bin_mask.any():
                metrics[f"train/logprob_parity/relative_position_bin_{bin_index}_abs_mean"] = float(
                    (trainer_log_probs - rollout_log_probs)[bin_mask].abs().mean().item()
                )

        flat_diff = (trainer_log_probs - rollout_log_probs).abs().masked_fill(~valid_mask, -1.0).flatten()
        top_count = min(100, token_count)
        top_values, top_indices = torch.topk(flat_diff, k=top_count)
        width = rollout_log_probs.shape[1]
        top_records = []
        for value, flat_index in zip(top_values.tolist(), top_indices.tolist(), strict=True):
            row, column = divmod(flat_index, width)
            token_id = int(batch.batch["responses"][row, column].item())
            top_records.append(
                {
                    "row": row,
                    "column": column,
                    "position_id": int(response_position_ids[row, column].item()),
                    "token_id": token_id,
                    "token": self.agent_proxy.tokenizer.decode(
                        [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
                    ),
                    "rollout_logprob": float(rollout_log_probs[row, column].item()),
                    "trainer_logprob": float(trainer_log_probs[row, column].item()),
                    "abs_diff": float(value),
                }
            )
        dump_root = Path(os.environ.get("TRACKING_ROOT", "."))
        dump_root.mkdir(parents=True, exist_ok=True)
        dump_path = dump_root / f"logprob_parity_step_{self.global_steps}.json"
        dump_path.write_text(json.dumps(top_records, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[rl] wrote logprob parity details to {dump_path}", flush=True)

        return metrics

    def fit(self):
        """Run rl PPO.

        Unlike veRL's default trainer, there is no static prompt dataloader.
        Each step samples task groups through ReTermAgentProxy, which
        schedules task and recovery attempts.
        """

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        ephemeral_actor_dir = self._prepare_multi_node_checkpoint_load()
        try:
            self._load_checkpoint()
        finally:
            if ephemeral_actor_dir is not None:
                cleanup_results = self._cleanup_non_driver_checkpoint_nodes(ephemeral_actor_dir)
                print(
                    "[rl] checkpoint_postload_cleanup_done "
                    f"path={ephemeral_actor_dir} workers={json.dumps(cleanup_results, sort_keys=True)}",
                    flush=True,
                )
        if getattr(self, "checkpoint_manager", None) is not None:
            self.checkpoint_manager.update_weights(self.global_steps)
        if not hasattr(self, "agent_proxy"):
            self.init_agent_proxy()
        self._load_agent_proxy_checkpoint()

        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False) or self.total_training_steps == 0:
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="rl Training")
        self.global_steps += 1

        while self.global_steps <= self.total_training_steps:
            step_start = time.time()
            metrics: dict[str, Any] = {"training/global_step": self.global_steps}

            section_start = time.time()
            batch = self.agent_proxy.rollout(DataProto(), val=False, seed=self.global_steps)
            metrics["timing/rollout_collect"] = time.time() - section_start
            if getattr(self, "checkpoint_manager", None) is not None:
                section_start = time.time()
                self.checkpoint_manager.sleep_replicas()
                metrics["timing/sleep_replicas"] = time.time() - section_start
            group_ids = batch.non_tensor_batch.get("group_ids")
            if group_ids is None:
                raise ValueError("ReTerm-GRPO rollout batch is missing group_ids")
            batch.non_tensor_batch["uid"] = np.array([str(group_id) for group_id in group_ids], dtype=object)
            # Recovery's official-Qwen representation flattens a logical attempt
            # into variable-length physical segment rows.  Preserve veRL's
            # normal DP workload balancing before the batch is sharded; without
            # this reorder, adjacent long segments can all land on one DP rank
            # while the other rank waits.  DataProto.reorder applies the same
            # permutation to tensors and attempt metadata, so loss targets and
            # rollout log-probs remain exactly aligned.
            if self.config.trainer.balance_batch:
                section_start = time.time()
                self._balance_batch(batch, metrics=metrics)
                metrics["timing/balance_batch"] = time.time() - section_start
            batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
            rollout_metrics = batch.meta_info.get("metrics", {})
            for key, value in rollout_metrics.items():
                metrics[f"train/{key}"] = float(value)
            all_sampled_reward_values = [
                float(value)
                for key, value in rollout_metrics.items()
                if key.endswith("/all_complete_groups_reward_mean")
            ]
            if all_sampled_reward_values:
                metrics["recovery/all_sampled_reward_mean"] = float(
                    np.mean(all_sampled_reward_values)
                )

            if "old_log_probs" in batch.batch.keys():
                metrics["perf/mfu/actor_infer"] = 0.0
                metrics["train/old_log_probs_source_rollout"] = 1.0
                if self._should_measure_logprob_parity():
                    section_start = time.time()
                    parity_batch, parity_selection_metrics = self._select_logprob_parity_batch(batch)
                    with torch.no_grad():
                        trainer_old_log_prob, parity_mfu = self._compute_old_log_prob(parity_batch)
                    metrics.update(parity_selection_metrics)
                    metrics.update(self._logprob_parity_metrics(parity_batch, trainer_old_log_prob))
                    metrics["timing/logprob_parity"] = time.time() - section_start
                    metrics["perf/mfu/logprob_parity"] = parity_mfu
                    parity_config = self.config.algorithm.get("logprob_parity", {})
                    if bool(parity_config.get("stop_after_measurement", False)):
                        metrics["timing/step"] = time.time() - step_start
                        logger.log(data=metrics, step=self.global_steps)
                        pprint(f"step:{self.global_steps} - " + " - ".join([f"{key}:{value}" for key, value in metrics.items()]))
                        progress_bar.update(1)
                        progress_bar.close()
                        return
            else:
                section_start = time.time()
                with torch.no_grad():
                    old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                metrics["timing/compute_old_log_prob"] = time.time() - section_start
                metrics["perf/mfu/actor_infer"] = old_log_prob_mfu
                metrics["train/old_log_probs_source_rollout"] = 0.0
                if "entropys" in old_log_prob.batch.keys():
                    old_log_prob.batch.pop("entropys")
                batch = batch.union(old_log_prob)

            if self.use_reference_policy:
                section_start = time.time()
                ref_log_prob = self._compute_ref_log_prob(batch)
                metrics["timing/compute_ref_log_prob"] = time.time() - section_start
                batch = batch.union(ref_log_prob)

            if self.use_critic:
                section_start = time.time()
                values = self._compute_values(batch)
                metrics["timing/compute_values"] = time.time() - section_start
                batch = batch.union(values)

            section_start = time.time()
            reward_tensor = batch.batch["rm_scores"]
            batch.batch["token_level_scores"] = reward_tensor
            if self.config.algorithm.use_kl_in_reward:
                batch, kl_metrics = apply_kl_penalty(
                    batch,
                    kl_ctrl=self.kl_ctrl_in_reward,
                    kl_penalty=self.config.algorithm.kl_penalty,
                )
                metrics.update(kl_metrics)
            else:
                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

            norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
            adv_estimator = str(self.config.algorithm.adv_estimator)
            if adv_estimator != "official_attempt_grpo":
                raise ValueError("ReTerm-GRPO requires algorithm.adv_estimator=official_attempt_grpo")
            if str(self.config.actor_rollout_ref.actor.loss_agg_mode) != "token-mean":
                raise ValueError("official segmented GRPO requires actor.loss_agg_mode=token-mean")
            metrics.update(
                compute_official_attempt_grpo_advantage(
                    batch,
                    normalize_by_std=bool(norm_adv_by_std_in_grpo),
                )
            )
            metrics["timing/advantage"] = time.time() - section_start

            if self.use_critic:
                section_start = time.time()
                critic_output = self._update_critic(batch)
                metrics["timing/update_critic"] = time.time() - section_start
                metrics.update(reduce_metrics(critic_output.meta_info["metrics"]))

            if self.config.trainer.critic_warmup <= self.global_steps:
                section_start = time.time()
                actor_output = self._update_actor(batch)
                metrics["timing/update_actor"] = time.time() - section_start
                metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

            should_save = (
                self.config.trainer.save_freq > 0
                and self.global_steps % self.config.trainer.save_freq == 0
            )
            if should_save:
                checkpoint_path = (
                    f"{self.config.trainer.default_local_dir}/global_step_{self.global_steps}"
                )
                checkpoint_start = time.time()
                print(
                    "[rl] checkpoint_save_start "
                    f"step={self.global_steps} path={checkpoint_path} "
                    "rollout_replicas_state=sleeping",
                    flush=True,
                )
                try:
                    # Save while rollout replicas are still asleep. Waking vLLM
                    # first can leave too little GPU headroom for FSDP state_dict.
                    self._save_checkpoint()
                except Exception as exc:
                    print(
                        "[rl] checkpoint_save_failed "
                        f"step={self.global_steps} path={checkpoint_path} "
                        f"error_type={type(exc).__name__} error={exc}",
                        flush=True,
                    )
                    raise
                checkpoint_elapsed = time.time() - checkpoint_start
                metrics["timing/checkpoint_save"] = checkpoint_elapsed
                print(
                    "[rl] checkpoint_save_done "
                    f"step={self.global_steps} path={checkpoint_path} "
                    f"elapsed_seconds={checkpoint_elapsed:.3f}",
                    flush=True,
                )

            if getattr(self, "checkpoint_manager", None) is not None:
                section_start = time.time()
                self.checkpoint_manager.update_weights(self.global_steps)
                metrics["timing/update_weights"] = time.time() - section_start

            if self.config.trainer.test_freq > 0 and self.global_steps % self.config.trainer.test_freq == 0:
                section_start = time.time()
                metrics.update(self._validate())
                metrics["timing/validate"] = time.time() - section_start

            row_rewards = batch.batch["rm_scores"].sum(dim=-1)
            metric_counted = batch.non_tensor_batch.get("metric_counted")
            if metric_counted is not None:
                metric_counted = np.asarray(metric_counted, dtype=bool)
                if len(metric_counted) != len(row_rewards):
                    raise ValueError("metric_counted metadata does not match rollout batch size")
                rewards = row_rewards[torch.as_tensor(metric_counted, device=row_rewards.device)].cpu().tolist()
            else:
                rewards = row_rewards.cpu().tolist()
            metrics["recovery/reward_mean"] = float(np.mean(rewards)) if rewards else 0.0
            metrics["recovery/reward_max"] = float(np.max(rewards)) if rewards else 0.0
            metrics["recovery/reward_min"] = float(np.min(rewards)) if rewards else 0.0
            metrics["timing/step"] = time.time() - step_start
            logger.log(data=metrics, step=self.global_steps)

            progress_bar.update(1)
            self.global_steps += 1

        progress_bar.close()
