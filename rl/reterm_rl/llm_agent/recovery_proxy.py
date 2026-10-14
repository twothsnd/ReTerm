"""veRL rollout proxy for ReTerm-GRPO sampling."""

from __future__ import annotations

import random
import asyncio
import hashlib
import math
import os
import re
import threading
import time
import traceback
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from verl import DataProto

from reterm_rl.agents import PolicyResponse, TerminalPolicyAgent, TerminalPolicyAgentConfig
from reterm_core.qwen_tools import TERMINAL_TOOLS
from reterm_rl.agents.terminal_policy import RolloutCancelledError, RolloutTimeoutError
from reterm_rl.task_adapter import TaskInfrastructureError, create_task_dataset
from reterm_rl.bootstrap import ensure_local_paths
from reterm_rl.llm_agent.recovery_buffer import NodeLocalRecoveryBuffer

ensure_local_paths()

from reterm_core.attempt import AttemptRunner  # noqa: E402
from reterm_core.errors import TaskSkip  # noqa: E402
from reterm_core.types import (  # noqa: E402
    AgentContext,
    AttemptStatus,
    BenchmarkResult,
)


def _to_plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    return dict(OmegaConf.to_container(value, resolve=True))


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


class ReTermAgentProxy:
    """Generate task and recovery rollouts for ReTerm-GRPO."""

    _INFRASTRUCTURE_FAILURE_CATEGORIES = (
        "environment",
        "evaluation",
        "generation_backend",
    )

    def __init__(
        self,
        config,
        actor_rollout_wg,
        tokenizer,
        llm_client=None,
        *,
        enable_distributed_sandboxes: bool = True,
    ):
        self.config = config
        self.actor_wg = actor_rollout_wg
        self.llm_client = llm_client
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.train_task_ids = self._discover_task_ids(val=False)
        self.val_task_ids = self._discover_task_ids(val=True)
        if not self.train_task_ids:
            raise ValueError("rl discovered no training tasks")
        if not self.val_task_ids:
            raise ValueError("rl discovered no validation tasks")
        self._task_sampling_seed = self._task_seed()
        self._task_streams = {
            False: {"epoch": -1, "cursor": 0, "order": []},
            True: {"epoch": -1, "cursor": 0, "order": []},
        }
        self._task_stream_lock = threading.Lock()
        self.trajectory_dir = self._trajectory_dir()
        self._trajectory_lock = threading.Lock()
        self._buffer_lease_lock = threading.Lock()
        self._rollout_deadline = threading.local()
        self._rollout_cancel_event = threading.local()
        self._generation_seed = threading.local()
        self._generation_turn = threading.local()
        dataset_cfg = _to_plain_dict(self.config.recovery_env.dataset)
        state_root = Path(
            str(dataset_cfg.get("state_root", "state"))
        ).expanduser().resolve()
        buffer_root = Path(
            str(
                getattr(
                    self.config.recovery_env,
                    "recovery_buffer_root",
                    state_root.parent / "recovery_buffer",
                )
            )
        ).expanduser().resolve()
        self._recovery_buffer = NodeLocalRecoveryBuffer(
            capacity=int(getattr(self.config.recovery_env, "recovery_buffer_capacity_per_node", 12)),
            storage_root=buffer_root,
        )
        self._sandbox_workers: list[Any] = []
        if enable_distributed_sandboxes:
            self._init_distributed_sandbox_workers()

    def _init_distributed_sandbox_workers(self) -> None:
        if os.environ.get("RETERM_RL_DISTRIBUTED_SANDBOXES", "1") != "1":
            return

        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        configured_ips = [
            value.strip()
            for value in os.environ.get("RETERM_RL_SANDBOX_NODE_IPS", "").split(",")
            if value.strip()
        ]
        nodes_by_ip = {
            str(node.get("NodeManagerAddress")): str(node.get("NodeID"))
            for node in ray.nodes()
            if node.get("Alive")
        }
        if not configured_ips:
            configured_ips = sorted(nodes_by_ip)
        missing = [node_ip for node_ip in configured_ips if node_ip not in nodes_by_ip]
        if missing:
            raise RuntimeError(
                "distributed sandbox Ray nodes are unavailable: "
                f"missing={missing} available={sorted(nodes_by_ip)}"
            )

        per_node_limit = max(1, int(os.environ.get("RETERM_RL_SANDBOX_WORKERS_PER_NODE", "128")))
        self._sandbox_per_node_limit = per_node_limit
        min_available_gb = max(
            0.0,
            float(os.environ.get("RETERM_RL_SANDBOX_MIN_AVAILABLE_GB", "0")),
        )
        # ``run_episode`` acquires a node-local semaphore *inside* the actor.
        # Ray actor calls waiting on that semaphore still consume actor
        # concurrency slots.  If those calls fill the small historical
        # ``per_node_limit + 8`` allowance, the control-plane
        # ``cancel_rollout`` RPC queues behind the work it is meant to cancel.
        # Size actor concurrency for every central executor thread plus a
        # dedicated control reserve; the semaphore remains the sole authority
        # for how many task environments may run at once.
        train_max_workers = int(getattr(self.config.es_manager.train, "max_workers", per_node_limit))
        val_max_workers = int(getattr(self.config.es_manager.val, "max_workers", 1))
        actor_call_budget = max(per_node_limit, train_max_workers, val_max_workers)
        control_reserve = 16
        actor_cls = ray.remote(
            num_cpus=0,
            max_concurrency=actor_call_budget + control_reserve,
        )(_DistributedSandboxEpisodeWorker)
        for node_index, node_ip in enumerate(configured_ips):
            worker = actor_cls.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=nodes_by_ip[node_ip],
                    soft=False,
                )
            ).remote(
                config=self.config,
                tokenizer=self.tokenizer,
                llm_client=self.llm_client,
                node_ip=node_ip,
                node_index=node_index,
                max_episodes=per_node_limit,
                min_available_gb=min_available_gb,
            )
            self._sandbox_workers.append(worker)
        statuses = ray.get([worker.status.remote() for worker in self._sandbox_workers])
        print(
            "[rl] distributed sandbox workers ready: "
            f"nodes={statuses} total_episode_limit={per_node_limit * len(statuses)}",
            flush=True,
        )

    def _new_sandbox_rollout_id(self, *, val: bool, rollout_seed: int) -> str:
        return f"{'val' if val else 'train'}-{rollout_seed}-{uuid.uuid4().hex}"

    def _initial_worker_limit(self, configured_workers: int) -> int:
        return max(1, int(configured_workers))

    def _submit_episode(
        self,
        executor: ThreadPoolExecutor,
        *,
        sandbox_rollout_id: str | None,
        task_id: str,
        env_id: int,
        group_id: int,
        sample_index: int,
        val: bool,
        rollout_seed: int,
        cancel_event: threading.Event | None = None,
        buffer_rollout_id: str | None = None,
    ):
        kwargs = {
            "task_id": task_id,
            "env_id": env_id,
            "group_id": group_id,
            "sample_index": sample_index,
            "val": val,
            "rollout_seed": rollout_seed,
            "buffer_rollout_id": buffer_rollout_id,
        }
        if not self._sandbox_workers:
            return executor.submit(self._run_one_episode, **kwargs, cancel_event=cancel_event)
        if sandbox_rollout_id is None:
            raise RuntimeError("distributed sandbox submission is missing rollout id")
        # Keep all attempts from one GRPO group on the same worker so they share
        # one task environment implementation and its state storage.
        worker = self._sandbox_workers[group_id % len(self._sandbox_workers)]
        return executor.submit(
            self._get_distributed_episode,
            worker,
            sandbox_rollout_id,
            kwargs,
            cancel_event,
        )

    @staticmethod
    def _get_distributed_episode(
        worker: Any,
        rollout_id: str,
        kwargs: dict[str, Any],
        cancel_event: threading.Event | None,
    ) -> dict[str, Any]:
        import ray

        object_ref = worker.run_episode.remote(rollout_id=rollout_id, kwargs=kwargs)
        while True:
            try:
                return ray.get(object_ref, timeout=0.25)
            except ray.exceptions.GetTimeoutError:
                if cancel_event is None or not cancel_event.is_set():
                    continue
                # Do not ray.cancel(object_ref) here.  run_episode may be in
                # the middle of an async vLLM RPC; cancelling its Ray task can
                # punch a hole in the ordered actor request stream and poison
                # the load balancer for the next optimizer step.  The central
                # waiter can return immediately while cancel_rollout() drives
                # node-local VM and generation teardown independently.
                raise RolloutCancelledError(
                    f"distributed sandbox rollout cancelled while awaiting Ray result: {rollout_id}"
                )
            except ray.exceptions.TaskCancelledError as exc:
                if cancel_event is not None and cancel_event.is_set():
                    raise RolloutCancelledError(
                        f"distributed sandbox Ray task cancelled: {rollout_id}"
                    ) from exc
                raise

    def _cancel_distributed_sandbox_rollout(self, rollout_id: str | None) -> None:
        if rollout_id is None or not self._sandbox_workers:
            return
        import ray

        ray.get([worker.cancel_rollout.remote(rollout_id) for worker in self._sandbox_workers])

    def _finish_distributed_sandbox_rollout(self, rollout_id: str | None) -> None:
        if rollout_id is None or not self._sandbox_workers:
            return
        import ray

        ray.get([worker.finish_rollout.remote(rollout_id) for worker in self._sandbox_workers])

    def _resolve_buffer_group(
        self,
        *,
        rollout_id: str | None,
        group_id: int,
        promote: bool,
    ) -> dict[str, Any] | None:
        if rollout_id is None:
            raise RuntimeError("reterm group resolution is missing rollout id")
        if not self._sandbox_workers:
            return self._recovery_buffer.resolve_provisional(
                rollout_id=rollout_id,
                source_group_id=group_id,
                promote=promote,
            )
        import ray

        worker = self._sandbox_workers[group_id % len(self._sandbox_workers)]
        return ray.get(
            worker.resolve_buffer_group.remote(
                rollout_id=rollout_id,
                group_id=group_id,
                promote=promote,
            )
        )

    @staticmethod
    def _get_distributed_buffered_recovery(
        worker: Any,
        rollout_id: str,
        lease_id: str,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        import ray

        return ray.get(
            worker.run_buffered_recovery.remote(
                rollout_id=rollout_id,
                lease_id=lease_id,
                kwargs=kwargs,
            )
        )

    def rollout(self, dataproto: DataProto, val: bool = False, seed: int | None = None) -> DataProto:
        mode_cfg = self.config.es_manager.val if val else self.config.es_manager.train
        env_groups = int(mode_cfg.env_groups)
        group_size = int(mode_cfg.group_size)
        rollout_seed = int(seed if seed is not None else self._task_sampling_seed)
        sandbox_rollout_id = self._new_sandbox_rollout_id(val=val, rollout_seed=rollout_seed)

        if not val:
            self._reterm_initial_group_quota(env_groups, group_size)
            bootstrap_steps = int(getattr(self.config.recovery_env, "recovery_buffer_bootstrap_steps", 2))
            if rollout_seed <= bootstrap_steps:
                env_groups = int(getattr(self.config.recovery_env, "recovery_buffer_bootstrap_groups", env_groups))
                if env_groups < 1:
                    raise ValueError("reterm bootstrap groups must be positive")
            buffers_ready = False
            if rollout_seed > bootstrap_steps and self._sandbox_workers:
                import ray

                statuses = ray.get([worker.status.remote() for worker in self._sandbox_workers])
                buffers_ready = sum(
                    int(status["recovery_buffer"]["available"]) for status in statuses
                ) >= 2 * len(self._sandbox_workers)
            if rollout_seed > bootstrap_steps and buffers_ready:
                return self._rollout_buffered_recovery_batch(
                    mode_cfg=mode_cfg,
                    target_groups=env_groups,
                    group_size=group_size,
                    rollout_seed=rollout_seed,
                )

        if self._steady_state_sampling_enabled(val=val):
            return self._rollout_steady_state(
                mode_cfg=mode_cfg,
                target_groups=env_groups,
                group_size=group_size,
                rollout_seed=rollout_seed,
                val=val,
                sandbox_rollout_id=sandbox_rollout_id,
            )

        return self._rollout_initial_groups(
            mode_cfg=mode_cfg,
            env_groups=env_groups,
            group_size=group_size,
            rollout_seed=rollout_seed,
            val=val,
            sandbox_rollout_id=sandbox_rollout_id,
        )

    def _rollout_initial_groups(
        self,
        *,
        mode_cfg: Any,
        env_groups: int,
        group_size: int,
        rollout_seed: int,
        val: bool,
        sandbox_rollout_id: str | None,
        max_workers_override: int | None = None,
        return_examples: bool = False,
    ) -> DataProto | tuple[list[dict[str, Any]], dict[str, float]]:
        total = env_groups * group_size
        group_examples: dict[int, list[dict[str, Any]]] = {group_id: [] for group_id in range(env_groups)}
        group_candidates: dict[int, list[dict[str, Any]]] = {group_id: [] for group_id in range(env_groups)}
        metrics: dict[str, list[float]] = {}
        worker_env_key = "RETERM_RL_VAL_ROLLOUT_WORKERS" if val else "RETERM_RL_ROLLOUT_WORKERS"
        configured_workers = int(getattr(mode_cfg, "max_workers", total))
        requested_workers = (
            int(max_workers_override)
            if max_workers_override is not None
            else int(os.environ.get(worker_env_key, str(configured_workers)))
        )
        max_workers = max(1, min(total, self._initial_worker_limit(requested_workers)))
        max_replacement_tasks = self._max_replacement_tasks()
        task_by_group = {group_id: task_id for group_id, task_id in enumerate(self._next_task_ids(env_groups, val=val))}
        attempts_by_group = {group_id: 0 for group_id in range(env_groups)}
        replacements_by_group = {group_id: 0 for group_id in range(env_groups)}
        resolved_groups: set[int] = set()

        executor = ThreadPoolExecutor(max_workers=max_workers)
        cancel_event = threading.Event()
        pending = {}
        rollout_complete = False
        rollout_start = time.time()
        submit_seconds = 0.0
        wait_seconds = 0.0
        future_result_seconds = 0.0
        metrics_merge_seconds = 0.0
        finish_group_seconds = 0.0

        def all_groups_complete() -> bool:
            return all(len(group_examples[group_id]) >= group_size for group_id in range(env_groups))

        def submit_group(group_id: int) -> None:
            nonlocal submit_seconds
            if len(group_examples[group_id]) >= group_size or group_candidates[group_id]:
                return
            submit_start = time.time()
            task_id = task_by_group[group_id]
            missing = group_size - len(group_examples[group_id])
            for _ in range(missing):
                env_id = group_id * group_size + attempts_by_group[group_id]
                sample_index = attempts_by_group[group_id]
                attempts_by_group[group_id] += 1
                future = self._submit_episode(
                    executor,
                    sandbox_rollout_id=sandbox_rollout_id,
                    task_id=task_id,
                    env_id=env_id,
                    group_id=group_id,
                    sample_index=sample_index,
                    val=val,
                    rollout_seed=rollout_seed,
                    cancel_event=cancel_event,
                    buffer_rollout_id=sandbox_rollout_id,
                )
                pending[future] = (group_id, task_id)
            submit_seconds += time.time() - submit_start

        def finish_group_if_ready(group_id: int) -> None:
            nonlocal finish_group_seconds
            finish_start = time.time()
            candidates = group_candidates[group_id]
            if len(group_examples[group_id]) + len(candidates) < group_size:
                return
            try:
                uses_group_replacement = (not val) and self._active_sampling_enabled()
                trainable: list[dict[str, Any]] = list(group_examples[group_id])
                non_trainable_count = 0
                over_context_count = 0
                for result in candidates:
                    check_start = time.time()
                    is_trainable = self._is_trainable_example(result, allow_discard=val)
                    metrics.setdefault("rollout_trainable_check_seconds", []).append(time.time() - check_start)
                    if is_trainable:
                        trainable.append(result)
                    else:
                        non_trainable_count += 1
                        over_context_start = time.time()
                        over_context_count += int(self._is_over_context_example(result))
                        metrics.setdefault("rollout_over_context_check_seconds", []).append(
                            time.time() - over_context_start
                        )

                if len(trainable) < group_size:
                    previous_trainable_count = len(group_examples[group_id])
                    newly_trainable_count = len(trainable) - previous_trainable_count
                    metrics.setdefault("rollout_non_trainable_refill_groups", []).append(1.0)
                    metrics.setdefault("rollout_non_trainable_refill", []).append(float(non_trainable_count))
                    if over_context_count:
                        metrics.setdefault("rollout_over_context_refill", []).append(float(over_context_count))
                    group_examples[group_id] = trainable
                    group_candidates[group_id] = []
                    max_attempts_for_task = self._max_refill_attempts_per_task(group_size)
                    if newly_trainable_count == 0 or attempts_by_group[group_id] >= max_attempts_for_task:
                        metrics.setdefault("rollout_non_trainable_task_replacements", []).append(1.0)
                        replaced = self._replace_group_task(
                            group_id=group_id,
                            task_by_group=task_by_group,
                            attempts_by_group=attempts_by_group,
                            replacements_by_group=replacements_by_group,
                            group_examples=group_examples,
                            group_candidates=group_candidates,
                            max_replacement_tasks=max_replacement_tasks,
                            val=val,
                            reason=(
                                f"refill_stalled newly_trainable={newly_trainable_count} "
                                f"non_trainable={non_trainable_count} attempts={attempts_by_group[group_id]}"
                            ),
                        )
                        if replaced and not val:
                            self._resolve_buffer_group(
                                rollout_id=sandbox_rollout_id,
                                group_id=group_id,
                                promote=False,
                            )
                        if not replaced:
                            raise RuntimeError(
                                f"group {group_id} exhausted {max_replacement_tasks} replacement tasks "
                                "while collecting trainable rollouts"
                            )
                    submit_group(group_id)
                    return

                if (
                    uses_group_replacement
                    and group_size > 1
                    and len(trainable) == group_size
                    and self._group_reward_std(trainable) == 0.0
                ):
                    rewards = [float(example["reward"]) for example in trainable]
                    metrics.setdefault("rollout_zero_std_groups", []).append(1.0)
                    metrics.setdefault("rollout_zero_std", []).append(1.0)
                    if rewards and rewards[0] <= 0.0:
                        metrics.setdefault("rollout_zero_std_all_fail", []).append(1.0)
                    else:
                        metrics.setdefault("rollout_zero_std_all_success", []).append(1.0)
                    replaced = self._replace_group_task(
                        group_id=group_id,
                        task_by_group=task_by_group,
                        attempts_by_group=attempts_by_group,
                        replacements_by_group=replacements_by_group,
                        group_examples=group_examples,
                        group_candidates=group_candidates,
                        max_replacement_tasks=max_replacement_tasks,
                        val=val,
                        reason=f"zero_std rewards={rewards}",
                    )
                    if replaced:
                        self._resolve_buffer_group(
                            rollout_id=sandbox_rollout_id,
                            group_id=group_id,
                            promote=False,
                        )
                        submit_group(group_id)
                        return
                    metrics.setdefault("rollout_active_sampling_fallback_zero_std", []).append(1.0)
                    print(
                        "[rl] active sampling replacement limit reached; "
                        f"accepting zero-std group={group_id} rewards={rewards}",
                        flush=True,
                    )

                if len(trainable) == group_size:
                    for example in trainable:
                        dataset_name = str(example.get("dataset_name", self._dataset_name()))
                        reward_metric = f"{dataset_name}/reward"
                        metrics.setdefault(reward_metric, []).append(float(example["reward"]))
                        metrics.setdefault("rollout_final_reward", []).append(float(example["reward"]))
                        metrics.setdefault("rollout_attempt_turns", []).append(
                            float(example.get("attempt_turns", 0.0))
                        )
                    for example in trainable:
                        dataset_name = str(example.get("dataset_name", self._dataset_name()))
                        if getattr(self, "trajectory_dir", None) is not None and not example.get("trajectory_saved"):
                            save_start = time.time()
                            self._save_trajectory(
                                dataset_name=dataset_name,
                                episode=example,
                                benchmark_result=example.get("benchmark_result"),
                            )
                            example["trajectory_saved"] = True
                            metrics.setdefault("rollout_save_trajectory_seconds", []).append(
                                time.time() - save_start
                            )
                    cleanup_start = time.time()
                    metrics.setdefault("rollout_group_cleanup_seconds", []).append(time.time() - cleanup_start)
                    group_examples[group_id] = trainable
                    group_candidates[group_id] = []
                    if not val:
                        self._resolve_buffer_group(
                            rollout_id=sandbox_rollout_id,
                            group_id=group_id,
                            promote=True,
                        )
                        resolved_groups.add(group_id)
                    return

            finally:
                finish_group_seconds += time.time() - finish_start

        try:
            for group_id in range(env_groups):
                submit_group(group_id)

            while pending and not rollout_complete:
                wait_start = time.time()
                done, _not_done = wait(pending, return_when=FIRST_COMPLETED)
                wait_seconds += time.time() - wait_start
                for future in done:
                    group_id, future_task_id = pending.pop(future)
                    future_result_start = time.time()
                    result = future.result()
                    future_result_seconds += time.time() - future_result_start
                    metrics_merge_start = time.time()
                    for key, value in result["metrics"].items():
                        metrics.setdefault(key, []).append(float(value))
                    metrics_merge_seconds += time.time() - metrics_merge_start

                    if future_task_id != task_by_group[group_id]:
                        metrics.setdefault("rollout_stale", []).append(1.0)
                        continue
                    if len(group_examples[group_id]) >= group_size:
                        metrics.setdefault("rollout_extra_after_group_complete", []).append(1.0)
                        continue

                    group_candidates[group_id].append(result)
                    finish_group_if_ready(group_id)

                    if all_groups_complete():
                        rollout_complete = True
                        break
        finally:
            cancel_event.set()
            for future in pending:
                future.cancel()
            self._cancel_distributed_sandbox_rollout(sandbox_rollout_id)
            shutdown_start = time.time()
            executor.shutdown(wait=True, cancel_futures=True)
            metrics.setdefault("rollout_executor_shutdown_seconds", []).append(time.time() - shutdown_start)
            self._finish_distributed_sandbox_rollout(sandbox_rollout_id)
            if not val:
                for group_id in range(env_groups):
                    if group_id in resolved_groups:
                        continue
                    self._resolve_buffer_group(
                        rollout_id=sandbox_rollout_id,
                        group_id=group_id,
                        promote=False,
                    )

        examples = [example for group_id in range(env_groups) for example in group_examples[group_id]]

        if return_examples:
            metrics.setdefault("rollout_collect_seconds", []).append(time.time() - rollout_start)
            return examples, self._aggregate_rollout_metrics(metrics, self._dataset_name())

        formulate_start = time.time()
        rollouts, formulate_metrics = self._formulate_rollouts(examples)
        metrics.setdefault("rollout_formulate_seconds", []).append(time.time() - formulate_start)
        metrics.setdefault("rollout_collect_seconds", []).append(time.time() - rollout_start)
        metrics.setdefault("rollout_submit_seconds", []).append(submit_seconds)
        metrics.setdefault("rollout_wait_seconds", []).append(wait_seconds)
        metrics.setdefault("rollout_future_result_seconds", []).append(future_result_seconds)
        metrics.setdefault("rollout_metrics_merge_seconds", []).append(metrics_merge_seconds)
        metrics.setdefault("rollout_finish_group_seconds", []).append(finish_group_seconds)
        for key, value in formulate_metrics.items():
            metrics.setdefault(key, []).append(value)
        rollouts.meta_info = {"metrics": self._aggregate_rollout_metrics(metrics, self._dataset_name())}
        return rollouts

    def _reterm_initial_group_quota(self, target_groups: int, group_size: int) -> int:
        initial_groups = int(getattr(self.config.recovery_env, "recovery_buffer_initial_groups", 6))
        recovery_groups = self._reterm_recovery_group_quota()
        if initial_groups < 1 or target_groups != initial_groups + recovery_groups or group_size != 16:
            raise ValueError("reterm requires (initial_groups + recovery_groups) groups x 16 attempts")
        return initial_groups

    def _reterm_recovery_group_quota(self) -> int:
        groups = int(getattr(self.config.recovery_env, "recovery_buffer_recovery_groups", 2))
        if groups < 1:
            raise ValueError("reterm recovery group quota must be positive")
        return groups

    def _collect_initial_group_examples(
        self,
        *,
        mode_cfg: Any,
        group_count: int,
        group_size: int,
        rollout_seed: int,
        sandbox_rollout_id: str,
        max_workers: int,
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        if self._steady_state_sampling_enabled(val=False):
            return self._rollout_steady_state(
                mode_cfg=mode_cfg,
                target_groups=group_count,
                group_size=group_size,
                rollout_seed=rollout_seed,
                val=False,
                sandbox_rollout_id=sandbox_rollout_id,
                max_workers_override=max_workers,
                return_examples=True,
            )
        return self._rollout_initial_groups(
            mode_cfg=mode_cfg,
            env_groups=group_count,
            group_size=group_size,
            rollout_seed=rollout_seed,
            val=False,
            sandbox_rollout_id=sandbox_rollout_id,
            max_workers_override=max_workers,
            return_examples=True,
        )

    def _rollout_buffered_recovery_batch(
        self,
        *,
        mode_cfg: Any,
        target_groups: int,
        group_size: int,
        rollout_seed: int,
    ) -> DataProto:
        """Collect the configured Initial and Recovery group quotas.

        The two lanes execute concurrently.  Group placement is node-affine:
        at 128 slots per node the Initial lane gets 96 and Recovery gets 32,
        on either one or two nodes, while optimizer normalization sees independent
        same-task groups.
        """

        initial_groups = self._reterm_initial_group_quota(target_groups, group_size)
        recovery_groups = self._reterm_recovery_group_quota()
        initial_slots, recovery_slots = self._reterm_lane_capacity()
        initial_rollout_id = self._new_sandbox_rollout_id(val=False, rollout_seed=rollout_seed)
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="reterm-lane") as lanes:
            initial_future = lanes.submit(
                self._collect_initial_group_examples,
                mode_cfg=mode_cfg,
                group_count=initial_groups,
                group_size=group_size,
                rollout_seed=rollout_seed,
                sandbox_rollout_id=initial_rollout_id,
                max_workers=initial_slots,
            )
            recovery_future = lanes.submit(
                self._collect_buffered_recovery_examples,
                group_size=group_size,
                rollout_seed=rollout_seed,
            )
            initial_examples, initial_metrics = initial_future.result()
            recovery_examples, recovery_metrics = recovery_future.result()

        recovery_group_ids = {row["group_id"] for row in recovery_examples}
        recovery_group_count = len(recovery_group_ids)
        if not 0 <= recovery_group_count <= recovery_groups:
            raise RuntimeError(
                "reterm Recovery lane returned an invalid accepted-group count: "
                f"{recovery_group_count}"
            )
        if len(recovery_examples) != recovery_group_count * group_size:
            raise RuntimeError(
                "reterm Recovery lane returned incomplete groups: "
                f"groups={recovery_group_count} examples={len(recovery_examples)} "
                f"group_size={group_size}"
            )

        fallback_group_count = recovery_groups - recovery_group_count
        fallback_examples: list[dict[str, Any]] = []
        fallback_metrics: dict[str, float] = {}
        if fallback_group_count:
            # Recovery states can legitimately disappear while this batch is
            # running: zero-variance groups are consumed, and invalid states
            # are not optimizer examples.  Do not abort the optimizer step.
            # Fill only the missing quota with ordinary Initial groups.
            fallback_rollout_id = self._new_sandbox_rollout_id(
                val=False,
                rollout_seed=rollout_seed,
            )
            raw_fallback_examples, fallback_metrics = self._collect_initial_group_examples(
                mode_cfg=mode_cfg,
                group_count=fallback_group_count,
                group_size=group_size,
                rollout_seed=rollout_seed,
                sandbox_rollout_id=fallback_rollout_id,
                max_workers=recovery_slots,
            )
            # Each Initial collection numbers groups and environments from
            # zero.  Namespace the fallback call so it can never collide with
            # the Initial groups already collected for this optimizer
            # step; otherwise GRPO would silently normalize unrelated tasks
            # together.
            for row in raw_fallback_examples:
                namespaced = dict(row)
                namespaced["group_id"] = (
                    f"buffer-fallback:{rollout_seed}:{row['group_id']}"
                )
                namespaced["env_id"] = f"buffer-fallback:{rollout_seed}:{row['env_id']}"
                if row.get("attempt_id") is not None:
                    namespaced["attempt_id"] = (
                        f"buffer-fallback:{rollout_seed}:{row['attempt_id']}"
                    )
                fallback_examples.append(namespaced)

        examples = [*initial_examples, *recovery_examples, *fallback_examples]
        rollouts, formulate_metrics = self._formulate_rollouts(examples)
        initial_train_examples = [*initial_examples, *fallback_examples]
        initial_train_rewards = [float(row.get("reward", 0.0)) for row in initial_train_examples]
        recovery_train_rewards = [float(row.get("reward", 0.0)) for row in recovery_examples]
        recovery_group_rewards: dict[str, list[float]] = {}
        for row in recovery_examples:
            recovery_group_rewards.setdefault(str(row["group_id"]), []).append(
                float(row.get("reward", 0.0))
            )
        recovery_group_successes = sorted(
            sum(reward > 0 for reward in rewards)
            for rewards in recovery_group_rewards.values()
        )
        metrics = {
            **{f"reterm/initial/{key}": value for key, value in initial_metrics.items()},
            **{f"reterm/recovery/{key}": value for key, value in recovery_metrics.items()},
            **{f"reterm/fallback_initial/{key}": value for key, value in fallback_metrics.items()},
            **formulate_metrics,
            "reterm/initial_train_groups": float(initial_groups + fallback_group_count),
            "reterm/recovery_train_groups": float(recovery_group_count),
            "reterm/fallback_initial_train_groups": float(fallback_group_count),
            "reterm/recovery_exhausted_fallback": float(fallback_group_count > 0),
            "reterm/initial_consume_on_spawn_enabled": 1.0,
            "reterm/recovery_cancelled_requeue_enabled": 1.0,
            "reterm/train_groups": float(target_groups),
            "reterm/initial_selected_success_rate": float(
                np.mean([reward > 0 for reward in initial_train_rewards])
            ),
            "reterm/initial_selected_reward_mean": float(np.mean(initial_train_rewards)),
        }
        initial_success_key = f"{self._dataset_name()}/success"
        initial_reward_key = f"{self._dataset_name()}/reward"
        if initial_success_key in initial_metrics:
            metrics["reterm/initial_all_sampled_success_rate"] = float(
                initial_metrics[initial_success_key]
            )
        if initial_reward_key in initial_metrics:
            metrics["reterm/initial_all_sampled_reward_mean"] = float(
                initial_metrics[initial_reward_key]
            )
        if recovery_train_rewards:
            metrics["reterm/recovery_selected_success_rate"] = float(
                np.mean([reward > 0 for reward in recovery_train_rewards])
            )
            metrics["reterm/recovery_selected_reward_mean"] = float(
                np.mean(recovery_train_rewards)
            )
            metrics["reterm/recovery_group_successes_min"] = float(
                min(recovery_group_successes)
            )
            metrics["reterm/recovery_group_successes_median"] = float(
                np.median(recovery_group_successes)
            )
            metrics["reterm/recovery_group_successes_max"] = float(
                max(recovery_group_successes)
            )
        rollouts.meta_info = {"metrics": metrics}
        return rollouts

    def _reterm_lane_capacity(self) -> tuple[int, int]:
        """Match node-local lane semaphores; preserve the 2 x 128 layout."""
        nodes = len(self._sandbox_workers)
        if nodes < 1:
            raise ValueError("reterm requires a task execution node")
        per_node = getattr(self, "_sandbox_per_node_limit", 128)
        recovery = max(1, per_node // 4)
        if recovery < 32:
            raise ValueError("reterm needs 32 Recovery slots per node for two 16-attempt candidates")
        return nodes * (per_node - recovery), nodes * recovery

    def _collect_buffered_recovery_examples(
        self,
        *,
        group_size: int,
        rollout_seed: int,
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        import ray

        accepted: list[list[dict[str, Any]]] = []
        target_groups = self._reterm_recovery_group_quota()
        metrics: dict[str, float] = {
            "target_groups": float(target_groups),
            "execution_slots": float(self._reterm_lane_capacity()[1]),
        }
        started = time.time()
        statuses_before = ray.get([worker.status.remote() for worker in self._sandbox_workers])
        available_before = [int(status["recovery_buffer"]["available"]) for status in statuses_before]
        metrics["buffer_available_before"] = float(sum(available_before))
        metrics["buffer_eligible_before"] = float(
            sum(
                int(entry["created_step"]) < int(rollout_seed)
                for status in statuses_before
                for entry in status["recovery_buffer"].get("entries", [])
            )
        )
        metrics["buffer_load_imbalance_before"] = float(max(available_before) - min(available_before))
        metrics["states_consumed"] = 0.0
        metrics["states_requeued"] = 0.0
        metrics["states_requeued_infra_or_invalid"] = 0.0
        metrics["states_requeued_completed_unselected"] = 0.0
        metrics["states_evicted_on_requeue"] = 0.0
        attempted_entry_ids: set[str] = set()
        sampling_round = 0
        while len(accepted) < target_groups:
            sampling_round += 1
            # Always launch two candidate states on each node. All candidates
            # share one global completion pool; there is no per-node winner.
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="reterm-node") as executor:
                lane_futures = [
                    executor.submit(
                        self._sample_buffered_recovery_candidates,
                        worker=self._sandbox_workers[node_index],
                        node_index=node_index,
                        lane_index=node_index,
                        group_size=group_size,
                        rollout_seed=rollout_seed,
                        candidate_count=2,
                        excluded_entry_ids=attempted_entry_ids,
                    )
                    for node_index in range(len(self._sandbox_workers))
                ]
                candidates = [candidate for future in lane_futures for candidate in future.result()]
            if not candidates:
                metrics["buffer_exhausted"] = 1.0
                break
            attempted_entry_ids.update(
                str(candidate["lease"]["entry_id"])
                for candidate in candidates
            )
            random.Random(rollout_seed * 1_000_003 + sampling_round).shuffle(candidates)

            consume: dict[tuple[int, str], bool] = {}
            requeue_reason: dict[tuple[int, str], str] = {}
            accepted_before = len(accepted)
            for candidate in candidates:
                node_index = int(candidate["node_index"])
                lease = candidate["lease"]
                group = candidate["group"]
                key = (node_index, str(lease["lease_id"]))
                if group is None:
                    metrics["infra_or_invalid_groups"] = metrics.get("infra_or_invalid_groups", 0.0) + 1.0
                    failure_reason = str(candidate.get("failure_reason") or "unknown")
                    metric_reason = re.sub(r"[^a-zA-Z0-9_]+", "_", failure_reason).strip("_")[:80]
                    metrics[f"infra_or_invalid_reason/{metric_reason or 'unknown'}"] = (
                        metrics.get(f"infra_or_invalid_reason/{metric_reason or 'unknown'}", 0.0)
                        + 1.0
                    )
                    print(
                        "[rl] buffered Recovery candidate is not trainable "
                        f"entry={candidate['lease'].get('entry_id')} "
                        f"task={candidate['lease'].get('task_id')} reason={failure_reason} "
                        f"detail={candidate.get('failure_detail', '')}",
                        flush=True,
                    )
                    consume[key] = False
                    requeue_reason[key] = "infra_or_invalid"
                    continue
                reward_std = self._group_reward_std(group)
                if reward_std == 0.0:
                    metric = "all_success_groups" if float(group[0]["reward"]) > 0 else "all_fail_groups"
                    metrics[metric] = metrics.get(metric, 0.0) + 1.0
                    consume[key] = True
                elif len(accepted) < target_groups:
                    accepted.append(group)
                    consume[key] = True
                else:
                    # Completed but not selected after the global target filled.
                    consume[key] = False
                    requeue_reason[key] = "completed_unselected"

            for candidate in candidates:
                node_index = int(candidate["node_index"])
                lease_id = str(candidate["lease"]["lease_id"])
                resolution = ray.get(
                    self._sandbox_workers[node_index].finish_recovery_state.remote(
                        lease_id=lease_id,
                        consume=consume[(node_index, lease_id)],
                    )
                )
                if consume[(node_index, lease_id)]:
                    metrics["states_consumed"] += 1.0
                else:
                    if bool(resolution.get("requeued", False)):
                        metrics["states_requeued"] += 1.0
                    if bool(resolution.get("evicted_on_requeue", False)):
                        metrics["states_evicted_on_requeue"] += 1.0
                    reason = requeue_reason.get((node_index, lease_id))
                    if reason == "infra_or_invalid":
                        metrics["states_requeued_infra_or_invalid"] += 1.0
                    elif reason == "completed_unselected":
                        metrics["states_requeued_completed_unselected"] += 1.0

            if len(accepted) == accepted_before and not any(consume.values()):
                # These entries remain at their original queue positions, but
                # are excluded from the remainder of this optimizer step. The
                # next loop therefore tries older, unseen states instead of
                # either spinning on the same failures or falling back after
                # only the first four candidates.
                metrics["infra_only_sampling_rounds"] = (
                    metrics.get("infra_only_sampling_rounds", 0.0) + 1.0
                )

        for group in accepted:
            for row in group:
                if self.trajectory_dir is not None:
                    self._save_trajectory(
                        dataset_name=str(row.get("dataset_name", self._dataset_name())),
                        episode=row,
                        benchmark_result=row.get("benchmark_result"),
                    )
        metrics["accepted_groups"] = float(len(accepted))
        metrics["unique_states_attempted"] = float(len(attempted_entry_ids))
        metrics["collect_seconds"] = time.time() - started
        statuses = ray.get([worker.status.remote() for worker in self._sandbox_workers])
        for node_index, status in enumerate(statuses):
            buffer_status = status["recovery_buffer"]
            metrics[f"node_{node_index}_buffer_available"] = float(buffer_status["available"])
            metrics[f"node_{node_index}_buffer_leased"] = float(buffer_status["leased"])
        available_after = [metrics[f"node_{i}_buffer_available"] for i in range(len(self._sandbox_workers))]
        metrics["buffer_load_imbalance_after"] = max(available_after) - min(available_after)
        return [row for group in accepted for row in group], metrics

    def _lease_buffer_state_for_node(
        self,
        *,
        target_node_index: int,
        rollout_seed: int,
        lane_index: int,
        excluded_entry_ids: set[str] | None = None,
    ) -> dict[str, Any] | None:
        """Lease a recent failure state from the worker's local queue."""
        import ray

        del lane_index
        with self._buffer_lease_lock:
            target = self._sandbox_workers[target_node_index]
            return ray.get(
                target.lease_recovery_state.remote(
                    max_created_step=int(rollout_seed) - 1,
                    excluded_entry_ids=sorted(excluded_entry_ids or ()),
                )
            )
    def _sample_buffered_recovery_candidates(
        self,
        *,
        worker: Any,
        node_index: int,
        lane_index: int,
        group_size: int,
        rollout_seed: int,
        candidate_count: int,
        excluded_entry_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        import ray

        leases = [
            self._lease_buffer_state_for_node(
                target_node_index=node_index,
                rollout_seed=rollout_seed,
                lane_index=lane_index,
                excluded_entry_ids=excluded_entry_ids,
            )
            for _ in range(int(candidate_count))
        ]
        leases = [lease for lease in leases if lease is not None]
        if not leases:
            return []
        groups: list[dict[str, Any]] = []
        rollout_ids: list[str] = []
        executor = ThreadPoolExecutor(max_workers=len(leases) * group_size)
        aborted = False
        try:
            pending_groups = []
            for candidate_index, lease in enumerate(leases):
                lease_id = str(lease["lease_id"])
                group_id = f"buffer:{lease['entry_id']}"
                rollout_id = (
                    f"buffer-recovery-{rollout_seed}-{lane_index}-{candidate_index}-"
                    f"{node_index}-{lease['entry_id']}"
                )
                rollout_ids.append(rollout_id)
                futures = [
                    executor.submit(
                        self._get_distributed_buffered_recovery,
                        worker,
                        rollout_id,
                        lease_id,
                        {
                            "env_id": node_index * 10_000 + candidate_index * group_size + sample_index,
                            "group_id": group_id,
                            "sample_index": sample_index,
                            "val": False,
                            "rollout_seed": rollout_seed,
                        },
                    )
                    for sample_index in range(group_size)
                ]
                pending_groups.append((lease, futures))

            # Submit every leased group's calls before waiting. Waiting inside
            # the submission loop serializes two groups into only 16 slots.
            for lease, futures in pending_groups:
                try:
                    group = [future.result() for future in futures]
                except Exception as exc:
                    failure_category = self._infrastructure_failure_category(exc)
                    if failure_category is None:
                        raise
                    groups.append({
                        "node_index": node_index,
                        "lease": lease,
                        "group": None,
                        "failure_reason": f"infrastructure_{failure_category}",
                        "failure_detail": f"{type(exc).__name__}: {exc}"[-2000:],
                    })
                    continue
                trainable = all(self._is_trainable_example(row) for row in group)
                failure_reason = None
                failure_detail = None
                if not trainable:
                    discard_reasons = sorted({
                        str(row.get("discard_reason"))
                        for row in group
                        if row.get("discard_reason")
                    })
                    if discard_reasons:
                        failure_reason = "discarded_rollout"
                        failure_detail = ",".join(discard_reasons)
                        discard_details = sorted({
                            str(row.get("discard_detail"))
                            for row in group
                            if row.get("discard_detail")
                        })
                        if discard_details:
                            failure_detail += ": " + " | ".join(discard_details)
                    elif any(self._is_over_context_example(row) for row in group):
                        failure_reason = "over_context"
                    else:
                        failure_reason = "non_trainable_provenance"
                groups.append(
                    {
                        "node_index": node_index,
                        "lease": lease,
                        "group": group if trainable else None,
                        "failure_reason": failure_reason,
                        "failure_detail": failure_detail,
                    }
                )
        except BaseException:
            aborted = True
            # Both groups may still be using their snapshots. Cancel first,
            # then drain the executor before returning any buffer leases.
            for rollout_id in rollout_ids:
                try:
                    ray.get(worker.cancel_rollout.remote(rollout_id))
                except Exception as cancel_error:
                    print(f"[rl] Recovery cancellation failed: {cancel_error}", flush=True)
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
            try:
                for rollout_id in rollout_ids:
                    ray.get(worker.finish_rollout.remote(rollout_id))
            except BaseException:
                aborted = True
                raise
            finally:
                if aborted:
                    for lease in leases:
                        ray.get(worker.finish_recovery_state.remote(
                            lease_id=str(lease["lease_id"]), consume=False,
                        ))
        return groups

    def _rollout_steady_state(
        self,
        *,
        mode_cfg: Any,
        target_groups: int,
        group_size: int,
        rollout_seed: int,
        val: bool,
        sandbox_rollout_id: str | None = None,
        max_workers_override: int | None = None,
        return_examples: bool = False,
    ) -> DataProto | tuple[list[dict[str, Any]], dict[str, float]]:
        """Continuously sample candidate groups until the train queue is full.

        ``target_groups`` controls the optimizer batch.  The executor worker
        count and the prefetched candidate-group pool independently control
        rollout concurrency.  Only complete, trainable groups with non-zero
        reward variance enter the optimizer queue.  All candidates in one call
        are generated by the same frozen rollout policy; the queue is never
        carried across optimizer steps.
        """

        total = target_groups * group_size
        worker_env_key = "RETERM_RL_VAL_ROLLOUT_WORKERS" if val else "RETERM_RL_ROLLOUT_WORKERS"
        configured_workers = int(getattr(mode_cfg, "max_workers", total))
        requested_workers = int(os.environ.get(worker_env_key, str(configured_workers)))
        max_workers = self._initial_worker_limit(
            requested_workers if max_workers_override is None else int(max_workers_override)
        )
        prefetch_factor = self._steady_state_prefetch_factor()
        pipeline_groups = max(
            target_groups,
            int(math.ceil(max_workers * prefetch_factor / group_size)),
        )
        max_candidate_groups = self._steady_state_max_candidate_groups(
            target_groups=target_groups,
            pipeline_groups=pipeline_groups,
        )

        accepted_groups: list[list[dict[str, Any]]] = []
        group_examples: dict[int, list[dict[str, Any]]] = {}
        group_candidates: dict[int, list[dict[str, Any]]] = {}
        task_by_group: dict[int, str] = {}
        attempts_by_group: dict[int, int] = {}
        retired_groups: set[int] = set()
        resolved_buffer_groups: set[int] = set()
        metrics: dict[str, list[float]] = {
            "rollout_target_train_groups": [float(target_groups)],
            "rollout_target_train_attempts": [float(total)],
            "rollout_sampling_workers": [float(max_workers)],
            "rollout_pipeline_groups": [float(pipeline_groups)],
        }
        executor = ThreadPoolExecutor(max_workers=max_workers)
        cancel_event = threading.Event()
        pending: dict[Any, tuple[int, str]] = {}
        next_group_id = 0
        started_groups = 0
        complete_groups = 0
        complete_group_attempts = 0
        complete_group_successes = 0
        complete_group_reward_sum = 0.0
        complete_zero_std_groups = 0
        complete_zero_std_all_success = 0
        complete_zero_std_all_fail = 0
        rollout_start = time.time()
        submit_seconds = 0.0
        wait_seconds = 0.0
        future_result_seconds = 0.0
        metrics_merge_seconds = 0.0
        finish_group_seconds = 0.0

        def submit_group(group_id: int) -> None:
            nonlocal submit_seconds
            if group_id in retired_groups or group_candidates[group_id]:
                return
            submit_start = time.time()
            task_id = task_by_group[group_id]
            missing = group_size - len(group_examples[group_id])
            for _ in range(missing):
                sample_index = attempts_by_group[group_id]
                attempts_by_group[group_id] += 1
                future = self._submit_episode(
                    executor,
                    sandbox_rollout_id=sandbox_rollout_id,
                    task_id=task_id,
                    env_id=group_id * group_size + sample_index,
                    group_id=group_id,
                    sample_index=sample_index,
                    val=val,
                    rollout_seed=rollout_seed,
                    cancel_event=cancel_event,
                    buffer_rollout_id=sandbox_rollout_id,
                )
                pending[future] = (group_id, task_id)
            submit_seconds += time.time() - submit_start

        def spawn_group() -> bool:
            nonlocal next_group_id, started_groups
            if started_groups >= max_candidate_groups:
                return False
            group_id = next_group_id
            next_group_id += 1
            started_groups += 1
            task_by_group[group_id] = self._next_task_id(val=val)
            attempts_by_group[group_id] = 0
            group_examples[group_id] = []
            group_candidates[group_id] = []
            metrics.setdefault("rollout_candidate_groups_started", []).append(1.0)
            submit_group(group_id)
            return True

        def retire_group(group_id: int) -> None:
            if group_id in retired_groups:
                return
            retired_groups.add(group_id)

        def resolve_buffer_group(group_id: int, *, promote: bool) -> None:
            if group_id in resolved_buffer_groups:
                return
            resolution = self._resolve_buffer_group(
                rollout_id=sandbox_rollout_id,
                group_id=group_id,
                promote=promote,
            )
            metric = (
                "rollout_buffer_states_promoted"
                if promote and resolution is not None
                else "rollout_buffer_groups_without_promoted_state"
            )
            metrics.setdefault(metric, []).append(1.0)
            resolved_buffer_groups.add(group_id)

        def save_and_measure(examples: list[dict[str, Any]]) -> None:
            for example in examples:
                dataset_name = str(example.get("dataset_name", self._dataset_name()))
                reward_metric = f"{dataset_name}/reward"
                metrics.setdefault(reward_metric, []).append(float(example["reward"]))
                metrics.setdefault("rollout_final_reward", []).append(float(example["reward"]))
                metrics.setdefault("rollout_attempt_turns", []).append(
                    float(example.get("attempt_turns", 0.0))
                )
            for example in examples:
                dataset_name = str(example.get("dataset_name", self._dataset_name()))
                if getattr(self, "trajectory_dir", None) is not None and not example.get("trajectory_saved"):
                    save_start = time.time()
                    self._save_trajectory(
                        dataset_name=dataset_name,
                        episode=example,
                        benchmark_result=example.get("benchmark_result"),
                    )
                    example["trajectory_saved"] = True
                    metrics.setdefault("rollout_save_trajectory_seconds", []).append(
                        time.time() - save_start
                    )

        def finish_group_if_ready(group_id: int) -> None:
            nonlocal finish_group_seconds
            nonlocal complete_groups, complete_group_attempts
            nonlocal complete_group_successes, complete_group_reward_sum
            nonlocal complete_zero_std_groups
            nonlocal complete_zero_std_all_success, complete_zero_std_all_fail
            finish_start = time.time()
            try:
                candidates = group_candidates[group_id]
                if len(group_examples[group_id]) + len(candidates) < group_size:
                    return
                trainable = list(group_examples[group_id])
                non_trainable_count = 0
                over_context_count = 0
                for result in candidates:
                    check_start = time.time()
                    is_trainable = self._is_trainable_example(result, allow_discard=val)
                    metrics.setdefault("rollout_trainable_check_seconds", []).append(time.time() - check_start)
                    if is_trainable:
                        trainable.append(result)
                    else:
                        non_trainable_count += 1
                        over_context_start = time.time()
                        over_context_count += int(self._is_over_context_example(result))
                        metrics.setdefault("rollout_over_context_check_seconds", []).append(
                            time.time() - over_context_start
                        )

                if len(trainable) < group_size:
                    previous_trainable_count = len(group_examples[group_id])
                    newly_trainable_count = len(trainable) - previous_trainable_count
                    metrics.setdefault("rollout_non_trainable_refill_groups", []).append(1.0)
                    metrics.setdefault("rollout_non_trainable_refill", []).append(float(non_trainable_count))
                    if over_context_count:
                        metrics.setdefault("rollout_over_context_refill", []).append(float(over_context_count))
                    group_examples[group_id] = trainable
                    group_candidates[group_id] = []
                    if (
                        newly_trainable_count == 0
                        or attempts_by_group[group_id] >= self._max_refill_attempts_per_task(group_size)
                    ):
                        metrics.setdefault("rollout_non_trainable_task_replacements", []).append(1.0)
                        retire_group(group_id)
                        if len(accepted_groups) < target_groups:
                            spawn_group()
                    else:
                        submit_group(group_id)
                    return

                reward_std = self._group_reward_std(trainable)
                rewards = [
                    float(example["reward"])
                    for example in trainable
                ]
                root_rewards = [
                    float(example.get("root_metric_reward", example["reward"]))
                    for example in trainable
                ]
                # Dashboard denominator: every fully assembled group of
                # ``group_size`` attempts. This includes optimizer-accepted
                # groups and complete zero-variance groups, but excludes
                # unfinished/in-flight candidate groups.
                complete_groups += 1
                complete_group_attempts += len(root_rewards)
                complete_group_successes += sum(reward > 0.0 for reward in root_rewards)
                complete_group_reward_sum += sum(root_rewards)
                if group_size > 1 and reward_std == 0.0:
                    complete_zero_std_groups += 1
                    metrics.setdefault("rollout_zero_std_groups", []).append(1.0)
                    metrics.setdefault("rollout_zero_std", []).append(1.0)
                    all_fail = bool(rewards and rewards[0] <= 0.0)
                    if all_fail:
                        complete_zero_std_all_fail += 1
                        metrics.setdefault("rollout_zero_std_all_fail", []).append(1.0)
                    else:
                        complete_zero_std_all_success += 1
                        metrics.setdefault("rollout_zero_std_all_success", []).append(1.0)
                    metrics.setdefault("rollout_candidate_groups_rejected", []).append(1.0)
                    print(
                        "[rl] steady-state sampler rejected zero-std group "
                        f"group={group_id} task={task_by_group[group_id]} rewards={rewards}",
                        flush=True,
                    )
                    resolve_buffer_group(group_id, promote=False)
                    retire_group(group_id)
                    if len(accepted_groups) < target_groups:
                        spawn_group()
                    return

                save_and_measure(trainable)
                resolve_buffer_group(group_id, promote=True)
                accepted_groups.append(trainable)
                metrics.setdefault("rollout_candidate_groups_accepted", []).append(1.0)
                metrics.setdefault("rollout_accepted_group_reward_std", []).append(reward_std)
                print(
                    "[rl] steady-state sampler queued trainable group "
                    f"group={group_id} task={task_by_group[group_id]} "
                    f"accepted={len(accepted_groups)}/{target_groups} reward_std={reward_std:.6f}",
                    flush=True,
                )
                retire_group(group_id)
                if len(accepted_groups) < target_groups:
                    spawn_group()
            finally:
                finish_group_seconds += time.time() - finish_start

        for _ in range(pipeline_groups):
            if not spawn_group():
                break

        try:
            while pending and len(accepted_groups) < target_groups:
                wait_start = time.time()
                done, _not_done = wait(pending, return_when=FIRST_COMPLETED)
                wait_seconds += time.time() - wait_start
                for future in done:
                    group_id, future_task_id = pending.pop(future)
                    future_result_start = time.time()
                    result = future.result()
                    future_result_seconds += time.time() - future_result_start
                    metrics_merge_start = time.time()
                    for key, value in result["metrics"].items():
                        metrics.setdefault(key, []).append(float(value))
                    metrics_merge_seconds += time.time() - metrics_merge_start

                    if group_id in retired_groups or future_task_id != task_by_group[group_id]:
                        metrics.setdefault("rollout_stale", []).append(1.0)
                        continue
                    group_candidates[group_id].append(result)
                    finish_group_if_ready(group_id)
                    if len(accepted_groups) >= target_groups:
                        cancel_event.set()
                        break

            if len(accepted_groups) < target_groups:
                raise RuntimeError(
                    "steady-state sampler exhausted its candidate budget before collecting "
                    f"{target_groups} non-zero-variance groups: accepted={len(accepted_groups)} "
                    f"started={started_groups} max_candidate_groups={max_candidate_groups}"
                )
        finally:
            cancel_event.set()
            unfinished_group_ids = [
                group_id for group_id in sorted(task_by_group) if group_id not in retired_groups
            ]
            metrics.setdefault("rollout_cancelled_candidate_groups", []).append(
                float(len(unfinished_group_ids))
            )
            outstanding = list(pending)
            for future in outstanding:
                future.cancel()
            self._cancel_distributed_sandbox_rollout(sandbox_rollout_id)
            shutdown_start = time.time()
            executor.shutdown(wait=True, cancel_futures=True)
            metrics.setdefault("rollout_executor_shutdown_seconds", []).append(time.time() - shutdown_start)
            self._finish_distributed_sandbox_rollout(sandbox_rollout_id)
            # Release candidate groups that remained in flight at cutoff.
            for group_id in list(task_by_group):
                resolve_buffer_group(group_id, promote=False)
                retire_group(group_id)

        examples = [example for group in accepted_groups[:target_groups] for example in group]
        metrics.setdefault("rollout_collect_seconds", []).append(time.time() - rollout_start)
        metrics.setdefault("rollout_submit_seconds", []).append(submit_seconds)
        metrics.setdefault("rollout_wait_seconds", []).append(wait_seconds)
        metrics.setdefault("rollout_future_result_seconds", []).append(future_result_seconds)
        metrics.setdefault("rollout_metrics_merge_seconds", []).append(metrics_merge_seconds)
        metrics.setdefault("rollout_finish_group_seconds", []).append(finish_group_seconds)
        metrics.setdefault("rollout_candidate_groups_started_total", []).append(float(started_groups))
        metrics.setdefault("rollout_candidate_groups_accepted_total", []).append(float(len(accepted_groups)))
        dataset_name = self._dataset_name()
        complete_denominator = float(complete_groups) if complete_groups else 1.0
        attempt_denominator = float(complete_group_attempts) if complete_group_attempts else 1.0
        metrics.setdefault(f"{dataset_name}/complete_groups_total", []).append(float(complete_groups))
        metrics.setdefault(f"{dataset_name}/complete_groups_trainable_total", []).append(
            float(len(accepted_groups))
        )
        metrics.setdefault(f"{dataset_name}/complete_groups_zero_std_total", []).append(
            float(complete_zero_std_groups)
        )
        metrics.setdefault(f"{dataset_name}/complete_groups_all_success_total", []).append(
            float(complete_zero_std_all_success)
        )
        metrics.setdefault(f"{dataset_name}/complete_groups_all_fail_total", []).append(
            float(complete_zero_std_all_fail)
        )
        metrics.setdefault(f"{dataset_name}/complete_groups_success_rate", []).append(
            float(complete_group_successes) / attempt_denominator
        )
        all_complete_groups_reward_mean = complete_group_reward_sum / attempt_denominator
        # Unfiltered sampling quality: include accepted mixed-reward groups and
        # rejected all-success/all-failure groups. Incomplete speculative groups
        # cancelled after the optimizer batch fills are intentionally excluded.
        metrics.setdefault(f"{dataset_name}/all_complete_groups_reward_mean", []).append(
            all_complete_groups_reward_mean
        )
        metrics.setdefault(f"{dataset_name}/complete_groups_reward_mean", []).append(
            all_complete_groups_reward_mean
        )
        metrics.setdefault(f"{dataset_name}/complete_groups_trainable_rate", []).append(
            float(len(accepted_groups)) / complete_denominator
        )
        metrics.setdefault(f"{dataset_name}/complete_groups_all_success_rate", []).append(
            float(complete_zero_std_all_success) / complete_denominator
        )
        metrics.setdefault(f"{dataset_name}/complete_groups_all_fail_rate", []).append(
            float(complete_zero_std_all_fail) / complete_denominator
        )
        if return_examples:
            return examples, self._aggregate_rollout_metrics(metrics, dataset_name)
        formulate_start = time.time()
        rollouts, formulate_metrics = self._formulate_rollouts(examples)
        metrics.setdefault("rollout_formulate_seconds", []).append(time.time() - formulate_start)
        for key, value in formulate_metrics.items():
            metrics.setdefault(key, []).append(value)
        rollouts.meta_info = {"metrics": self._aggregate_rollout_metrics(metrics, self._dataset_name())}
        return rollouts

    def _steady_state_sampling_enabled(self, *, val: bool) -> bool:
        if val or not self._active_sampling_enabled():
            return False
        active_sampling = getattr(self.config.recovery_env, "active_sampling", {})
        try:
            steady_state = active_sampling.get("steady_state", {})
        except AttributeError:
            steady_state = getattr(active_sampling, "steady_state", {})
        try:
            enabled = steady_state.get("enabled", False)
        except AttributeError:
            enabled = getattr(steady_state, "enabled", False)
        return bool(enabled)

    def _steady_state_prefetch_factor(self) -> float:
        active_sampling = getattr(self.config.recovery_env, "active_sampling", {})
        try:
            steady_state = active_sampling.get("steady_state", {})
        except AttributeError:
            steady_state = getattr(active_sampling, "steady_state", {})
        try:
            value = steady_state.get("prefetch_factor", 2.0)
        except AttributeError:
            value = getattr(steady_state, "prefetch_factor", 2.0)
        return max(1.0, float(value))

    def _steady_state_max_candidate_groups(self, *, target_groups: int, pipeline_groups: int) -> int:
        active_sampling = getattr(self.config.recovery_env, "active_sampling", {})
        try:
            steady_state = active_sampling.get("steady_state", {})
        except AttributeError:
            steady_state = getattr(active_sampling, "steady_state", {})
        try:
            configured = steady_state.get("max_candidate_groups")
        except AttributeError:
            configured = getattr(steady_state, "max_candidate_groups", None)
        default = max(pipeline_groups, target_groups * 16)
        return max(pipeline_groups, int(configured if configured is not None else default))

    @classmethod
    def _aggregate_rollout_metrics(
        cls,
        metrics: dict[str, list[float]],
        dataset_name: str,
    ) -> dict[str, float]:
        aggregated = {key: float(np.mean(values)) if values else 0.0 for key, values in metrics.items()}
        failure_key = f"{dataset_name}/infrastructure_failures"
        failure_values = metrics.get(failure_key, [])
        if failure_values:
            failure_count = float(sum(failure_values))
            episode_count = float(len(failure_values))
            aggregated[failure_key] = failure_count / episode_count
            aggregated[f"{dataset_name}/infrastructure_failure_rate"] = failure_count / episode_count
            aggregated[f"{dataset_name}/infrastructure_failure_count"] = failure_count
            aggregated[f"{dataset_name}/rollout_episode_count"] = episode_count
            for category in cls._INFRASTRUCTURE_FAILURE_CATEGORIES:
                category_key = f"{dataset_name}/infrastructure_{category}"
                category_values = metrics.get(category_key, [])
                category_count = float(sum(category_values))
                aggregated[category_key] = category_count / episode_count
                aggregated[f"{category_key}_count"] = category_count
        return aggregated

    def _replace_group_task(
        self,
        *,
        group_id: int,
        task_by_group: dict[int, str],
        attempts_by_group: dict[int, int],
        replacements_by_group: dict[int, int],
        group_examples: dict[int, list[dict[str, Any]]],
        group_candidates: dict[int, list[dict[str, Any]]],
        max_replacement_tasks: int,
        val: bool,
        reason: str,
    ) -> bool:
        if replacements_by_group[group_id] >= max_replacement_tasks:
            return False
        replacements_by_group[group_id] += 1
        old_task = task_by_group[group_id]
        # Validation must keep the exact held-out task set. Retry the same task
        # instead of silently replacing it with another validation example.
        task_by_group[group_id] = old_task if val else self._next_task_id(val=False)
        attempts_by_group[group_id] = 0
        group_examples[group_id] = []
        group_candidates[group_id] = []
        print(
            "[rl] replacing rollout group task "
            f"group={group_id} old_task={old_task} new_task={task_by_group[group_id]} reason={reason}"
        )
        return True

    def _max_replacement_tasks(self) -> int:
        env_override = os.environ.get("RETERM_RL_MAX_REPLACEMENT_TASKS")
        if env_override is not None:
            return max(0, int(env_override))
        active_sampling = getattr(self.config.recovery_env, "active_sampling", {})
        try:
            value = active_sampling.get("max_replacement_rounds", 3)
        except AttributeError:
            value = getattr(active_sampling, "max_replacement_rounds", 3)
        return max(0, int(value))

    def _active_sampling_enabled(self) -> bool:
        active_sampling = getattr(self.config.recovery_env, "active_sampling", {})
        try:
            enabled = active_sampling.get("enabled", False)
        except AttributeError:
            enabled = getattr(active_sampling, "enabled", False)
        return bool(enabled)

    @staticmethod
    def _max_refill_attempts_per_task(group_size: int) -> int:
        max_refill_batches = max(1, int(os.environ.get("RETERM_RL_MAX_REFILL_BATCHES_PER_TASK", "2")))
        return group_size * max_refill_batches

    def _group_reward_std(self, examples: list[dict[str, Any]]) -> float:
        if not examples:
            return 0.0
        rewards = np.array(
            [float(example["reward"]) for example in examples],
            dtype=np.float64,
        )
        return float(np.std(rewards))

    def _next_task_ids(self, count: int, *, val: bool) -> list[str]:
        with self._task_stream_lock:
            return [self._next_task_id_locked(val=val) for _ in range(count)]

    def _next_task_id(self, *, val: bool) -> str:
        with self._task_stream_lock:
            return self._next_task_id_locked(val=val)

    def _next_task_id_locked(self, *, val: bool) -> str:
        stream = self._task_streams[val]
        if stream["cursor"] >= len(stream["order"]):
            self._reshuffle_task_stream_locked(val=val)
        task_id = stream["order"][stream["cursor"]]
        stream["cursor"] += 1
        return task_id

    def _reshuffle_task_stream_locked(self, *, val: bool) -> None:
        stream = self._task_streams[val]
        stream["epoch"] += 1
        order = list(self.val_task_ids if val else self.train_task_ids)
        seed_offset = 10_000 if val else 0
        rng = random.Random(self._task_sampling_seed + seed_offset + stream["epoch"])
        rng.shuffle(order)
        stream["order"] = order
        stream["cursor"] = 0

    def state_dict(self) -> dict[str, Any]:
        with self._task_stream_lock:
            return {
                "version": 1,
                "task_sampling_seed": self._task_sampling_seed,
                "train_task_ids": list(self.train_task_ids),
                "val_task_ids": list(self.val_task_ids),
                "task_streams": {
                    "train": self._copy_task_stream(self._task_streams[False]),
                    "val": self._copy_task_stream(self._task_streams[True]),
                },
            }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not state:
            return
        if int(state.get("task_sampling_seed", self._task_sampling_seed)) != self._task_sampling_seed:
            raise ValueError("checkpoint task sampler seed does not match current config")
        saved_train_task_ids = state.get("train_task_ids")
        if saved_train_task_ids is not None and list(saved_train_task_ids) != list(self.train_task_ids):
            raise ValueError("checkpoint training task list does not match current dataset")
        saved_val_task_ids = state.get("val_task_ids")
        if saved_val_task_ids is not None and list(saved_val_task_ids) != list(self.val_task_ids):
            raise ValueError("checkpoint validation task list does not match current dataset")
        streams = state.get("task_streams") or {}
        with self._task_stream_lock:
            self._task_streams = {
                False: self._normalize_task_stream(streams.get("train", {})),
                True: self._normalize_task_stream(streams.get("val", {})),
            }

    @staticmethod
    def _copy_task_stream(stream: dict[str, Any]) -> dict[str, Any]:
        return {
            "epoch": int(stream.get("epoch", -1)),
            "cursor": int(stream.get("cursor", 0)),
            "order": list(stream.get("order", [])),
        }

    @staticmethod
    def _normalize_task_stream(stream: dict[str, Any]) -> dict[str, Any]:
        order = list(stream.get("order", []))
        cursor = int(stream.get("cursor", 0))
        if cursor < 0 or cursor > len(order):
            raise ValueError(f"invalid task stream cursor={cursor} for order length={len(order)}")
        return {
            "epoch": int(stream.get("epoch", -1)),
            "cursor": cursor,
            "order": order,
        }

    def _task_seed(self) -> int:
        env_seed = os.environ.get("RETERM_RL_TASK_SEED")
        if env_seed is not None:
            return int(env_seed)
        trainer_cfg = getattr(self.config, "trainer", None)
        if trainer_cfg is not None:
            try:
                seed = trainer_cfg.get("seed")
            except AttributeError:
                seed = getattr(trainer_cfg, "seed", None)
            if seed is not None:
                return int(seed)
        return 42

    def _run_one_episode(
        self,
        *,
        task_id: str,
        env_id: int,
        group_id: int,
        sample_index: int,
        val: bool,
        rollout_seed: int,
        cancel_event: threading.Event | None = None,
        buffer_rollout_id: str | None = None,
    ) -> dict[str, Any]:
        wall_limit = self._rollout_wall_time_limit_seconds()
        start_time = time.monotonic()
        dataset = None
        cancel_watcher_stop = threading.Event()
        cancel_watcher: threading.Thread | None = None
        dataset_name = self._dataset_name()
        buffer_candidate: dict[str, Any] | None = None
        self._generation_seed.value = self._stable_generation_seed(
            rollout_seed=rollout_seed,
            task_id=task_id,
            sample_index=sample_index,
            val=val,
        )
        self._generation_turn.value = 0
        self._rollout_cancel_event.value = cancel_event
        try:
            if cancel_event is not None and cancel_event.is_set():
                raise RolloutCancelledError("speculative rollout cancelled before environment setup")
            dataset = self._make_dataset(val=val)
            # Pass cancellation to environment setup and task execution.
            dataset.rollout_cancel_event = cancel_event
            dataset_name = dataset.name
            if cancel_event is not None:
                cancel_watcher = threading.Thread(
                    target=self._watch_dataset_cancellation,
                    args=(dataset, cancel_event, cancel_watcher_stop),
                    name=f"rollout-cancel-{task_id}-{env_id}",
                    daemon=True,
                )
                cancel_watcher.start()
            agent_cfg = TerminalPolicyAgentConfig(**_to_plain_dict(self.config.recovery_env.agent))
            if wall_limit is not None:
                agent_cfg.max_wall_time_seconds = wall_limit
            agent = TerminalPolicyAgent(
                policy=self._generate_response,
                config=agent_cfg,
                deadline_setter=self._set_rollout_deadline,
                cancel_event=cancel_event,
            )
            runner = AttemptRunner(
                benchmark=dataset,
                agent=agent,
                model_name=str(self.config.actor_rollout_ref.model.path),
            )
            attempt_start = time.time()
            result = runner.run_initial(task_id)
            messages_start = time.time()
            candidate_messages, candidate_start_index = self._messages_and_current_attempt_start(result)
            messages_elapsed = time.time() - messages_start
            if (
                not val
                and not result.success
                and result.attempts
                and result.attempts[0].status is not AttemptStatus.ERROR
                and result.attempts[0].state_after is not None
                and self._is_trainable_example(
                    {
                        "task_id": task_id,
                        "env_id": env_id,
                        "messages": candidate_messages,
                        "training_start_index": candidate_start_index,
                    }
                )
            ):
                if not buffer_rollout_id:
                    raise RuntimeError("initial rollout is missing its rollout id")
                detach = getattr(dataset, "detach_snapshot", None)
                if not callable(detach):
                    raise RuntimeError("dataset does not support snapshot ownership transfer")
                detach(result.attempts[0].state_after)
                buffer_candidate = self._recovery_buffer.offer_provisional(
                    rollout_id=buffer_rollout_id,
                    task_id=task_id,
                    initial_attempt=result.attempts[0],
                    created_step=int(rollout_seed),
                    source_group_id=group_id,
                )
            attempt_elapsed = time.time() - attempt_start
        except RolloutCancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            # Closing a live VM intentionally interrupts whatever shell or
            # verifier call is blocked inside the attempt. Preserve that as
            # speculative cancellation, not as a sampled infra-failure row.
            if cancel_event is not None and cancel_event.is_set():
                raise RolloutCancelledError(
                    f"speculative rollout cancelled during environment operation: {task_id}"
                ) from exc
            print(f"[rl] task {dataset_name}/{task_id} failed during rollout: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            failure_category = self._infrastructure_failure_category(exc)
            if failure_category is None:
                print(
                    "[rl] aborting rollout step because the exception is not a recognized "
                    f"infrastructure failure: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                raise
            assistant_content = "Rollout unavailable because the execution infrastructure failed."
            assistant_token_ids = self.tokenizer.encode(assistant_content, add_special_tokens=False)
            episode = {
                "env_id": env_id,
                "group_id": group_id,
                "task_id": task_id,
                "messages": [
                    {"role": "system", "content": "ReTerm rollout failed before completion."},
                    {"role": "user", "content": f"Task {task_id} could not be prepared or executed."},
                    {
                        "role": "assistant",
                        "content": assistant_content,
                        "token_ids": assistant_token_ids,
                        "log_probs": [0.0] * len(assistant_token_ids),
                    },
                ],
                "current_attempt_start_index": 2,
                "training_start_index": 2,
                "attempt_turns": 0,
                "reward": 0.0,
                "discard_reason": f"infrastructure_{failure_category}",
                "metric_valid": False,
                "error_type": type(exc).__name__,
                "failure_category": failure_category,
                "metrics": {
                    f"{dataset_name}/success": 0.0,
                    f"{dataset_name}/reward": 0.0,
                    f"{dataset_name}/rollout_errors": 1.0,
                    **{
                        f"{dataset_name}/{key}": value
                        for key, value in self._infrastructure_metrics(failure_category).items()
                    },
                },
            }
            self._save_trajectory(
                dataset_name=dataset_name,
                episode=episode,
                error=f"{type(exc).__name__}: {exc}",
            )
            return episode
        finally:
            cancel_watcher_stop.set()
            if dataset is not None:
                dataset.close()
            if cancel_watcher is not None:
                cancel_watcher.join(timeout=5.0)
            if wall_limit is not None:
                self._rollout_deadline.value = None
            self._generation_seed.value = None
            self._generation_turn.value = 0
            self._rollout_cancel_event.value = None

        reward = self._episode_reward(result)
        messages, current_attempt_start_index = candidate_messages, candidate_start_index
        training_start_index = current_attempt_start_index
        attempt_turns = len(result.attempts[-1].agent_result.metadata.get("turns", []))
        result_metrics = self._result_metrics(result, reward=reward)
        result_metrics.update(self._infrastructure_metrics())
        result_metrics["rollout_wall_time_seconds"] = float(time.monotonic() - start_time)
        result_metrics["rollout_attempt_seconds"] = attempt_elapsed
        result_metrics["rollout_messages_seconds"] = messages_elapsed
        episode = {
            "env_id": env_id,
            "group_id": group_id,
            "task_id": task_id,
            "messages": messages,
            "current_attempt_start_index": current_attempt_start_index,
            "training_start_index": training_start_index,
            "attempt_turns": attempt_turns,
            "reward": reward,
            "root_metric_reward": float(result.success),
            "metric_valid": True,
            "dataset_name": dataset.name,
            "benchmark_result": result,
            "metrics": {f"{dataset.name}/{key}": value for key, value in result_metrics.items()},
        }
        if buffer_candidate is not None:
            episode["buffer_candidate"] = buffer_candidate
        return episode

    def _run_buffered_recovery_episode(
        self,
        *,
        lease_id: str,
        env_id: int,
        group_id: str,
        sample_index: int,
        val: bool,
        rollout_seed: int,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        """Restore one buffered failure and sample one Recovery-only attempt."""

        entry = self._recovery_buffer.get_lease(lease_id)
        dataset = None
        watcher_stop = threading.Event()
        watcher: threading.Thread | None = None
        wall_limit = self._rollout_wall_time_limit_seconds()
        self._generation_seed.value = self._stable_generation_seed(
            rollout_seed=rollout_seed,
            task_id=f"buffer:{entry.entry_id}",
            sample_index=sample_index,
            val=val,
        )
        self._generation_turn.value = 0
        self._rollout_cancel_event.value = cancel_event
        try:
            if cancel_event is not None and cancel_event.is_set():
                raise RolloutCancelledError("buffered Recovery cancelled before restore")
            dataset = self._make_dataset(val=val)
            dataset.rollout_cancel_event = cancel_event
            if cancel_event is not None:
                watcher = threading.Thread(
                    target=self._watch_dataset_cancellation,
                    args=(dataset, cancel_event, watcher_stop),
                    daemon=True,
                )
                watcher.start()
            agent_cfg = TerminalPolicyAgentConfig(**_to_plain_dict(self.config.recovery_env.agent))
            if wall_limit is not None:
                agent_cfg.max_wall_time_seconds = wall_limit
            agent = TerminalPolicyAgent(
                policy=self._generate_response,
                config=agent_cfg,
                deadline_setter=self._set_rollout_deadline,
                cancel_event=cancel_event,
            )
            runner = AttemptRunner(
                benchmark=dataset,
                agent=agent,
                model_name=str(self.config.actor_rollout_ref.model.path),
            )
            benchmark_result = runner.run_recovery(entry.task_id, entry.initial_attempt)
            attempt = benchmark_result.attempts[-1]
            success = benchmark_result.success
            messages = attempt.agent_result.metadata.get("messages")
            start_index = attempt.agent_result.metadata.get("current_attempt_start_index")
            if not isinstance(messages, list) or not isinstance(start_index, int):
                raise ValueError("buffered Recovery attempt lacks official Qwen message provenance")
            reward = float(success)
            loss_weight = float(getattr(self.config.recovery_env, "recovery_loss_weight", 1.0))
            if not math.isfinite(loss_weight) or loss_weight < 0.0:
                raise ValueError("recovery_env.recovery_loss_weight must be finite and non-negative")
            episode = {
                "env_id": f"{env_id}:buffer-recovery",
                "group_id": group_id,
                "task_id": entry.task_id,
                "attempt_id": f"{group_id}:{sample_index}",
                "attempt_role": "buffer-recovery",
                "messages": list(messages),
                "current_attempt_start_index": start_index,
                # Critical: prior Initial reasoning/actions are context only.
                "training_start_index": start_index,
                "attempt_turns": len(attempt.agent_result.metadata.get("turns", [])),
                "reward": reward,
                "loss_weight": loss_weight,
                "root_metric_reward": reward,
                "metric_valid": attempt.status is not AttemptStatus.ERROR,
                "dataset_name": dataset.name,
                "benchmark_result": benchmark_result,
                "buffer_entry_id": entry.entry_id,
                "metrics": {
                    f"{dataset.name}/{key}": value
                    for key, value in self._result_metrics(benchmark_result, reward=reward).items()
                },
            }
            if attempt.status is AttemptStatus.ERROR:
                episode["discard_reason"] = "reterm_recovery_attempt_error"
                episode["discard_detail"] = str(
                    attempt.agent_result.error or "Recovery attempt returned ERROR without detail"
                )[-2000:]
            return episode
        finally:
            watcher_stop.set()
            if dataset is not None:
                dataset.close()
            if watcher is not None:
                watcher.join(timeout=5.0)
            self._generation_seed.value = None
            self._generation_turn.value = 0
            self._rollout_cancel_event.value = None

    @staticmethod
    def _watch_dataset_cancellation(
        dataset: Any,
        cancel_event: threading.Event,
        stop_event: threading.Event,
    ) -> None:
        """Tear down an active sandbox as soon as its rollout is cancelled."""

        while not stop_event.is_set():
            if cancel_event.wait(timeout=0.1):
                if not stop_event.is_set():
                    dataset.cancel_active_environment()
                return

    @staticmethod
    def _stable_generation_seed(*, rollout_seed: int, task_id: str, sample_index: int, val: bool) -> int:
        payload = f"{rollout_seed}|{int(val)}|{task_id}|{sample_index}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF

    @staticmethod
    def _infrastructure_failure_category(exc: Exception) -> str | None:
        if isinstance(exc, TaskInfrastructureError):
            return exc.category if exc.category in ReTermAgentProxy._INFRASTRUCTURE_FAILURE_CATEGORIES else "environment"
        if isinstance(exc, TaskSkip):
            return "environment"
        if isinstance(exc, RolloutTimeoutError):
            return "generation_backend"
        return None

    @classmethod
    def _infrastructure_metrics(cls, category: str | None = None) -> dict[str, float]:
        if category is not None and category not in cls._INFRASTRUCTURE_FAILURE_CATEGORIES:
            raise ValueError(f"unknown infrastructure failure category: {category}")
        metrics = {"infrastructure_failures": float(category is not None)}
        metrics.update(
            {
                f"infrastructure_{known_category}": float(category == known_category)
                for known_category in cls._INFRASTRUCTURE_FAILURE_CATEGORIES
            }
        )
        return metrics

    def _generate_response(self, messages: list[dict[str, Any]], context: AgentContext) -> PolicyResponse:
        cancel_event_holder = getattr(self, "_rollout_cancel_event", None)
        cancel_event = getattr(cancel_event_holder, "value", None)
        if cancel_event is not None and cancel_event.is_set():
            raise RolloutCancelledError("speculative rollout cancelled before LLM generation")
        prompt_ids = self._render_official_qwen_prompt_token_ids(messages)
        if self.llm_client is None:
            raise RuntimeError(
                "rl DPPO rollout requires llm_client/vLLM generation with sampled-token logprobs; "
                "actor_wg.generate_sequences fallback is disabled."
            )
        response = self._generate_response_with_llm_client(
            prompt_ids,
            cancel_event=cancel_event,
        )
        metadata = dict(response.metadata or {})
        sampled_ids = metadata.get("raw_sampled_token_ids", response.token_ids or ())
        sampled_log_probs = metadata.get("raw_sampled_log_probs", response.log_probs or ())
        metadata["official_qwen_segment"] = {
            "prompt_ids": [int(token_id) for token_id in prompt_ids],
            "response_ids": [int(token_id) for token_id in sampled_ids],
            "old_log_probs": [float(value) for value in sampled_log_probs],
        }
        response.metadata = metadata
        return response

    def _render_official_qwen_prompt_token_ids(self, messages: list[dict[str, Any]]) -> list[int]:
        """Render one generation prompt through Qwen's native official interface."""

        native_messages: list[dict[str, Any]] = []
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError(f"Qwen history message {index} is not a dict")
            native = message.get("native_message")
            if native is not None:
                if not isinstance(native, dict):
                    raise ValueError(f"Qwen native history message {index} is not a dict")
                native_messages.append(dict(native))
                continue
            role = message.get("role")
            content = message.get("content")
            if role not in {"system", "user"} or not isinstance(content, str):
                raise ValueError(
                    f"Qwen official-native history message {index} lacks native provenance: role={role!r}"
                )
            native_messages.append({"role": role, "content": content})
        return self._apply_qwen_chat_template_token_ids(native_messages, add_generation_prompt=True)

    def _generate_response_with_llm_client(
        self,
        prompt_ids: list[int],
        *,
        cancel_event: threading.Event | None = None,
    ) -> PolicyResponse:
        rollout_cfg = self.config.actor_rollout_ref.rollout
        if not bool(getattr(rollout_cfg, "calculate_log_probs", False)):
            raise RuntimeError("rl DPPO rollout requires actor_rollout_ref.rollout.calculate_log_probs=true")
        max_tokens = self._generation_max_tokens(prompt_len=len(prompt_ids), rollout_cfg=rollout_cfg)
        timeout = self._bounded_llm_timeout(float(os.environ.get("RETERM_RL_LLM_TIMEOUT", "300")))
        sampling_params = {
            "temperature": float(rollout_cfg.temperature),
            "top_p": float(rollout_cfg.top_p),
            "top_k": int(rollout_cfg.top_k),
            "logprobs": True,
            "max_tokens": max_tokens,
        }
        generation_seed = getattr(self._generation_seed, "value", None)
        if generation_seed is not None:
            generation_turn = int(getattr(self._generation_turn, "value", 0))
            sampling_params["seed"] = (int(generation_seed) + generation_turn) & 0x7FFFFFFF
            self._generation_turn.value = generation_turn + 1

        async def _run():
            request_id = f"reterm-{random.randrange(2**63)}"
            generation = asyncio.create_task(
                self.llm_client.generate(
                    request_id=request_id,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    _backend_request_id=request_id,
                )
            )
            started = time.monotonic()
            rollout_cancelled = False
            try:
                while True:
                    remaining = timeout - (time.monotonic() - started)
                    if remaining <= 0:
                        raise asyncio.TimeoutError
                    done, _pending = await asyncio.wait(
                        {generation},
                        timeout=min(1.0, remaining),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if generation in done:
                        return generation.result()
                    if cancel_event is not None and cancel_event.is_set():
                        rollout_cancelled = True
                        abort_request = getattr(self.llm_client, "abort_request", None)
                        if abort_request is not None:
                            try:
                                await asyncio.wait_for(
                                    abort_request(
                                        request_id,
                                        backend_request_id=request_id,
                                        # This is a precise single-request
                                        # abort. Clearing the server-wide
                                        # prefix cache once per speculative
                                        # tail request serializes cancellation
                                        # and evicts useful prefixes for the
                                        # accepted batch.
                                        reset_prefix_cache=False,
                                    ),
                                    timeout=10.0,
                                )
                            except Exception:
                                # The request may have completed between the
                                # cancellation poll and the abort RPC. Local
                                # task cancellation below remains mandatory.
                                pass
                        raise RolloutCancelledError(
                            "speculative rollout cancelled during LLM generation"
                        )
            except asyncio.TimeoutError as exc:
                raise RolloutTimeoutError(
                    f"rollout exceeded wall-clock limit during LLM generation after {timeout:.1f}s"
                ) from exc
            finally:
                if not generation.done():
                    generation.cancel()
                    if rollout_cancelled:
                        try:
                            await asyncio.wait_for(
                                asyncio.gather(generation, return_exceptions=True),
                                timeout=10.0,
                            )
                        except asyncio.TimeoutError:
                            pass
                    else:
                        await asyncio.gather(generation, return_exceptions=True)

        output = asyncio.run(_run())
        if output.log_probs is None:
            raise RuntimeError("rl DPPO rollout expected vLLM token logprobs, but output.log_probs is None")
        sampled_token_ids = [int(value) for value in output.token_ids]
        sampled_log_probs = [float(value) for value in output.log_probs]
        content, token_ids, log_probs = self._decoded_generation_with_log_probs(
            token_ids=sampled_token_ids,
            log_probs=sampled_log_probs,
        )
        if token_ids is None or log_probs is None:
            raise RuntimeError("rl DPPO rollout could not align generated token_ids with sampled logprobs")
        return PolicyResponse(
            content=content,
            token_ids=token_ids,
            log_probs=log_probs,
            metadata={
                "raw_sampled_token_ids": sampled_token_ids,
                "raw_sampled_log_probs": sampled_log_probs,
            },
        )

    @staticmethod
    def _rollout_wall_time_limit_seconds() -> float | None:
        raw = os.environ.get("RETERM_RL_ROLLOUT_WALL_TIME_LIMIT_SECONDS")
        if raw is None or raw.strip() == "":
            return None
        value = float(raw)
        return value if value > 0 else None

    def _bounded_llm_timeout(self, configured_timeout: float) -> float:
        deadline = getattr(self._rollout_deadline, "value", None)
        if deadline is None:
            return configured_timeout
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RolloutTimeoutError("rollout exceeded wall-clock limit before LLM generation")
        return max(1.0, min(configured_timeout, remaining))

    def _set_rollout_deadline(self, deadline: float | None) -> None:
        self._rollout_deadline.value = deadline

    def _generation_max_tokens(self, *, prompt_len: int, rollout_cfg: Any) -> int:
        limits = [int(rollout_cfg.response_length)]
        max_model_len = getattr(rollout_cfg, "max_model_len", None)
        if max_model_len is not None:
            limits.append(int(max_model_len) - prompt_len)
        max_train_tokens = self._max_train_token_len()
        if max_train_tokens is not None:
            im_end_len = len(self.tokenizer.encode(self._qwen_im_end(), add_special_tokens=False))
            limits.append(max_train_tokens - prompt_len - im_end_len - self._train_context_reserve_tokens())
        max_tokens = min(limits)
        if max_tokens <= 0:
            max_context = max_train_tokens if max_train_tokens is not None else int(max_model_len)
            raise RuntimeError(f"Prompt length ({prompt_len}) exceeds maximum context length ({max_context})")
        configured_max_tokens = int(rollout_cfg.response_length)
        if max_tokens < configured_max_tokens:
            print(
                "[rl] capping rollout max_tokens "
                f"from {configured_max_tokens} to {max_tokens} for prompt_len={prompt_len}"
            )
        return max_tokens

    def _formulate_rollouts(self, examples: list[dict[str, Any]]) -> tuple[DataProto, dict[str, float]]:
        return self._formulate_official_segment_rollouts(examples)

    def _formulate_official_segment_rollouts(
        self,
        examples: list[dict[str, Any]],
    ) -> tuple[DataProto, dict[str, float]]:
        """Materialize official-template generations while retaining attempt identity.

        Adjacent generations are coalesced only when the next official prompt
        contains the complete materialized row as an exact token prefix.  This
        is an exact compute optimization: incompatible Qwen history rewrites
        remain independent rows, while compatible tool loops avoid recomputing
        their shared prefix.
        """

        started = time.time()
        rows: list[dict[str, Any]] = []
        raw_segment_count = 0
        raw_materialized_tokens = 0
        exact_prefix_merges = 0
        incompatible_boundaries = 0
        reused_prefix_tokens = 0
        for example in examples:
            segments = self._official_segments_for_example(example)
            coalesced, coalesce_metrics = self._coalesce_official_segments(segments)
            attempt_id = str(example.get("attempt_id", f"{example['group_id']}:{example['env_id']}"))
            attempt_action_tokens = sum(len(segment[1]) for segment in segments)
            if attempt_action_tokens <= 0:
                raise ValueError(f"official Qwen attempt {attempt_id} has no sampled response tokens")
            raw_segment_count += len(segments)
            raw_materialized_tokens += sum(len(prompt_ids) + len(response_ids) for prompt_ids, response_ids, _ in segments)
            exact_prefix_merges += int(coalesce_metrics["exact_prefix_merges"])
            incompatible_boundaries += int(coalesce_metrics["incompatible_boundaries"])
            reused_prefix_tokens += int(coalesce_metrics["reused_prefix_tokens"])
            for row_index, row in enumerate(coalesced):
                rows.append(
                    {
                        "example": example,
                        "attempt_id": attempt_id,
                        "segment_indices": list(row["segment_indices"]),
                        "is_final_segment": row_index == len(coalesced) - 1,
                        "attempt_action_tokens": attempt_action_tokens,
                        "token_ids": list(row["token_ids"]),
                        "mask": torch.tensor(row["mask"], dtype=torch.bool),
                        "old_log_probs": torch.tensor(row["old_log_probs"], dtype=torch.float32),
                        "training_padding": False,
                    }
                )
        if not rows:
            raise ValueError("official Qwen segmented rollout batch is empty")

        real_row_count = len(rows)
        row_multiple = self._actor_data_parallel_size()
        padding_rows = (-real_row_count) % row_multiple
        if padding_rows:
            # The actor engine requires its global mini-batch to divide evenly
            # across data-parallel ranks.  Add tiny, fully masked rows rather
            # than duplicating a real trajectory and silently changing loss.
            seed_ids = rows[0]["token_ids"][:2]
            if len(seed_ids) < 2:
                raise ValueError("official Qwen training rows require at least two context tokens")
            for padding_index in range(padding_rows):
                rows.append(
                    {
                        "example": {
                            "env_id": f"padding-{padding_index}",
                            "group_id": "__padding__",
                            "task_id": "__padding__",
                            "reward": 0.0,
                            "metric_valid": False,
                            "discard_reason": "training_padding",
                        },
                        "attempt_id": f"__padding__:{padding_index}",
                        "segment_indices": [],
                        "is_final_segment": False,
                        "attempt_action_tokens": 0,
                        "token_ids": list(seed_ids),
                        "mask": torch.zeros(len(seed_ids), dtype=torch.bool),
                        "old_log_probs": torch.zeros(len(seed_ids), dtype=torch.float32),
                        "training_padding": True,
                    }
                )

        seq_len = max(len(row["token_ids"]) for row in rows)
        pad_token_id = int(self.tokenizer.pad_token_id)
        input_ids = self._pad_token_id_lists([row["token_ids"] for row in rows], seq_len, pad_token_id)
        attention_mask = self._attention_mask_from_lengths([len(row["token_ids"]) for row in rows], seq_len)
        position_ids = attention_mask.cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        response_mask_full = self._pad_rendered_masks([row["mask"] for row in rows], seq_len)
        old_log_probs_full = self._pad_rendered_float_tensors([row["old_log_probs"] for row in rows], seq_len)
        response_mask = response_mask_full[:, 1:].clone()
        loss_mask = response_mask.clone()
        old_log_probs = old_log_probs_full[:, 1:].clone()
        score_tensor = torch.zeros_like(old_log_probs)
        for row_index, row in enumerate(rows):
            if not row["is_final_segment"]:
                continue
            selected = torch.where(response_mask[row_index])[0]
            if len(selected) == 0:
                raise ValueError(f"official Qwen final segment row {row_index} has no response tokens")
            score_tensor[row_index, selected[-1]] = float(row["example"]["reward"])

        data = DataProto()
        data.batch = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "responses": input_ids[:, 1:],
                "loss_mask": loss_mask,
                "rm_scores": score_tensor,
                "original_rm_scores": score_tensor.clone(),
                "response_mask": response_mask,
                "old_log_probs": old_log_probs,
            },
            batch_size=input_ids.shape[0],
        )
        segment_indices = np.empty(len(rows), dtype=object)
        segment_indices[:] = [tuple(row["segment_indices"]) for row in rows]
        data.non_tensor_batch = {
            "env_ids": np.array(
                [
                    f"{row['example']['env_id']}:segments-"
                    + ("-".join(str(value) for value in row["segment_indices"]) or "padding")
                    for row in rows
                ],
                dtype=object,
            ),
            "attempt_ids": np.array([row["attempt_id"] for row in rows], dtype=object),
            "attempt_rewards": np.array([float(row["example"]["reward"]) for row in rows], dtype=object),
            "attempt_loss_weights": np.array(
                [float(row["example"].get("loss_weight", 1.0)) for row in rows], dtype=object
            ),
            "attempt_action_tokens": np.array([int(row["attempt_action_tokens"]) for row in rows], dtype=object),
            "segment_indices": segment_indices,
            "final_segments": np.array([bool(row["is_final_segment"]) for row in rows], dtype=object),
            "training_padding": np.array([bool(row["training_padding"]) for row in rows], dtype=object),
            "group_ids": np.array([row["example"]["group_id"] for row in rows], dtype=object),
            "task_ids": np.array([row["example"]["task_id"] for row in rows], dtype=object),
            "data_source": np.array([self._dataset_name()] * len(rows), dtype=object),
            "reward": np.array([float(row["example"]["reward"]) for row in rows], dtype=object),
            # Validation should count each attempt once, not once per generation segment.
            "metric_valid": np.array(
                [bool(row["example"].get("metric_valid", True)) and row["is_final_segment"] for row in rows],
                dtype=object,
            ),
            # Non-final segment rows and data-parallel padding are physical
            # training rows, not additional evaluated task attempts.
            "metric_counted": np.array(
                [
                    bool(row["is_final_segment"])
                    and not bool(row["training_padding"])
                    and bool(row["example"].get("metric_counted", True))
                    for row in rows
                ],
                dtype=object,
            ),
            "metric_rewards": np.array(
                [float(row["example"].get("metric_reward", row["example"]["reward"])) for row in rows],
                dtype=np.float32,
            ),
            "discard_reason": np.array([row["example"].get("discard_reason") for row in rows], dtype=object),
        }
        materialized_tokens = sum(len(row["token_ids"]) for row in rows if not row["training_padding"])
        action_tokens = sum(int(row["attempt_action_tokens"]) for row in rows if row["is_final_segment"])
        return data, {
            "rollout_formulate_render_seconds": time.time() - started,
            "official_qwen_attempts": float(len(examples)),
            "official_qwen_raw_segments": float(raw_segment_count),
            "official_qwen_training_rows": float(real_row_count),
            "official_qwen_training_padding_rows": float(padding_rows),
            "official_qwen_raw_segments_per_attempt": float(raw_segment_count / len(examples)),
            "official_qwen_training_rows_per_attempt": float(real_row_count / len(examples)),
            "official_qwen_exact_prefix_merges": float(exact_prefix_merges),
            "official_qwen_incompatible_boundaries": float(incompatible_boundaries),
            "official_qwen_reused_prefix_tokens": float(reused_prefix_tokens),
            "official_qwen_raw_materialized_tokens": float(raw_materialized_tokens),
            "official_qwen_materialized_tokens": float(materialized_tokens),
            "official_qwen_materialized_token_ratio": float(materialized_tokens / raw_materialized_tokens),
            "official_qwen_action_tokens": float(action_tokens),
        }

    @staticmethod
    def _coalesce_official_segments(
        segments: list[tuple[list[int], list[int], list[float]]],
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """Coalesce only generations whose official contexts are exact extensions."""

        rows: list[dict[str, Any]] = []
        exact_prefix_merges = 0
        incompatible_boundaries = 0
        reused_prefix_tokens = 0
        for segment_index, (prompt_ids, response_ids, response_log_probs) in enumerate(segments):
            if not response_ids or len(response_ids) != len(response_log_probs):
                raise ValueError("official Qwen segment response/logprob lengths do not align")
            prompt = [int(value) for value in prompt_ids]
            response = [int(value) for value in response_ids]
            log_probs = [float(value) for value in response_log_probs]
            if rows:
                current = rows[-1]
                current_ids = current["token_ids"]
                if len(prompt) >= len(current_ids) and prompt[: len(current_ids)] == current_ids:
                    prompt_suffix = prompt[len(current_ids) :]
                    reused_prefix_tokens += len(current_ids)
                    current["token_ids"].extend([*prompt_suffix, *response])
                    current["mask"].extend([False] * len(prompt_suffix) + [True] * len(response))
                    current["old_log_probs"].extend([0.0] * len(prompt_suffix) + log_probs)
                    current["segment_indices"].append(segment_index)
                    exact_prefix_merges += 1
                    continue
                incompatible_boundaries += 1
            rows.append(
                {
                    "token_ids": [*prompt, *response],
                    "mask": [False] * len(prompt) + [True] * len(response),
                    "old_log_probs": [0.0] * len(prompt) + log_probs,
                    "segment_indices": [segment_index],
                }
            )
        return rows, {
            "exact_prefix_merges": exact_prefix_merges,
            "incompatible_boundaries": incompatible_boundaries,
            "reused_prefix_tokens": reused_prefix_tokens,
        }

    def _actor_data_parallel_size(self) -> int:
        trainer_cfg = self.config.trainer
        total_gpus = int(getattr(trainer_cfg, "n_gpus_per_node", 1)) * int(
            getattr(trainer_cfg, "nnodes", 1)
        )
        actor_cfg = self.config.actor_rollout_ref.actor
        megatron_cfg = getattr(actor_cfg, "megatron", {})
        try:
            tp_size = int(megatron_cfg.get("tensor_model_parallel_size", 1))
            pp_size = int(megatron_cfg.get("pipeline_model_parallel_size", 1))
            cp_size = int(megatron_cfg.get("context_parallel_size", 1))
        except AttributeError:
            tp_size = int(getattr(megatron_cfg, "tensor_model_parallel_size", 1))
            pp_size = int(getattr(megatron_cfg, "pipeline_model_parallel_size", 1))
            cp_size = int(getattr(megatron_cfg, "context_parallel_size", 1))
        model_parallel_size = max(1, tp_size * pp_size * cp_size)
        if total_gpus % model_parallel_size != 0:
            raise ValueError(
                "official Qwen segment rows cannot derive actor data parallelism: "
                f"gpus={total_gpus} tp={tp_size} pp={pp_size} cp={cp_size}"
            )
        return max(1, total_gpus // model_parallel_size)

    @staticmethod
    def _official_segments_for_example(
        example: dict[str, Any],
    ) -> list[tuple[list[int], list[int], list[float]]]:
        messages = example.get("messages")
        if not isinstance(messages, list):
            raise ValueError("rollout example is missing messages")
        start_index = example.get("training_start_index", example.get("current_attempt_start_index"))
        if not isinstance(start_index, int):
            raise ValueError("rollout example is missing training_start_index")
        segments: list[tuple[list[int], list[int], list[float]]] = []
        for message_index, message in enumerate(messages[start_index:], start=start_index):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            raw = message.get("official_qwen_segment")
            if not isinstance(raw, dict):
                message_type = message.get("message_type")
                content_length = len(message.get("content", "")) if isinstance(message.get("content"), str) else -1
                raise ValueError(
                    "assistant message is missing official_qwen_segment provenance: "
                    f"message_index={message_index} message_type={message_type!r} "
                    f"content_length={content_length} keys={sorted(message)}"
                )
            prompt_ids = raw.get("prompt_ids")
            response_ids = raw.get("response_ids")
            old_log_probs = raw.get("old_log_probs")
            if not isinstance(prompt_ids, list) or not isinstance(response_ids, list) or not isinstance(old_log_probs, list):
                raise ValueError("official_qwen_segment contains malformed token provenance")
            if not response_ids or len(response_ids) != len(old_log_probs):
                raise ValueError("official_qwen_segment response/logprob lengths do not align")
            segments.append(
                (
                    [int(value) for value in prompt_ids],
                    [int(value) for value in response_ids],
                    [float(value) for value in old_log_probs],
                )
            )
        if not segments:
            raise ValueError("rollout example contains no official Qwen generation segments")
        return segments

    def _is_trainable_example(self, example: dict[str, Any], *, allow_discard: bool = False) -> bool:
        if example.get("discard_reason") and not allow_discard:
            print(
                "[rl] dropping discarded rollout "
                f"task={example.get('task_id')} env_id={example.get('env_id')} "
                f"reason={example.get('discard_reason')}",
                flush=True,
            )
            return False
        try:
            segments = self._official_segments_for_example(example)
        except Exception as exc:  # noqa: BLE001
            print(
            "[rl] dropping non-trainable rollout "
                f"task={example.get('task_id')} env_id={example.get('env_id')}: {type(exc).__name__}: {exc}"
            )
            return False
        max_train_tokens = self._max_train_token_len()
        if max_train_tokens is not None and any(
            len(prompt_ids) + len(response_ids) > max_train_tokens
            for prompt_ids, response_ids, _old_log_probs in segments
        ):
            return False
        return all(
            all(np.isfinite(value) for value in old_log_probs)
            for _prompt_ids, _response_ids, old_log_probs in segments
        )

    def _is_over_context_example(self, example: dict[str, Any]) -> bool:
        metrics = example.get("metrics")
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                if key.endswith("/filtered_over_context") and float(value) > 0:
                    return True
        try:
            max_train_tokens = self._max_train_token_len()
            if max_train_tokens is not None and any(
                len(prompt_ids) + len(response_ids) > max_train_tokens
                for prompt_ids, response_ids, _old_log_probs in self._official_segments_for_example(example)
            ):
                return True
        except Exception:
            pass
        return self._contains_over_context_error(example.get("benchmark_result"))

    def _max_train_token_len(self) -> int | None:
        actor_cfg = self.config.actor_rollout_ref.actor
        max_token_len = getattr(actor_cfg, "ppo_max_token_len_per_gpu", None)
        if max_token_len is None:
            return None
        sp_size = int(getattr(actor_cfg, "ulysses_sequence_parallel_size", 1) or 1)
        return int(max_token_len) * max(1, sp_size)

    @staticmethod
    def _train_context_reserve_tokens() -> int:
        return max(0, int(os.environ.get("RETERM_RL_TRAIN_CONTEXT_RESERVE_TOKENS", "256")))

    def _contains_over_context_error(self, benchmark_result: Any) -> bool:
        if benchmark_result is None:
            return False
        if isinstance(benchmark_result, BenchmarkResult):
            attempts = benchmark_result.attempts
        elif isinstance(benchmark_result, dict):
            attempts = benchmark_result.get("attempts", [])
        else:
            return False
        for attempt in attempts:
            if isinstance(attempt, dict):
                agent_result = attempt.get("agent_result", {})
                error = agent_result.get("error") if isinstance(agent_result, dict) else None
                metadata = agent_result.get("metadata", {}) if isinstance(agent_result, dict) else {}
            else:
                error = getattr(attempt.agent_result, "error", None)
                metadata = attempt.agent_result.metadata
            if self._is_over_context_error_text(error):
                return True
            if isinstance(metadata, dict) and metadata.get("stopped_by_context_limit"):
                return True
        return False

    @staticmethod
    def _is_over_context_error_text(error: Any) -> bool:
        if not isinstance(error, str):
            return False
        return "Prompt length (" in error and "maximum context length" in error


    def _apply_qwen_chat_template_token_ids(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
    ) -> list[int]:
        self._ensure_qwen_chat_template()
        token_ids = self.tokenizer.apply_chat_template(
            messages,
            tools=self._qwen_tool_schemas(),
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
        )
        if isinstance(token_ids, dict) or (
            hasattr(token_ids, "__contains__") and "input_ids" in token_ids
        ):
            token_ids = token_ids["input_ids"]
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if token_ids and isinstance(token_ids[0], list):
            if len(token_ids) != 1:
                raise ValueError(f"unexpected batched chat template output with batch size {len(token_ids)}")
            token_ids = token_ids[0]
        return [int(token_id) for token_id in token_ids]

    def _decoded_generation_with_log_probs(
        self,
        *,
        token_ids: list[int],
        log_probs: list[float],
    ) -> tuple[str, list[int] | None, list[float] | None]:
        if len(token_ids) != len(log_probs):
            return "", None, None

        # Strip special IDs only from the text fed to the action parser. The
        # sampled action IDs/logprobs returned below remain byte-for-byte intact.
        stripped_ids, _ = self._strip_special_tokens(token_ids, log_probs)
        if not stripped_ids:
            return "", [int(token_id) for token_id in token_ids], [float(value) for value in log_probs]

        content = self.tokenizer.decode(
            stripped_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        return content, [int(token_id) for token_id in token_ids], [float(value) for value in log_probs]

    def _strip_special_tokens(
        self,
        token_ids: list[int],
        log_probs: list[float],
    ) -> tuple[list[int], list[float]]:
        special_ids = set(self.tokenizer.all_special_ids)
        filtered = [
            (token_id, log_prob)
            for token_id, log_prob in zip(token_ids, log_probs)
            if token_id not in special_ids
        ]
        return [token_id for token_id, _log_prob in filtered], [
            float(log_prob) for _token_id, log_prob in filtered
        ]

    def _qwen_im_end(self) -> str:
        self._ensure_qwen_chat_template()
        return "<|im_end|>\n"

    def _ensure_qwen_chat_template(self) -> None:
        probe = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "x"}],
            tools=self._qwen_tool_schemas(),
            add_generation_prompt=True,
            tokenize=False,
        )
        expected_suffix = "<|im_start|>user\nx<|im_end|>\n<|im_start|>assistant\n<think>\n"
        if not probe.endswith(expected_suffix):
            raise ValueError("ReTerm-GRPO requires the Qwen3 chat template")

    def _qwen_tool_schemas(self) -> list[dict[str, Any]]:
        return TERMINAL_TOOLS

    @staticmethod
    def _pad_rendered_masks(masks: list[torch.Tensor], seq_len: int) -> torch.Tensor:
        padded: list[torch.Tensor] = []
        for index, mask in enumerate(masks):
            if len(mask) > seq_len:
                raise ValueError(f"rendered mask {index} length {len(mask)} exceeds padded sequence length {seq_len}")
            pad = torch.zeros(seq_len - len(mask), dtype=torch.bool)
            padded.append(torch.cat([pad, mask], dim=0))
        return torch.stack(padded, dim=0)

    @staticmethod
    def _pad_rendered_float_tensors(values: list[torch.Tensor], seq_len: int) -> torch.Tensor:
        padded: list[torch.Tensor] = []
        for index, value in enumerate(values):
            if len(value) > seq_len:
                raise ValueError(f"rendered tensor {index} length {len(value)} exceeds padded sequence length {seq_len}")
            pad = torch.zeros(seq_len - len(value), dtype=torch.float32)
            padded.append(torch.cat([pad, value.to(dtype=torch.float32)], dim=0))
        return torch.stack(padded, dim=0)

    @staticmethod
    def _pad_token_id_lists(values: list[list[int]], seq_len: int, pad_token_id: int) -> torch.Tensor:
        padded: list[torch.Tensor] = []
        for index, value in enumerate(values):
            if len(value) > seq_len:
                raise ValueError(f"rendered token ids {index} length {len(value)} exceeds padded sequence length {seq_len}")
            pad = torch.full((seq_len - len(value),), pad_token_id, dtype=torch.long)
            padded.append(torch.cat([pad, torch.tensor(value, dtype=torch.long)], dim=0))
        return torch.stack(padded, dim=0)

    @staticmethod
    def _attention_mask_from_lengths(lengths: list[int], seq_len: int) -> torch.Tensor:
        masks: list[torch.Tensor] = []
        for index, length in enumerate(lengths):
            if length > seq_len:
                raise ValueError(f"sequence {index} length {length} exceeds padded sequence length {seq_len}")
            masks.append(torch.cat([torch.zeros(seq_len - length, dtype=torch.long), torch.ones(length, dtype=torch.long)]))
        return torch.stack(masks, dim=0)

    def _episode_reward(self, result: BenchmarkResult) -> float:
        return float(result.success)

    def _result_metrics(self, result: BenchmarkResult, *, reward: float) -> dict[str, float]:
        success_attempt = 0
        action_count = 0
        format_error_count = 0
        timeout_count = 0
        submit_count = 0
        over_context_count = 0
        wall_time_count = 0
        command_elapsed_seconds: list[float] = []
        successful_command_elapsed_seconds: list[float] = []
        for attempt in result.attempts:
            if attempt.outcome is not None and attempt.outcome.success and success_attempt == 0:
                success_attempt = int(attempt.attempt_index)
            over_context_count += int(
                self._is_over_context_error_text(getattr(attempt.agent_result, "error", None))
                or bool(attempt.agent_result.metadata.get("stopped_by_context_limit", False))
            )
            wall_time_count += int(bool(attempt.agent_result.metadata.get("stopped_by_wall_time", False)))
            action_count += len(attempt.agent_result.actions)
            for action in attempt.agent_result.actions:
                timeout_count += int(bool(action.metadata.get("timed_out", False)))
                submit_count += int(bool(action.metadata.get("done", False)))
                elapsed = action.metadata.get("command_elapsed_seconds")
                if elapsed is not None:
                    elapsed = float(elapsed)
                    command_elapsed_seconds.append(elapsed)
                    if action.metadata.get("exit_code") == 0 and not action.metadata.get("timed_out", False):
                        successful_command_elapsed_seconds.append(elapsed)
            for turn in attempt.agent_result.metadata.get("turns", []):
                format_error_count += int("format_error" in turn)

        attempts = len(result.attempts)
        return {
            "success": float(result.success),
            "attempts": float(attempts),
            "success_attempt": float(success_attempt),
            "recovered_after_failure": float(success_attempt > 1),
            "unrecovered_failure": float(not result.success),
            "actions": float(action_count),
            "format_errors": float(format_error_count),
            "command_timeouts": float(timeout_count),
            "submit_actions": float(submit_count),
            "filtered_over_context": float(over_context_count),
            "filtered_wall_time": float(wall_time_count),
            "actions_per_attempt": float(action_count / attempts) if attempts else 0.0,
            "format_error_rate": float(format_error_count / max(1, attempts)),
            "command_elapsed_mean_seconds": _mean(command_elapsed_seconds),
            "successful_command_elapsed_mean_seconds": _mean(successful_command_elapsed_seconds),
            "successful_command_count": float(len(successful_command_elapsed_seconds)),
        }

    def _messages_and_current_attempt_start(self, result: BenchmarkResult) -> tuple[list[dict[str, str]], int]:
        for attempt in reversed(result.attempts):
            messages = attempt.agent_result.metadata.get("messages")
            if messages:
                start_index = attempt.agent_result.metadata.get("current_attempt_start_index")
                if not isinstance(start_index, int):
                    raise ValueError("agent result is missing current_attempt_start_index")
                return list(messages), start_index
        return [
            {"role": "system", "content": "ReTerm rollout produced no agent messages."},
            {"role": "user", "content": "The task ended before an agent interaction could be recorded."},
            {"role": "assistant", "content": ""},
        ], 2

    def _trajectory_dir(self) -> Path | None:
        root = os.environ.get("RETERM_RL_TRACKING_ROOT")
        if not root:
            run_id = os.environ.get("RUN_ID")
            if run_id:
                root = str(Path("rl") / "runs" / "tracking" / run_id)
        if not root:
            return None
        path = Path(root) / "trajectories"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _save_trajectory(
        self,
        *,
        dataset_name: str,
        episode: dict[str, Any],
        benchmark_result: BenchmarkResult | None = None,
        error: str | None = None,
    ) -> None:
        if self.trajectory_dir is None:
            return
        payload: dict[str, Any] = {
            "dataset": dataset_name,
            "env_id": episode.get("env_id"),
            "group_id": episode.get("group_id"),
            "task_id": episode.get("task_id"),
            "reward": episode.get("reward"),
            "metric_valid": episode.get("metric_valid", True),
            "discard_reason": episode.get("discard_reason"),
            "error_type": episode.get("error_type"),
            "failure_category": episode.get("failure_category"),
            "metrics": episode.get("metrics", {}),
            "messages": episode.get("messages", []),
            "current_attempt_start_index": episode.get("current_attempt_start_index"),
            "training_start_index": episode.get("training_start_index"),
            "attempt_turns": episode.get("attempt_turns"),
            "root_metric_reward": episode.get("root_metric_reward"),
        }
        if error is not None:
            payload["error"] = error
        if benchmark_result is not None:
            # ``dataclasses.asdict`` recursively duplicates the complete
            # multi-attempt trajectory before serialization.  At 64K context
            # this consumed roughly one second per accepted attempt and blocked
            # steady-state group completion.  orjson serializes dataclasses
            # directly, avoiding that second in-memory copy while preserving
            # the same JSON object structure.
            payload["benchmark_result"] = benchmark_result
        out = self.trajectory_dir / "episodes.jsonl"
        line = orjson.dumps(
            payload,
            option=(
                orjson.OPT_APPEND_NEWLINE
                | orjson.OPT_NON_STR_KEYS
                | orjson.OPT_SERIALIZE_DATACLASS
                | orjson.OPT_SERIALIZE_NUMPY
            ),
            default=str,
        )
        with self._trajectory_lock:
            # One O_APPEND write keeps every JSONL record intact.  Opening in
            # unbuffered binary mode also avoids retaining an enormous Python
            # text buffer after each 64K trajectory.
            with out.open("ab", buffering=0) as f:
                f.write(line)

    def _discover_task_ids(self, *, val: bool) -> list[str]:
        dataset = self._make_dataset(val=val)
        try:
            return dataset.list_tasks()
        finally:
            dataset.close()

    def _dataset_name(self) -> str:
        if not hasattr(self, "config"):
            return "task"
        cfg = _to_plain_dict(self.config.recovery_env.dataset)
        return str(cfg.pop("name", "task"))

    def _make_dataset(self, *, val: bool = False):
        cfg = _to_plain_dict(self.config.recovery_env.dataset)
        cfg.pop("name", None)
        default_task_ids_file = cfg.pop("task_ids_file", None)
        train_task_ids_file = cfg.pop("train_task_ids_file", None)
        val_task_ids_file = cfg.pop("val_task_ids_file", None)
        if val and val_task_ids_file:
            selected_task_ids_file = val_task_ids_file
        elif not val and train_task_ids_file:
            selected_task_ids_file = train_task_ids_file
        else:
            selected_task_ids_file = default_task_ids_file
        if selected_task_ids_file:
            cfg["task_ids_file"] = selected_task_ids_file
        return create_task_dataset(cfg)


class _DistributedSandboxEpisodeWorker:
    """Node-pinned, RAM-bounded executor for complete Recovery episodes.

    The central TaskRunner still owns task sampling, GRPO group selection, and
    tensor formulation. The task adapter executes each stateful episode on
    the assigned Ray node.
    """

    def __init__(
        self,
        *,
        config: Any,
        tokenizer: Any,
        llm_client: Any,
        node_ip: str,
        node_index: int,
        max_episodes: int,
        min_available_gb: float,
    ) -> None:
        self.node_ip = str(node_ip)
        self.node_index = int(node_index)
        self.max_episodes = max(1, int(max_episodes))
        self.min_available_gb = max(0.0, float(min_available_gb))
        self._episode_slots = threading.BoundedSemaphore(self.max_episodes)
        recovery_slots = max(1, self.max_episodes // 4)
        self._initial_lane_slots = threading.BoundedSemaphore(self.max_episodes - recovery_slots)
        self._recovery_lane_slots = threading.BoundedSemaphore(recovery_slots)
        self._rollout_events: dict[str, threading.Event] = {}
        self._rollout_active_calls: dict[str, int] = {}
        self._rollout_finishing: set[str] = set()
        self._rollout_events_lock = threading.Lock()
        self._proxy = ReTermAgentProxy(
            config=config,
            actor_rollout_wg=None,
            tokenizer=tokenizer,
            llm_client=llm_client,
            enable_distributed_sandboxes=False,
        )
        # The central TaskRunner writes accepted trajectories. Keeping worker
        # writers disabled prevents two node-local files from looking like one
        # complete rollout ledger.
        self._proxy.trajectory_dir = None

    @staticmethod
    def _mem_available_gb() -> float:
        with Path("/proc/meminfo").open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1024.0 / 1024.0
        raise RuntimeError("/proc/meminfo does not report MemAvailable")

    def _begin_rollout_call(self, rollout_id: str) -> threading.Event:
        with self._rollout_events_lock:
            event = self._rollout_events.setdefault(rollout_id, threading.Event())
            self._rollout_active_calls[rollout_id] = self._rollout_active_calls.get(rollout_id, 0) + 1
            return event

    def _end_rollout_call(self, rollout_id: str) -> None:
        with self._rollout_events_lock:
            remaining = self._rollout_active_calls.get(rollout_id, 0) - 1
            if remaining > 0:
                self._rollout_active_calls[rollout_id] = remaining
                return
            self._rollout_active_calls.pop(rollout_id, None)
            if rollout_id in self._rollout_finishing:
                self._rollout_finishing.discard(rollout_id)
                self._rollout_events.pop(rollout_id, None)

    @staticmethod
    def _raise_if_cancelled(event: threading.Event, detail: str) -> None:
        if event.is_set():
            raise RolloutCancelledError(detail)

    def _acquire_episode_slot(self, event: threading.Event) -> None:
        while not self._episode_slots.acquire(timeout=1.0):
            self._raise_if_cancelled(event, "distributed sandbox rollout cancelled while queued")

    def _wait_for_memory(self, event: threading.Event) -> float:
        while True:
            self._raise_if_cancelled(event, "distributed sandbox rollout cancelled by memory guard")
            available_gb = self._mem_available_gb()
            if available_gb >= self.min_available_gb:
                return available_gb
            print(
                "[rl] sandbox memory guard waiting: "
                f"node={self.node_ip} available={available_gb:.1f}GiB "
                f"floor={self.min_available_gb:.1f}GiB",
                flush=True,
            )
            event.wait(timeout=2.0)

    def run_episode(self, *, rollout_id: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        rollout_id = str(rollout_id)
        event = self._begin_rollout_call(rollout_id)
        acquired = False
        lane_acquired = False
        try:
            initial_lane_slots = getattr(self, "_initial_lane_slots", None)
            if initial_lane_slots is not None:
                while not initial_lane_slots.acquire(timeout=1.0):
                    self._raise_if_cancelled(event, "reterm Initial cancelled while lane-queued")
                lane_acquired = True
            self._acquire_episode_slot(event)
            acquired = True
            available_gb = self._wait_for_memory(event)
            try:
                result = self._proxy._run_one_episode(**dict(kwargs), cancel_event=event)
            except RolloutCancelledError as exc:
                # The central waiter intentionally stops awaiting speculative
                # tails as soon as the optimizer batch is complete.  Returning
                # a successful sentinel prevents Ray from reporting those
                # expected worker-side cancellations as unhandled task errors.
                # This sentinel can never enter a completed batch because the
                # central rollout loop has already set its shared cancel event.
                return {
                    "cancelled": True,
                    "cancel_reason": str(exc),
                    "metrics": {"rollout_cancelled_tail": 1.0},
                    "sandbox_node_ip": self.node_ip,
                    "sandbox_node_index": self.node_index,
                }
            result["sandbox_node_ip"] = self.node_ip
            result["sandbox_node_index"] = self.node_index
            result["metrics"]["rollout_distributed_sandbox"] = 1.0
            result["metrics"][f"rollout_sandbox_node_{self.node_index}"] = 1.0
            result["metrics"]["rollout_sandbox_mem_available_gb_at_start"] = available_gb
            return result
        finally:
            if acquired:
                self._episode_slots.release()
            if lane_acquired:
                initial_lane_slots.release()
            self._end_rollout_call(rollout_id)

    def lease_recovery_state(
        self,
        *,
        max_created_step: int | None = None,
        excluded_entry_ids: list[str] | None = None,
    ) -> dict[str, Any] | None:
        leased = self._proxy._recovery_buffer.lease_newest(
            max_created_step=max_created_step,
            excluded_entry_ids=set(excluded_entry_ids or ()),
        )
        if leased is None:
            return None
        lease_id, entry = leased
        return {"lease_id": lease_id, **entry.public_metadata()}

    def run_buffered_recovery(
        self,
        *,
        rollout_id: str,
        lease_id: str,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        rollout_id = str(rollout_id)
        event = self._begin_rollout_call(rollout_id)
        acquired = False
        lane_acquired = False
        try:
            recovery_lane_slots = getattr(self, "_recovery_lane_slots", None)
            if recovery_lane_slots is not None:
                while not recovery_lane_slots.acquire(timeout=1.0):
                    self._raise_if_cancelled(event, "reterm Recovery cancelled while lane-queued")
                lane_acquired = True
            self._acquire_episode_slot(event)
            acquired = True
            self._wait_for_memory(event)
            result = self._proxy._run_buffered_recovery_episode(
                lease_id=str(lease_id), cancel_event=event, **dict(kwargs)
            )
            result["sandbox_node_ip"] = self.node_ip
            result["sandbox_node_index"] = self.node_index
            return result
        finally:
            if acquired:
                self._episode_slots.release()
            if lane_acquired:
                recovery_lane_slots.release()
            self._end_rollout_call(rollout_id)

    def finish_recovery_state(self, *, lease_id: str, consume: bool) -> dict[str, Any]:
        return self._proxy._recovery_buffer.finish(str(lease_id), consume=bool(consume))

    def cancel_rollout(self, rollout_id: str) -> bool:
        rollout_id = str(rollout_id)
        with self._rollout_events_lock:
            self._rollout_events.setdefault(rollout_id, threading.Event()).set()
        return True

    def finish_rollout(self, rollout_id: str) -> dict[str, int | bool]:
        rollout_id = str(rollout_id)
        with self._rollout_events_lock:
            event = self._rollout_events.setdefault(rollout_id, threading.Event())
            event.set()
            active_calls = self._rollout_active_calls.get(rollout_id, 0)
            if active_calls:
                self._rollout_finishing.add(rollout_id)
            else:
                self._rollout_events.pop(rollout_id, None)
                self._rollout_finishing.discard(rollout_id)
        return {"finished": active_calls == 0, "active_calls": active_calls}

    def resolve_buffer_group(
        self,
        *,
        rollout_id: str,
        group_id: int,
        promote: bool,
    ) -> dict[str, Any] | None:
        return self._proxy._recovery_buffer.resolve_provisional(
            rollout_id=str(rollout_id),
            source_group_id=int(group_id),
            promote=bool(promote),
        )

    def status(self) -> dict[str, Any]:
        return {
            "hostname": os.uname().nodename,
            "node_ip": self.node_ip,
            "node_index": self.node_index,
            "max_episodes": self.max_episodes,
            "min_available_gb": self.min_available_gb,
            "mem_available_gb": round(self._mem_available_gb(), 1),
            "tracked_rollouts": len(self._rollout_events),
            "active_rollout_calls": sum(self._rollout_active_calls.values()),
            "initial_lane_capacity": (
                self.max_episodes - max(1, self.max_episodes // 4)
                if getattr(self, "_initial_lane_slots", None) is not None
                else self.max_episodes
            ),
            "recovery_lane_capacity": (
                max(1, self.max_episodes // 4)
                if getattr(self, "_recovery_lane_slots", None) is not None
                else 0
            ),
            "recovery_buffer": self._proxy._recovery_buffer.status(),
        }
