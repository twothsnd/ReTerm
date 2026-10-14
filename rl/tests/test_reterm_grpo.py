import asyncio
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from verl import DataProto
from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from reterm_core.qwen_tools import TERMINAL_TOOLS
from reterm_rl.agents.terminal_policy import (
    PolicyResponse,
    RolloutCancelledError,
    TerminalPolicyAgent,
    TerminalPolicyAgentConfig,
)
from reterm_core.types import (
    AgentContext,
    AgentRunResult,
    AttemptRecord,
    AttemptStatus,
    BenchmarkResult,
    StateSnapshot,
    Task,
    TaskOutcome,
)
from reterm_rl.agents.terminal_io import CommandResult
from reterm_rl.llm_agent.recovery_proxy import ReTermAgentProxy
from reterm_rl.trainer.agent_trainer import (
    ReTermTrainer,
    _remove_ephemeral_checkpoint_tree,
    compute_official_attempt_grpo_advantage,
)


class OfficialQwenSegmentTest(unittest.TestCase):
    def test_reterm_8plus2_bootstrap_and_empty_buffer_dispatch(self):
        proxy = object.__new__(ReTermAgentProxy)
        proxy.config = SimpleNamespace(
            recovery_env=SimpleNamespace(
                recovery_buffer_initial_groups=8,
                recovery_buffer_bootstrap_steps=2, recovery_buffer_bootstrap_groups=8,
            ),
            es_manager=SimpleNamespace(train=SimpleNamespace(env_groups=10, group_size=16)),
        )
        status = mock.Mock(return_value={"recovery_buffer": {"available": 2}})
        worker = SimpleNamespace(status=SimpleNamespace(remote=status))
        proxy._sandbox_workers = [worker, worker]
        proxy._new_sandbox_rollout_id = mock.Mock(return_value="rollout")
        proxy._steady_state_sampling_enabled = mock.Mock(return_value=True)
        proxy._rollout_steady_state = mock.Mock()
        proxy._rollout_buffered_recovery_batch = mock.Mock()
        with mock.patch("ray.get", side_effect=lambda value: value):
            for step in (1, 2):
                proxy.rollout(mock.Mock(), seed=step)
                self.assertEqual(proxy._rollout_steady_state.call_args.kwargs["target_groups"], 8)
            status.assert_not_called()
            proxy.rollout(mock.Mock(), seed=3)
            self.assertEqual(proxy._rollout_buffered_recovery_batch.call_args.kwargs["target_groups"], 10)
            status.return_value = {"recovery_buffer": {"available": 0}}
            proxy.rollout(mock.Mock(), seed=4)
            self.assertEqual(proxy._rollout_steady_state.call_args.kwargs["target_groups"], 10)
        with self.assertRaises(ValueError):
            proxy._reterm_initial_group_quota(8, 16)

    def test_initial_sampling_can_run_without_dynamic_sampling(self):
        proxy = object.__new__(ReTermAgentProxy)
        proxy.config = SimpleNamespace(
            recovery_env=SimpleNamespace(
                active_sampling={"enabled": False},
                recovery_buffer_initial_groups=8,
                recovery_buffer_recovery_groups=2,
                recovery_buffer_bootstrap_steps=2,
                recovery_buffer_bootstrap_groups=8,
            ),
            es_manager=SimpleNamespace(train=SimpleNamespace(env_groups=10, group_size=16)),
        )
        proxy._sandbox_workers = []
        proxy._task_sampling_seed = 1
        proxy._rollout_initial_groups = mock.Mock()
        proxy._rollout_steady_state = mock.Mock()
        proxy.rollout(mock.Mock(), seed=1)
        self.assertEqual(proxy._rollout_initial_groups.call_args.kwargs["env_groups"], 8)
        proxy._rollout_steady_state.assert_not_called()

    def test_buffered_batch_uses_static_initial_groups_when_dynamic_sampling_is_off(self):
        proxy = object.__new__(ReTermAgentProxy)
        proxy.config = SimpleNamespace(recovery_env=SimpleNamespace(
            active_sampling={"enabled": False},
            recovery_buffer_initial_groups=6,
            recovery_buffer_recovery_groups=2,
        ))
        proxy._reterm_lane_capacity = mock.Mock(return_value=(96, 32))
        proxy._new_sandbox_rollout_id = mock.Mock(return_value="rollout")
        proxy._dataset_name = mock.Mock(return_value="task")
        initial = [{"group_id": group, "env_id": index, "reward": 0.0}
                   for group in range(6) for index in range(group * 16, (group + 1) * 16)]
        recovery = [{"group_id": f"recovery:{group}", "env_id": index, "reward": 1.0}
                    for group in range(2) for index in range(group * 16, (group + 1) * 16)]
        proxy._rollout_initial_groups = mock.Mock(return_value=(initial, {}))
        proxy._rollout_steady_state = mock.Mock()
        proxy._collect_buffered_recovery_examples = mock.Mock(return_value=(recovery, {}))
        proxy._formulate_rollouts = mock.Mock(return_value=(SimpleNamespace(meta_info={}), {}))
        proxy._rollout_buffered_recovery_batch(
            mode_cfg=SimpleNamespace(), target_groups=8, group_size=16, rollout_seed=3,
        )
        self.assertEqual(proxy._rollout_initial_groups.call_args.kwargs["env_groups"], 6)
        proxy._rollout_steady_state.assert_not_called()

    def test_static_grpo_keeps_a_failure_from_a_training_group(self):
        proxy = object.__new__(ReTermAgentProxy)
        proxy.config = SimpleNamespace(recovery_env=SimpleNamespace(active_sampling={"enabled": False}))
        proxy._sandbox_workers = []
        proxy._dataset_name = mock.Mock(return_value="task")
        proxy._next_task_ids = mock.Mock(return_value=["task-id"])
        proxy._initial_worker_limit = mock.Mock(return_value=2)
        proxy._max_replacement_tasks = mock.Mock(return_value=0)
        proxy._is_trainable_example = mock.Mock(return_value=True)
        proxy._resolve_buffer_group = mock.Mock(return_value={"entry_id": "failure"})
        proxy._submit_episode = lambda executor, **kwargs: executor.submit(
            lambda: {"reward": 0.0, "metrics": {}, "task_id": "task-id", "group_id": 0}
        )
        examples, _metrics = proxy._rollout_initial_groups(
            mode_cfg=SimpleNamespace(max_workers=2), env_groups=1, group_size=2,
            rollout_seed=3, val=False, sandbox_rollout_id="rollout", return_examples=True,
        )
        self.assertEqual(len(examples), 2)
        self.assertTrue(any(call.kwargs["promote"] for call in proxy._resolve_buffer_group.call_args_list))

    def test_reterm_dispatches_after_bootstrap_when_four_states_exist(self):
        proxy = object.__new__(ReTermAgentProxy)
        proxy.config = SimpleNamespace(
            recovery_env=SimpleNamespace(
                recovery_buffer_bootstrap_steps=2,
            ),
            es_manager=SimpleNamespace(
                train=SimpleNamespace(env_groups=8, group_size=16),
            ),
        )
        worker = SimpleNamespace(status=SimpleNamespace(remote=mock.Mock(return_value={
            "recovery_buffer": {"available": 2},
        })))
        proxy._sandbox_workers = [worker, worker]
        proxy._task_sampling_seed = 1
        proxy._new_sandbox_rollout_id = mock.Mock(return_value="unused-bootstrap-id")
        proxy._rollout_buffered_recovery_batch = mock.Mock(return_value="buffer-batch")

        with mock.patch("ray.get", side_effect=lambda value: value):
            result = proxy.rollout(mock.Mock(), val=False, seed=3)

        self.assertEqual(result, "buffer-batch")
        proxy._rollout_buffered_recovery_batch.assert_called_once_with(
            mode_cfg=proxy.config.es_manager.train,
            target_groups=8,
            group_size=16,
            rollout_seed=3,
        )

    def test_reterm_cpu_end_to_end_builds_eight_independent_grpo_groups(self):
        self._check_reterm_batch(6, 2)

    def test_reterm_8plus2_cpu_end_to_end_and_fallback(self):
        for recovery_count in (0, 1, 2):
            with self.subTest(recovery_count=recovery_count):
                self._check_reterm_batch(8, recovery_count)

    def test_reterm_local_8plus2_cpu_batch_and_fallback(self):
        for recovery_count in (0, 1, 2):
            with self.subTest(recovery_count=recovery_count):
                self._check_reterm_batch(8, recovery_count, nodes=1)

    def test_reterm_local_empty_lease_does_not_contact_peer(self):
        proxy = object.__new__(ReTermAgentProxy)
        proxy._buffer_lease_lock = threading.Lock()
        proxy._sandbox_workers = [SimpleNamespace(
            lease_recovery_state=SimpleNamespace(remote=mock.Mock(return_value=None))
        )]
        with mock.patch("ray.get", side_effect=lambda value: value):
            self.assertIsNone(proxy._lease_buffer_state_for_node(
                target_node_index=0, rollout_seed=3, lane_index=0,
            ))

    def test_reterm_local_empty_collection(self):
        proxy = object.__new__(ReTermAgentProxy)
        proxy.config = SimpleNamespace(recovery_env=SimpleNamespace())
        proxy._sandbox_workers = [SimpleNamespace(status=SimpleNamespace(
            remote=mock.Mock(return_value={"recovery_buffer": {"available": 0, "leased": 0}})
        ))]
        proxy._sample_buffered_recovery_candidates = mock.Mock(return_value=[])
        with mock.patch("ray.get", side_effect=lambda value: value):
            rows, metrics = proxy._collect_buffered_recovery_examples(group_size=16, rollout_seed=3)
        self.assertEqual(rows, [])
        self.assertEqual(metrics["execution_slots"], 32)
        self.assertEqual(metrics["buffer_load_imbalance_after"], 0)
        self.assertEqual(proxy._sample_buffered_recovery_candidates.call_count, 1)
        self.assertEqual(proxy._sample_buffered_recovery_candidates.call_args.kwargs["candidate_count"], 2)

    def test_reterm_two_candidates_start_all_32_calls_before_waiting(self):
        proxy = object.__new__(ReTermAgentProxy)
        proxy._lease_buffer_state_for_node = mock.Mock(side_effect=[
            {"lease_id": f"lease{i}", "entry_id": f"entry{i}"} for i in range(2)
        ])
        barrier = threading.Barrier(32)
        def run(worker, rollout_id, lease_id, kwargs):
            barrier.wait(timeout=5)
            return {"reward": kwargs["sample_index"] % 2}
        proxy._get_distributed_buffered_recovery = run
        proxy._is_trainable_example = lambda row: True
        proxy._infrastructure_failure_category = lambda exc: None
        worker = SimpleNamespace(
            finish_rollout=SimpleNamespace(remote=mock.Mock()),
            finish_recovery_state=SimpleNamespace(remote=mock.Mock()),
        )
        with mock.patch("ray.get", side_effect=lambda value: value):
            groups = proxy._sample_buffered_recovery_candidates(
                worker=worker, node_index=0, lane_index=0,
                group_size=16, rollout_seed=3, candidate_count=2,
            )
        self.assertEqual([len(group["group"]) for group in groups], [16, 16])
        self.assertEqual([group["lease"]["lease_id"] for group in groups], ["lease0", "lease1"])
        self.assertEqual(worker.finish_rollout.remote.call_count, 2)

    def test_reterm_abort_drains_calls_before_returning_leases(self):
        proxy = object.__new__(ReTermAgentProxy)
        proxy._lease_buffer_state_for_node = mock.Mock(side_effect=[
            {"lease_id": f"lease{i}", "entry_id": f"entry{i}"} for i in range(2)
        ])
        barrier = threading.Barrier(32)
        cancelled = threading.Event()
        lock = threading.Lock()
        active = 0
        def run(worker, rollout_id, lease_id, kwargs):
            nonlocal active
            with lock:
                active += 1
            try:
                barrier.wait(timeout=5)
                if lease_id == "lease0" and kwargs["sample_index"] == 0:
                    raise ValueError("injected failure")
                assert cancelled.wait(timeout=5)
                return {"reward": 0}
            finally:
                with lock:
                    active -= 1
        def release(**kwargs):
            self.assertEqual(active, 0)
            self.assertFalse(kwargs["consume"])
        proxy._get_distributed_buffered_recovery = run
        proxy._is_trainable_example = lambda row: True
        proxy._infrastructure_failure_category = lambda exc: None
        worker = SimpleNamespace(
            cancel_rollout=SimpleNamespace(remote=mock.Mock(side_effect=lambda _: cancelled.set())),
            finish_rollout=SimpleNamespace(remote=mock.Mock()),
            finish_recovery_state=SimpleNamespace(remote=mock.Mock(side_effect=release)),
        )
        with mock.patch("ray.get", side_effect=lambda value: value):
            with self.assertRaisesRegex(ValueError, "injected failure"):
                proxy._sample_buffered_recovery_candidates(
                    worker=worker, node_index=0, lane_index=0,
                    group_size=16, rollout_seed=3, candidate_count=2,
                )
        self.assertEqual(worker.finish_recovery_state.remote.call_count, 2)

    def test_reterm_finish_rpc_failure_returns_leased_state(self):
        proxy = object.__new__(ReTermAgentProxy)
        proxy._lease_buffer_state_for_node = mock.Mock(
            side_effect=[{"lease_id": "lease", "entry_id": "entry"}]
        )
        proxy._get_distributed_buffered_recovery = mock.Mock(return_value={"reward": 1.0})
        proxy._is_trainable_example = lambda row: True
        worker = SimpleNamespace(
            finish_rollout=SimpleNamespace(remote=mock.Mock(side_effect=RuntimeError("finish failed"))),
            finish_recovery_state=SimpleNamespace(remote=mock.Mock(return_value={})),
        )
        with mock.patch("ray.get", side_effect=lambda value: value):
            with self.assertRaisesRegex(RuntimeError, "finish failed"):
                proxy._sample_buffered_recovery_candidates(
                    worker=worker, node_index=0, lane_index=0,
                    group_size=1, rollout_seed=3, candidate_count=1,
                )
        worker.finish_recovery_state.remote.assert_called_once_with(
            lease_id="lease", consume=False
        )

    def test_reterm_local_8plus4_cpu_batch_and_every_fallback(self):
        for recovery_count in range(5):
            with self.subTest(recovery_count=recovery_count):
                self._check_reterm_batch(8, recovery_count, nodes=1, recovery_quota=4)

    def test_reterm_8plus4_collects_four_groups_in_two_rounds(self):
        proxy = object.__new__(ReTermAgentProxy)
        proxy.config = SimpleNamespace(recovery_env=SimpleNamespace(recovery_buffer_recovery_groups=4))
        worker = SimpleNamespace(
            status=SimpleNamespace(remote=mock.Mock(return_value={
                "recovery_buffer": {"available": 4, "leased": 0, "entries": []}
            })),
            finish_recovery_state=SimpleNamespace(remote=mock.Mock(return_value={})),
        )
        proxy._sandbox_workers = [worker]
        proxy.trajectory_dir = None
        rounds = []
        for start in (0, 2):
            rounds.append([{
                "node_index": 0,
                "lease": {"lease_id": str(i), "entry_id": str(i)},
                "group": [{"group_id": str(i), "reward": float(j % 2)} for j in range(16)],
            } for i in range(start, start + 2)])
        proxy._sample_buffered_recovery_candidates = mock.Mock(side_effect=rounds)
        with mock.patch("ray.get", side_effect=lambda value: value):
            rows, metrics = proxy._collect_buffered_recovery_examples(group_size=16, rollout_seed=3)
        self.assertEqual(len(rows), 64)
        self.assertEqual(metrics["accepted_groups"], 4)
        self.assertEqual(proxy._sample_buffered_recovery_candidates.call_count, 2)
        self.assertEqual(worker.finish_recovery_state.remote.call_count, 4)

    def _check_reterm_batch(self, initial_count, recovery_count, nodes=2, recovery_quota=2):
        """Smoke the post-environment path without Ray, vLLM, or GPUs."""

        proxy = object.__new__(ReTermAgentProxy)
        proxy.tokenizer = SimpleNamespace(pad_token_id=0)
        proxy.config = SimpleNamespace(recovery_env=SimpleNamespace(
            recovery_buffer_initial_groups=initial_count,
            recovery_buffer_recovery_groups=recovery_quota,
        ))
        proxy._sandbox_workers = [object() for _ in range(nodes)]
        proxy._dataset_name = lambda: "task"
        proxy._actor_data_parallel_size = lambda: 2 * nodes
        proxy._new_sandbox_rollout_id = mock.Mock(return_value="initial-rollout")
        proxy._steady_state_sampling_enabled = mock.Mock(return_value=True)

        def make_groups(prefix, count):
            rows = []
            for group_index in range(count):
                group_id = f"{prefix}:{group_index}"
                for sample_index in range(16):
                    reward = float(sample_index % 2)
                    rows.append({
                        "env_id": f"{group_id}:{sample_index}",
                        "group_id": group_id,
                        "task_id": group_id,
                        "reward": reward,
                        "metric_valid": True,
                        "messages": [{
                            "role": "assistant",
                            "content": "sampled",
                            "official_qwen_segment": {
                                "prompt_ids": [10, 11],
                                "response_ids": [20 + sample_index],
                                "old_log_probs": [-0.1],
                            },
                        }],
                        "training_start_index": 0,
                    })
            return rows

        initial = make_groups("initial", initial_count)
        recovery = make_groups("buffer", recovery_count)
        proxy._rollout_steady_state = mock.Mock(side_effect=[
            (initial, {"accepted": float(initial_count)}),
            (make_groups("initial", recovery_quota - recovery_count), {}),
        ])
        proxy._collect_buffered_recovery_examples = mock.Mock(
            return_value=(recovery, {"accepted": 2.0})
        )

        batch = proxy._rollout_buffered_recovery_batch(
            mode_cfg=SimpleNamespace(),
            target_groups=initial_count + recovery_quota,
            group_size=16,
            rollout_seed=3,
        )
        advantage_metrics = compute_official_attempt_grpo_advantage(
            batch, normalize_by_std=True
        )

        attempts = (initial_count + recovery_quota) * 16
        self.assertEqual(len(set(batch.non_tensor_batch["group_ids"])), initial_count + recovery_quota)
        self.assertEqual(len(set(batch.non_tensor_batch["attempt_ids"])), attempts)
        self.assertEqual(advantage_metrics["train/official_attempt_grpo_attempts"], float(attempts))
        self.assertEqual(float(batch.batch["response_mask"].sum()), float(attempts))
        self.assertEqual(batch.meta_info["metrics"]["reterm/initial_train_groups"], initial_count + recovery_quota - recovery_count)
        self.assertEqual(batch.meta_info["metrics"]["reterm/recovery_train_groups"], recovery_count)
        self.assertEqual(proxy._rollout_steady_state.call_args_list[0].kwargs["target_groups"], initial_count)
        self.assertEqual(proxy._rollout_steady_state.call_args_list[0].kwargs["max_workers_override"], nodes * 96)
        if recovery_count < recovery_quota:
            self.assertEqual(proxy._rollout_steady_state.call_args_list[1].kwargs["max_workers_override"], nodes * 32)
        self.assertTrue(torch.equal(batch.batch["loss_mask"], batch.batch["response_mask"]))
        self.assertTrue(torch.all(batch.batch["advantages"][batch.batch["response_mask"] > 0] != 0))

    def test_reterm_exhaustion_fills_missing_recovery_group_with_initial_group(self):
        """A partial Recovery lane must still produce exactly eight GRPO groups."""

        proxy = object.__new__(ReTermAgentProxy)
        proxy.tokenizer = SimpleNamespace(pad_token_id=0)
        proxy.config = SimpleNamespace(recovery_env=SimpleNamespace())
        proxy._sandbox_workers = [object(), object()]
        proxy._dataset_name = lambda: "task"
        proxy._actor_data_parallel_size = lambda: 4
        proxy._new_sandbox_rollout_id = mock.Mock(
            side_effect=["initial-rollout", "fallback-rollout"]
        )
        proxy._steady_state_sampling_enabled = mock.Mock(return_value=True)

        def make_groups(prefix, count):
            rows = []
            for group_index in range(count):
                group_id = f"{prefix}:{group_index}"
                for sample_index in range(16):
                    rows.append({
                        "env_id": f"{group_id}:{sample_index}",
                        "group_id": group_id,
                        "task_id": group_id,
                        "reward": float(sample_index % 2),
                        "metric_valid": True,
                        "messages": [{
                            "role": "assistant",
                            "content": "sampled",
                            "official_qwen_segment": {
                                "prompt_ids": [10, 11],
                                "response_ids": [20 + sample_index],
                                "old_log_probs": [-0.1],
                            },
                        }],
                        "training_start_index": 0,
                    })
            return rows

        initial = make_groups("initial", 6)
        recovery = make_groups("buffer", 1)
        # Deliberately reuse group_id=initial:0 in the fallback result. The
        # scheduler must namespace it before optimizer formulation.
        fallback = make_groups("initial", 1)
        proxy._rollout_steady_state = mock.Mock(
            side_effect=[
                (initial, {"accepted": 6.0}),
                (fallback, {"accepted": 1.0}),
            ]
        )
        proxy._collect_buffered_recovery_examples = mock.Mock(
            return_value=(recovery, {"accepted_groups": 1.0})
        )

        batch = proxy._rollout_buffered_recovery_batch(
            mode_cfg=SimpleNamespace(),
            target_groups=8,
            group_size=16,
            rollout_seed=3,
        )
        advantage_metrics = compute_official_attempt_grpo_advantage(
            batch, normalize_by_std=True
        )

        group_ids = set(batch.non_tensor_batch["group_ids"])
        self.assertEqual(len(group_ids), 8)
        self.assertEqual(len(set(batch.non_tensor_batch["attempt_ids"])), 128)
        self.assertTrue(any(str(value).startswith("buffer-fallback:3:") for value in group_ids))
        self.assertEqual(advantage_metrics["train/official_attempt_grpo_attempts"], 128.0)
        self.assertEqual(batch.meta_info["metrics"]["reterm/initial_train_groups"], 7.0)
        self.assertEqual(batch.meta_info["metrics"]["reterm/recovery_train_groups"], 1.0)
        self.assertEqual(
            batch.meta_info["metrics"]["reterm/fallback_initial_train_groups"], 1.0
        )
        second_call = proxy._rollout_steady_state.call_args_list[1]
        self.assertEqual(second_call.kwargs["target_groups"], 1)
        self.assertEqual(second_call.kwargs["sandbox_rollout_id"], "fallback-rollout")

    def test_reterm_empty_recovery_buffer_returns_partial_batch_for_fallback(self):
        proxy = object.__new__(ReTermAgentProxy)
        proxy.config = SimpleNamespace(recovery_env=SimpleNamespace())
        proxy._sandbox_workers = [
            SimpleNamespace(status=SimpleNamespace(remote=mock.Mock(return_value={
                "recovery_buffer": {"available": 0, "leased": 0},
            }))),
            SimpleNamespace(status=SimpleNamespace(remote=mock.Mock(return_value={
                "recovery_buffer": {"available": 0, "leased": 0},
            }))),
        ]
        proxy._sample_buffered_recovery_candidates = mock.Mock(return_value=[])

        with mock.patch("ray.get", side_effect=lambda value: value):
            examples, metrics = proxy._collect_buffered_recovery_examples(
                group_size=16,
                rollout_seed=3,
            )

        self.assertEqual(examples, [])
        self.assertEqual(metrics["accepted_groups"], 0.0)
        self.assertEqual(metrics["buffer_exhausted"], 1.0)

    def test_reterm_completed_unselected_state_is_requeued_and_reported(self):
        def rpc(**kwargs):
            return SimpleNamespace(remote=mock.Mock(**kwargs))

        def finish(*, lease_id, consume):
            return {
                "lease_id": lease_id,
                "consumed": consume,
                "requeued": not consume,
                "evicted_on_requeue": False,
            }

        entries = [
            {"entry_id": f"entry-{index}", "created_step": 1}
            for index in range(4)
        ]
        workers = [
            SimpleNamespace(
                status=rpc(return_value={
                    "recovery_buffer": {
                        "available": 2,
                        "leased": 0,
                        "entries": entries[node_index * 2 : node_index * 2 + 2],
                    }
                }),
                finish_recovery_state=rpc(side_effect=finish),
            )
            for node_index in range(2)
        ]

        def mixed_group(group_id):
            return [
                {"group_id": group_id, "reward": 0.0},
                {"group_id": group_id, "reward": 1.0},
            ]

        candidates = {
            0: [
                {"node_index": 0, "lease": {"lease_id": "l0", "entry_id": "entry-0"}, "group": mixed_group("g0")},
                {"node_index": 0, "lease": {"lease_id": "l1", "entry_id": "entry-1"}, "group": mixed_group("g1")},
            ],
            1: [
                {"node_index": 1, "lease": {"lease_id": "l2", "entry_id": "entry-2"}, "group": mixed_group("g2")},
                {"node_index": 1, "lease": {"lease_id": "l3", "entry_id": "entry-3"}, "group": [
                    {"group_id": "g3", "reward": 0.0},
                    {"group_id": "g3", "reward": 0.0},
                ]},
            ],
        }
        proxy = object.__new__(ReTermAgentProxy)
        proxy.config = SimpleNamespace(recovery_env=SimpleNamespace())
        proxy._sandbox_workers = workers
        proxy.trajectory_dir = None
        proxy._sample_buffered_recovery_candidates = mock.Mock(
            side_effect=lambda **kwargs: candidates[kwargs["node_index"]]
        )

        with mock.patch("ray.get", side_effect=lambda value: value):
            examples, metrics = proxy._collect_buffered_recovery_examples(
                group_size=2,
                rollout_seed=3,
            )

        self.assertEqual(len(examples), 4)
        self.assertEqual(metrics["accepted_groups"], 2.0)
        self.assertEqual(metrics["states_consumed"], 3.0)
        self.assertEqual(metrics["states_requeued"], 1.0)
        self.assertEqual(metrics["states_requeued_completed_unselected"], 1.0)
        self.assertEqual(metrics["states_requeued_infra_or_invalid"], 0.0)
        self.assertEqual(metrics["states_evicted_on_requeue"], 0.0)
        self.assertEqual(metrics["buffer_eligible_before"], 4.0)

    def test_all_failed_recovery_switches_to_next_state_until_mixed_group(self):
        def rpc(**kwargs):
            return SimpleNamespace(remote=mock.Mock(**kwargs))

        def finish(*, lease_id, consume):
            return {"consumed": consume, "requeued": not consume, "evicted_on_requeue": False}

        entries = [{"entry_id": f"e{i}", "created_step": 1} for i in range(6)]
        workers = [
            SimpleNamespace(
                status=rpc(return_value={"recovery_buffer": {
                    "available": 3, "leased": 0, "entries": entries[i * 3 : i * 3 + 3],
                }}),
                finish_recovery_state=rpc(side_effect=finish),
            ) for i in range(2)
        ]

        def group(group_id, successes):
            return [
                {"group_id": group_id, "reward": float(i < successes)}
                for i in range(16)
            ]

        calls = {0: 0, 1: 0}

        def sample(**kwargs):
            node = kwargs["node_index"]
            round_index = calls[node]
            calls[node] += 1
            if round_index > 1:
                return []
            return [
                {"node_index": node, "lease": {"lease_id": f"l{node}-{round_index}-{i}",
                 "entry_id": f"e{node * 3 + round_index * 2 + i}"},
                 "group": group(f"g{node}-{round_index}-{i}", 0 if round_index == 0 else 1)}
                for i in range(2)
            ]

        proxy = object.__new__(ReTermAgentProxy)
        proxy.config = SimpleNamespace(recovery_env=SimpleNamespace())
        proxy._sandbox_workers = workers
        proxy.trajectory_dir = None
        proxy._sample_buffered_recovery_candidates = sample

        with mock.patch("ray.get", side_effect=lambda value: value):
            rows, metrics = proxy._collect_buffered_recovery_examples(group_size=16, rollout_seed=3)

        self.assertEqual(len(rows), 32)
        self.assertTrue(all("-1-" in row["group_id"] for row in rows))
        self.assertEqual(metrics["all_fail_groups"], 4.0)
        self.assertEqual(metrics["accepted_groups"], 2.0)

    def test_reterm_continues_after_invalid_round_until_later_mixed_states(self):
        def rpc(**kwargs):
            return SimpleNamespace(remote=mock.Mock(**kwargs))

        def finish(*, lease_id, consume):
            return {
                "lease_id": lease_id,
                "consumed": consume,
                "requeued": not consume,
                "evicted_on_requeue": False,
            }

        entries = [
            {"entry_id": f"entry-{index}", "created_step": 1}
            for index in range(8)
        ]
        workers = [
            SimpleNamespace(
                status=rpc(return_value={
                    "recovery_buffer": {
                        "available": 4,
                        "leased": 0,
                        "entries": entries[node_index * 4 : node_index * 4 + 4],
                    }
                }),
                finish_recovery_state=rpc(side_effect=finish),
            )
            for node_index in range(2)
        ]
        calls = {0: 0, 1: 0}

        def sample(**kwargs):
            node = kwargs["node_index"]
            round_index = calls[node]
            calls[node] += 1
            base = node * 4 + round_index * 2
            result = []
            for offset in range(2):
                index = base + offset
                lease = {"lease_id": f"l{index}", "entry_id": f"entry-{index}"}
                if round_index == 0:
                    result.append({
                        "node_index": node,
                        "lease": lease,
                        "group": None,
                        "failure_reason": "infrastructure_sandbox",
                        "failure_detail": "synthetic restore failure",
                    })
                else:
                    result.append({
                        "node_index": node,
                        "lease": lease,
                        "group": [
                            {"group_id": f"g{index}", "reward": 0.0},
                            {"group_id": f"g{index}", "reward": 1.0},
                        ],
                    })
            if round_index == 1:
                self.assertEqual(
                    set(kwargs["excluded_entry_ids"]),
                    {"entry-0", "entry-1", "entry-4", "entry-5"},
                )
            return result

        proxy = object.__new__(ReTermAgentProxy)
        proxy.config = SimpleNamespace(recovery_env=SimpleNamespace())
        proxy._sandbox_workers = workers
        proxy.trajectory_dir = None
        proxy._sample_buffered_recovery_candidates = mock.Mock(side_effect=sample)

        with mock.patch("ray.get", side_effect=lambda value: value):
            examples, metrics = proxy._collect_buffered_recovery_examples(
                group_size=2,
                rollout_seed=3,
            )

        self.assertEqual(len(examples), 4)
        self.assertEqual(metrics["accepted_groups"], 2.0)
        self.assertEqual(metrics["infra_only_sampling_rounds"], 1.0)
        self.assertEqual(metrics["states_requeued_infra_or_invalid"], 4.0)
        self.assertEqual(metrics["unique_states_attempted"], 8.0)

    def test_reterm_zero_recovery_groups_degrades_to_eight_initial_groups(self):
        proxy = object.__new__(ReTermAgentProxy)
        proxy.config = SimpleNamespace(recovery_env=SimpleNamespace())
        proxy._dataset_name = lambda: "task"
        proxy._sandbox_workers = [object(), object()]
        proxy._new_sandbox_rollout_id = mock.Mock(
            side_effect=["initial-rollout", "fallback-rollout"]
        )
        proxy._steady_state_sampling_enabled = mock.Mock(return_value=True)

        def make_groups(prefix, count):
            return [
                {
                    "group_id": f"{prefix}:{group_index}",
                    "env_id": f"{prefix}:{group_index}:{sample_index}",
                }
                for group_index in range(count)
                for sample_index in range(16)
            ]

        initial = make_groups("initial", 6)
        fallback = make_groups("initial", 2)
        proxy._rollout_steady_state = mock.Mock(
            side_effect=[
                (initial, {"accepted": 6.0}),
                (fallback, {"accepted": 2.0}),
            ]
        )
        proxy._collect_buffered_recovery_examples = mock.Mock(
            return_value=([], {"accepted_groups": 0.0, "buffer_exhausted": 1.0})
        )
        batch = SimpleNamespace(meta_info={})
        proxy._formulate_rollouts = mock.Mock(return_value=(batch, {}))

        result = proxy._rollout_buffered_recovery_batch(
            mode_cfg=SimpleNamespace(),
            target_groups=8,
            group_size=16,
            rollout_seed=3,
        )

        examples = proxy._formulate_rollouts.call_args.args[0]
        self.assertIs(result, batch)
        self.assertEqual(len(examples), 128)
        self.assertEqual(len({row["group_id"] for row in examples}), 8)
        self.assertTrue(
            all(
                str(row["group_id"]).startswith("buffer-fallback:3:")
                for row in examples[96:]
            )
        )
        self.assertEqual(result.meta_info["metrics"]["reterm/initial_train_groups"], 8.0)
        self.assertEqual(result.meta_info["metrics"]["reterm/recovery_train_groups"], 0.0)
        self.assertEqual(
            result.meta_info["metrics"]["reterm/fallback_initial_train_groups"], 2.0
        )

    def test_recovery_state_leases_stay_with_the_worker(self):
        worker = SimpleNamespace(
            lease_recovery_state=SimpleNamespace(remote=mock.Mock(return_value={"lease_id": "local"}))
        )
        proxy = object.__new__(ReTermAgentProxy)
        proxy._buffer_lease_lock = threading.Lock()
        proxy._sandbox_workers = [worker]
        with mock.patch("ray.get", side_effect=lambda value: value):
            lease = proxy._lease_buffer_state_for_node(
                target_node_index=0, rollout_seed=3, lane_index=0
            )
        self.assertEqual(lease, {"lease_id": "local"})
        worker.lease_recovery_state.remote.assert_called_once_with(
            max_created_step=2, excluded_entry_ids=[]
        )

    def test_dataset_cancellation_watcher_interrupts_live_environment(self):
        cancelled = threading.Event()
        stopped = threading.Event()
        cleaned = threading.Event()
        dataset = SimpleNamespace(cancel_active_environment=cleaned.set)
        watcher = threading.Thread(
            target=ReTermAgentProxy._watch_dataset_cancellation,
            args=(dataset, cancelled, stopped),
        )
        watcher.start()
        cancelled.set()
        watcher.join(timeout=2)
        self.assertFalse(watcher.is_alive())
        self.assertTrue(cleaned.is_set())

    def test_dataset_cancellation_watcher_does_not_clean_after_normal_stop(self):
        cancelled = threading.Event()
        stopped = threading.Event()
        cleaned = threading.Event()
        dataset = SimpleNamespace(cancel_active_environment=cleaned.set)
        stopped.set()
        cancelled.set()
        ReTermAgentProxy._watch_dataset_cancellation(dataset, cancelled, stopped)
        self.assertFalse(cleaned.is_set())

    def test_slow_generation_observes_explicit_cancel_event(self):
        proxy = object.__new__(ReTermAgentProxy)
        proxy.config = SimpleNamespace(
            actor_rollout_ref=SimpleNamespace(
                rollout=SimpleNamespace(
                    calculate_log_probs=True,
                    temperature=0.6,
                    top_p=1.0,
                    top_k=-1,
                )
            )
        )
        proxy._generation_seed = threading.local()
        proxy._generation_turn = threading.local()
        proxy._generation_seed.value = None
        proxy._bounded_llm_timeout = lambda value: value
        proxy._generation_max_tokens = lambda **_kwargs: 16

        async def slow_generate(**_kwargs):
            await asyncio.sleep(60)

        async def pending_once(*_args, **_kwargs):
            return set(), set()

        aborted: list[tuple[str, str | None, bool]] = []

        async def abort_request(request_id, *, backend_request_id=None, reset_prefix_cache=True):
            aborted.append((request_id, backend_request_id, reset_prefix_cache))
            return {"aborted": True}

        proxy.llm_client = SimpleNamespace(generate=slow_generate, abort_request=abort_request)
        cancelled = threading.Event()
        cancelled.set()
        with mock.patch("reterm_rl.llm_agent.recovery_proxy.asyncio.wait", new=pending_once):
            with self.assertRaises(RolloutCancelledError):
                proxy._generate_response_with_llm_client([1, 2], cancel_event=cancelled)
        self.assertEqual(len(aborted), 1)
        self.assertEqual(aborted[0][0], aborted[0][1])
        self.assertTrue(aborted[0][0].startswith("reterm-"))
        self.assertFalse(aborted[0][2])

    def test_native_history_records_only_executed_outer_tool_call(self):
        agent = TerminalPolicyAgent(
            policy=lambda _messages, _context: "",
            config=TerminalPolicyAgentConfig(
            ),
        )
        raw = (
            "inspect first\n</think>\n<tool_call>\n<function=bash>\n"
            "<parameter=command>\npwd\n</parameter>\n</function>\n</tool_call>"
        )
        action = agent._parse_qwen_xml_tool_action(raw)
        self.assertIsNotNone(action)
        action.tool_call_id = "call-1"
        native = agent._qwen_native_assistant_message(raw, action=action)
        self.assertEqual(native["content"], "")
        self.assertEqual(native["reasoning_content"], "inspect first")
        self.assertEqual(native["tool_calls"][0]["function"]["arguments"], {"command": "pwd"})

    def test_tool_call_inside_unclosed_think_is_format_error_not_native_call(self):
        agent = TerminalPolicyAgent(
            policy=lambda _messages, _context: "",
            config=TerminalPolicyAgentConfig(
            ),
        )
        raw = (
            "reasoning puts the action here <tool_call><function=bash>"
            "<parameter=command>pwd</parameter></function></tool_call>"
        )
        self.assertIsNone(agent._parse_qwen_xml_tool_action(raw))
        native = agent._qwen_native_assistant_message(raw)
        self.assertEqual(native["content"], "")
        self.assertIn("<tool_call>", native["reasoning_content"])
        self.assertNotIn("tool_calls", native)

    def test_agent_records_native_assistant_and_tool_messages_online(self):
        responses = iter(
            [
                PolicyResponse(
                    content=(
                        "inspect\n</think>\n<tool_call>\n<function=bash>\n"
                        "<parameter=command>\npwd\n</parameter>\n</function>\n</tool_call>"
                    ),
                    token_ids=[1],
                    log_probs=[-0.1],
                    metadata={"official_qwen_segment": {
                        "prompt_ids": [10], "response_ids": [1], "old_log_probs": [-0.1]
                    }},
                ),
                PolicyResponse(
                    content=(
                        "verified\n</think>\n<tool_call>\n<function=done>\n</function>\n</tool_call>"
                    ),
                    token_ids=[2],
                    log_probs=[-0.2],
                    metadata={"official_qwen_segment": {
                        "prompt_ids": [10, 11], "response_ids": [2], "old_log_probs": [-0.2]
                    }},
                ),
            ]
        )
        agent = TerminalPolicyAgent(
            policy=lambda _messages, _context: next(responses),
            config=TerminalPolicyAgentConfig(
                max_turns_per_attempt=2,
            ),
        )
        environment = SimpleNamespace(
            exec=lambda _command, timeout=None: CommandResult(True, "/work", 0)
        )
        result = agent.run(
            Task(task_id="task", prompt="task"),
            "task",
            environment,
            AgentContext(benchmark="test", task_id="task", attempt_index=1),
        )
        messages = result.metadata["messages"]
        first_assistant = messages[2]["native_message"]
        first_tool = messages[3]["native_message"]
        self.assertEqual(first_assistant["content"], "")
        self.assertEqual(first_assistant["reasoning_content"], "inspect")
        self.assertEqual(first_assistant["tool_calls"][0]["function"]["name"], "bash")
        self.assertEqual(first_tool["role"], "tool")
        self.assertEqual(first_tool["tool_call_id"], first_assistant["tool_calls"][0]["id"])
        self.assertEqual(messages[4]["native_message"]["tool_calls"][0]["function"]["name"], "done")

    def test_tool_schema_matches_sft(self):
        self.assertEqual(TERMINAL_TOOLS[0]["function"]["description"],
                         "Execute exactly one non-interactive shell command in the task environment.")
        self.assertFalse(TERMINAL_TOOLS[0]["function"]["parameters"]["additionalProperties"])
        self.assertEqual(TERMINAL_TOOLS[1]["function"]["description"],
                         "Submit the current task state for official evaluation.")

    def test_segments_preserve_per_generation_prompt_and_logprob_provenance(self):
        example = {
            "training_start_index": 2,
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "task"},
                {
                    "role": "assistant",
                    "content": "first",
                    "official_qwen_segment": {
                        "prompt_ids": [1, 2],
                        "response_ids": [3, 4],
                        "old_log_probs": [-0.1, -0.2],
                    },
                },
                {"role": "user", "content": "recover"},
                {
                    "role": "assistant",
                    "content": "second",
                    "official_qwen_segment": {
                        "prompt_ids": [8, 9, 10],
                        "response_ids": [11],
                        "old_log_probs": [-0.3],
                    },
                },
            ],
        }
        segments = ReTermAgentProxy._official_segments_for_example(example)
        self.assertEqual(segments[0], ([1, 2], [3, 4], [-0.1, -0.2]))
        self.assertEqual(segments[1], ([8, 9, 10], [11], [-0.3]))

    def test_recovery_attempt_node_does_not_retrain_initial_attempt_tokens(self):
        example = {
            "training_start_index": 5,
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "task"},
                {"role": "assistant", "content": "initial", "official_qwen_segment": {
                    "prompt_ids": [1], "response_ids": [2], "old_log_probs": [-0.1]
                }},
                {"role": "user", "content": "initial observation"},
                {"role": "user", "content": "recover"},
                {"role": "assistant", "content": "recovery", "official_qwen_segment": {
                    "prompt_ids": [1, 2, 3], "response_ids": [4], "old_log_probs": [-0.2]
                }},
            ],
        }

        segments = ReTermAgentProxy._official_segments_for_example(example)

        self.assertEqual(segments, [([1, 2, 3], [4], [-0.2])])

    def test_buffer_recovery_history_is_context_but_never_an_action_target(self):
        proxy = object.__new__(ReTermAgentProxy)
        proxy.tokenizer = SimpleNamespace(pad_token_id=0)
        proxy._dataset_name = lambda: "task"
        proxy._actor_data_parallel_size = lambda: 1
        example = {
            "env_id": "buffer-sample",
            "group_id": "buffer:state",
            "task_id": "task",
            "reward": 1.0,
            "loss_weight": 0.5,
            "metric_valid": True,
            "training_start_index": 5,
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "task"},
                {"role": "assistant", "content": "initial", "official_qwen_segment": {
                    "prompt_ids": [10, 11], "response_ids": [12, 13],
                    "old_log_probs": [-0.1, -0.2],
                }},
                {"role": "user", "content": "observation"},
                {"role": "user", "content": "recover"},
                {"role": "assistant", "content": "recovery", "official_qwen_segment": {
                    # The official prompt contains the earlier attempt, but
                    # only response IDs 24 and 25 are newly sampled actions.
                    "prompt_ids": [10, 11, 12, 13, 20, 21, 22, 23],
                    "response_ids": [24, 25],
                    "old_log_probs": [-0.3, -0.4],
                }},
            ],
        }

        data, _metrics = proxy._formulate_official_segment_rollouts([example])

        self.assertEqual(data.batch["input_ids"][0].tolist(), [10, 11, 12, 13, 20, 21, 22, 23, 24, 25])
        self.assertEqual(int(data.batch["response_mask"].sum()), 2)
        selected = data.batch["responses"][0][data.batch["response_mask"][0]]
        self.assertEqual(selected.tolist(), [24, 25])
        self.assertEqual(data.batch["rm_scores"].sum().item(), 1.0)
        self.assertEqual(data.non_tensor_batch["attempt_loss_weights"].tolist(), [0.5])

    def test_official_renderer_consumes_native_history_not_raw_xml(self):
        proxy = object.__new__(ReTermAgentProxy)
        captured = {}
        proxy._apply_qwen_chat_template_token_ids = lambda messages, add_generation_prompt: (
            captured.update(messages=messages, add_generation_prompt=add_generation_prompt) or [1]
        )
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
            {
                "role": "assistant",
                "content": "raw sampled XML retained only for audit",
                "native_message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "inspect",
                    "tool_calls": [{
                        "id": "c1", "type": "function",
                        "function": {"name": "bash", "arguments": {"command": "pwd"}},
                    }],
                },
            },
            {
                "role": "user",
                "content": "<tool_response>/work</tool_response>",
                "native_message": {"role": "tool", "content": "/work", "tool_call_id": "c1"},
            },
        ]
        self.assertEqual(proxy._render_official_qwen_prompt_token_ids(messages), [1])
        self.assertEqual(captured["messages"][2], messages[2]["native_message"])
        self.assertEqual(captured["messages"][3]["role"], "tool")
        self.assertNotIn("raw sampled XML", str(captured["messages"]))
        self.assertTrue(captured["add_generation_prompt"])

    def test_segment_formulation_places_attempt_reward_only_on_final_segment(self):
        proxy = object.__new__(ReTermAgentProxy)
        proxy.tokenizer = SimpleNamespace(pad_token_id=0)
        proxy._dataset_name = lambda: "task"
        proxy._actor_data_parallel_size = lambda: 1
        example = {
            "env_id": 7,
            "group_id": 2,
            "task_id": "task",
            "reward": 0.7,
            "metric_valid": True,
            "training_start_index": 2,
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "task"},
                {"role": "assistant", "content": "a", "official_qwen_segment": {
                    "prompt_ids": [10, 11], "response_ids": [12, 13], "old_log_probs": [-0.1, -0.2]
                }},
                {"role": "user", "content": "recover"},
                {"role": "assistant", "content": "b", "official_qwen_segment": {
                    "prompt_ids": [20], "response_ids": [21], "old_log_probs": [-0.3]
                }},
            ],
        }
        data, metrics = proxy._formulate_official_segment_rollouts([example])
        self.assertEqual(len(data), 2)
        rewards = data.batch["rm_scores"].sum(dim=-1).tolist()
        self.assertEqual(rewards[0], 0.0)
        self.assertAlmostEqual(rewards[1], 0.7, places=6)
        self.assertEqual(data.non_tensor_batch["attempt_ids"].tolist(), ["2:7", "2:7"])
        self.assertEqual(data.non_tensor_batch["metric_valid"].tolist(), [False, True])
        self.assertEqual(data.non_tensor_batch["metric_counted"].tolist(), [False, True])
        self.assertEqual(metrics["official_qwen_raw_segments_per_attempt"], 2.0)
        self.assertEqual(metrics["official_qwen_training_rows_per_attempt"], 2.0)

    def test_exact_prefix_compatible_segments_coalesce_without_changing_action_mask(self):
        segments = [
            ([10, 11], [12, 13], [-0.1, -0.2]),
            ([10, 11, 12, 13, 14], [15], [-0.3]),
            ([10, 11, 12, 13, 14, 15, 16], [17, 18], [-0.4, -0.5]),
        ]

        rows, metrics = ReTermAgentProxy._coalesce_official_segments(segments)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["token_ids"], list(range(10, 19)))
        self.assertEqual(
            rows[0]["mask"],
            [False, False, True, True, False, True, False, True, True],
        )
        self.assertEqual(
            rows[0]["old_log_probs"],
            [0.0, 0.0, -0.1, -0.2, 0.0, -0.3, 0.0, -0.4, -0.5],
        )
        self.assertEqual(rows[0]["segment_indices"], [0, 1, 2])
        self.assertEqual(metrics["exact_prefix_merges"], 2)
        self.assertEqual(metrics["incompatible_boundaries"], 0)
        self.assertEqual(metrics["reused_prefix_tokens"], 10)

    def test_history_rewrite_starts_new_row_instead_of_guessing_a_merge(self):
        segments = [
            ([1, 2], [3, 4], [-0.1, -0.2]),
            # Shares a prefix with the first prompt but drops sampled token 4.
            ([1, 2, 3, 9], [5], [-0.3]),
        ]

        rows, metrics = ReTermAgentProxy._coalesce_official_segments(segments)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["token_ids"], [1, 2, 3, 4])
        self.assertEqual(rows[1]["token_ids"], [1, 2, 3, 9, 5])
        self.assertEqual(metrics["exact_prefix_merges"], 0)
        self.assertEqual(metrics["incompatible_boundaries"], 1)

    def test_formulation_coalesces_compatible_rows_and_adds_only_masked_dp_padding(self):
        proxy = object.__new__(ReTermAgentProxy)
        proxy.tokenizer = SimpleNamespace(pad_token_id=0)
        proxy._dataset_name = lambda: "task"
        proxy._actor_data_parallel_size = lambda: 2
        example = {
            "env_id": 1,
            "group_id": 0,
            "task_id": "task",
            "reward": 1.0,
            "metric_valid": True,
            "training_start_index": 2,
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "task"},
                {"role": "assistant", "content": "a", "official_qwen_segment": {
                    "prompt_ids": [10, 11], "response_ids": [12], "old_log_probs": [-0.1]
                }},
                {"role": "user", "content": "tool"},
                {"role": "assistant", "content": "b", "official_qwen_segment": {
                    "prompt_ids": [10, 11, 12, 13], "response_ids": [14], "old_log_probs": [-0.2]
                }},
            ],
        }

        data, metrics = proxy._formulate_official_segment_rollouts([example])

        self.assertEqual(len(data), 2)
        self.assertEqual(metrics["official_qwen_raw_segments"], 2.0)
        self.assertEqual(metrics["official_qwen_training_rows"], 1.0)
        self.assertEqual(metrics["official_qwen_training_padding_rows"], 1.0)
        self.assertEqual(data.non_tensor_batch["training_padding"].tolist(), [False, True])
        self.assertEqual(int(data.batch["response_mask"][0].sum()), 2)
        self.assertEqual(int(data.batch["response_mask"][1].sum()), 0)
        # Causal shift: targets are input_ids[:, 1:]. Only sampled IDs 12 and
        # 14 carry their exact rollout logprobs and enter the policy loss.
        self.assertEqual(data.batch["responses"][0].tolist(), [11, 12, 13, 14])
        self.assertEqual(data.batch["response_mask"][0].tolist(), [False, True, False, True])
        self.assertTrue(
            torch.allclose(
                data.batch["old_log_probs"][0],
                torch.tensor([0.0, -0.1, 0.0, -0.2]),
            )
        )
        self.assertEqual(data.non_tensor_batch["metric_counted"].tolist(), [True, False])


class OfficialAttemptGrpoAdvantageTest(unittest.TestCase):
    def test_initial_and_buffer_recovery_rewards_never_normalize_across_groups(self):
        response_mask = torch.ones((6, 1), dtype=torch.bool)
        batch = DataProto()
        batch.batch = TensorDict({"response_mask": response_mask}, batch_size=6)
        batch.non_tensor_batch = {
            "attempt_ids": np.array(["i0", "i1", "r0", "r1", "r2", "r3"], dtype=object),
            "attempt_rewards": np.array([1, 0, 1, 1, 1, 0], dtype=object),
            "group_ids": np.array(
                ["initial", "initial", "buffer:state", "buffer:state", "buffer:state", "buffer:state"],
                dtype=object,
            ),
        }

        compute_official_attempt_grpo_advantage(batch, normalize_by_std=False)

        values = batch.batch["advantages"][:, 0].tolist()
        self.assertEqual(values[:2], [0.5, -0.5])
        self.assertEqual(values[2:], [0.25, 0.25, 0.25, -0.75])

    def test_recovery_loss_weight_normalizes_by_sum_without_changing_reward(self):
        batch = DataProto()
        batch.batch = TensorDict({"response_mask": torch.ones((4, 1), dtype=torch.bool)}, batch_size=4)
        batch.non_tensor_batch = {
            "attempt_ids": np.array(["i0", "i1", "r0", "r1"], dtype=object),
            "attempt_rewards": np.array([1.0, 0.0, 1.0, 0.0], dtype=object),
            "attempt_loss_weights": np.array([1.0, 1.0, 0.5, 0.5], dtype=object),
            "group_ids": np.array(["initial", "initial", "recovery", "recovery"], dtype=object),
        }

        compute_official_attempt_grpo_advantage(batch, normalize_by_std=False)

        torch.testing.assert_close(
            batch.batch["advantages"][:, 0], torch.tensor([2/3, -2/3, 1/3, -1/3])
        )
        self.assertEqual(batch.non_tensor_batch["attempt_rewards"].tolist(), [1.0, 0.0, 1.0, 0.0])

    def test_weighted_attempt_mean_handles_group_counts_segments_and_padding(self):
        for initial_groups, recovery_groups, weight in [(8, 2, 0.5), (9, 1, 0.5), (8, 2, 1.0), (8, 2, 0.0)]:
            with self.subTest(initial=initial_groups, recovery=recovery_groups, weight=weight):
                ids, rewards, weights, groups, masks, branches = [], [], [], [], [], []
                for g in range(initial_groups + recovery_groups):
                    recovery = g >= initial_groups
                    for c in range(16):
                        for segment in range(1 + c % 2):
                            ids.append(f"{g}:{c}")
                            rewards.append(float(c % 2 == 0))
                            weights.append(weight if recovery else 1.0)
                            groups.append(str(g))
                            masks.append([True, segment == 0])
                            branches.append(recovery)
                ids.append("padding"); rewards.append(0); weights.append(999); groups.append("padding")
                masks.append([False, False]); branches.append(False)
                batch = DataProto()
                mask = torch.tensor(masks)
                batch.batch = TensorDict({"response_mask": mask}, batch_size=len(ids))
                batch.non_tensor_batch = {
                    "attempt_ids": np.array(ids, dtype=object),
                    "attempt_rewards": np.array(rewards),
                    "attempt_loss_weights": np.array(weights),
                    "group_ids": np.array(groups, dtype=object),
                    "training_padding": np.array([False] * (len(ids) - 1) + [True]),
                }
                metrics = compute_official_attempt_grpo_advantage(batch, normalize_by_std=False)
                branch = torch.tensor(branches)
                contribution = batch.batch["advantages"].abs().sum(dim=1) / mask.sum()
                denominator = initial_groups + weight * recovery_groups
                self.assertAlmostEqual(float(contribution[~branch].sum()), 0.5 * initial_groups / denominator, places=6)
                self.assertAlmostEqual(float(contribution[branch].sum()), 0.5 * weight * recovery_groups / denominator, places=6)
                self.assertAlmostEqual(metrics["train/official_attempt_grpo_weight_sum"], 16 * denominator)

    def test_advantage_is_task_relative_and_equal_attempt_weighted(self):
        # Attempt A has two segments and three action tokens; attempt B has one of each.
        response_mask = torch.tensor(
            [[1, 1, 0], [1, 0, 0], [1, 0, 0]], dtype=torch.bool
        )
        batch = DataProto()
        batch.batch = TensorDict({"response_mask": response_mask}, batch_size=3)
        batch.non_tensor_batch = {
            "attempt_ids": np.array(["a", "a", "b"], dtype=object),
            "attempt_rewards": np.array([1.0, 1.0, 0.0], dtype=object),
            "group_ids": np.array(["task", "task", "task"], dtype=object),
        }

        compute_official_attempt_grpo_advantage(batch, normalize_by_std=False)

        advantages = batch.batch["advantages"]
        self.assertTrue(bool((advantages[0:2][response_mask[0:2]] > 0).all()))
        self.assertTrue(bool((advantages[2][response_mask[2]] < 0).all()))
        # Under token-mean aggregation, each complete attempt contributes equally.
        self.assertAlmostEqual(
            float(advantages[0:2].sum()),
            -float(advantages[2].sum()),
            places=6,
        )

    def test_net_generated_token_normalization_ignores_prompt_and_padding_rows(self):
        # Attempt A has three sampled tokens across two physical rows; attempt B has
        # one sampled token. Prompt lengths are absent from response_mask and a
        # data-parallel padding row has no sampled actions.
        response_mask = torch.tensor(
            [[1, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.bool
        )
        batch = DataProto()
        batch.batch = TensorDict({"response_mask": response_mask}, batch_size=4)
        batch.non_tensor_batch = {
            "attempt_ids": np.array(["a", "a", "b", "padding"], dtype=object),
            "attempt_rewards": np.array([1.0, 1.0, 0.0, 0.0], dtype=object),
            "group_ids": np.array(["task", "task", "task", "padding"], dtype=object),
            "training_padding": np.array([False, False, False, True], dtype=object),
        }

        metrics = compute_official_attempt_grpo_advantage(batch, normalize_by_std=False)

        advantages = batch.batch["advantages"]
        self.assertEqual(float(advantages[3].sum()), 0.0)
        self.assertAlmostEqual(float(advantages[0:2].sum()), 1.0, places=6)
        self.assertAlmostEqual(float(advantages[2].sum()), -1.0, places=6)
        self.assertEqual(metrics["train/official_attempt_grpo_attempts"], 2.0)
        self.assertEqual(metrics["train/official_attempt_grpo_padding_rows"], 1.0)

    def test_token_mean_policy_loss_equals_mean_of_per_attempt_net_generation_losses(self):
        response_mask = torch.tensor(
            [[1, 1, 0], [1, 0, 0], [1, 0, 0]], dtype=torch.bool
        )
        batch = DataProto()
        batch.batch = TensorDict({"response_mask": response_mask}, batch_size=3)
        batch.non_tensor_batch = {
            "attempt_ids": np.array(["a", "a", "b"], dtype=object),
            "attempt_rewards": np.array([1.0, 1.0, 0.0], dtype=object),
            "group_ids": np.array(["task", "task", "task"], dtype=object),
        }
        compute_official_attempt_grpo_advantage(batch, normalize_by_std=False)

        old_log_prob = torch.zeros_like(batch.batch["advantages"])
        log_ratio = torch.tensor(
            [[0.10, -0.10, 0.0], [0.05, 0.0, 0.0], [0.02, 0.0, 0.0]],
            dtype=torch.float32,
        )
        config = OmegaConf.create({
            "clip_ratio": 0.2,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.2,
            "clip_ratio_c": 3.0,
            "global_batch_info": {"batch_num_tokens": 4, "dp_size": 1},
        })

        actual, _metrics = compute_policy_loss_vanilla(
            old_log_prob=old_log_prob,
            log_prob=log_ratio,
            advantages=batch.batch["advantages"],
            response_mask=response_mask,
            loss_agg_mode="token-mean",
            config=config,
        )

        ratios_a = torch.exp(torch.tensor([0.10, -0.10, 0.05]))
        ratio_b = torch.exp(torch.tensor(0.02))
        expected = 0.5 * ((-0.5 * ratios_a).mean() + 0.5 * ratio_b)
        self.assertAlmostEqual(float(actual), float(expected), places=6)


class ReTermTrainerTest(unittest.TestCase):
    def test_checkpoint_retention_scans_checkpoints_from_previous_processes(self):
        with tempfile.TemporaryDirectory() as temporary_root:
            root = Path(temporary_root) / "checkpoints"
            for step in (50, 60, 70, 80):
                step_dir = root / f"global_step_{step}"
                (step_dir / "actor").mkdir(parents=True)
                (step_dir / "actor" / "shard.pt").write_bytes(b"checkpoint")
                (step_dir / ".node_sync_complete.json").write_text("{}\n")
            # An incomplete save must never be removed by retention cleanup.
            incomplete = root / "global_step_40"
            (incomplete / "actor").mkdir(parents=True)

            trainer = object.__new__(ReTermTrainer)
            trainer.config = OmegaConf.create({
                "trainer": {
                    "default_local_dir": str(root),
                    "max_actor_ckpt_to_keep": 2,
                }
            })
            trainer.global_steps = 80

            with mock.patch.object(
                ReTermTrainer,
                "_multi_node_checkpoint_sync_config",
                return_value=("host", "identity"),
            ):
                trainer._prune_checkpoint_history()

            self.assertFalse((root / "global_step_50").exists())
            self.assertFalse((root / "global_step_60").exists())
            self.assertTrue((root / "global_step_70").is_dir())
            self.assertTrue((root / "global_step_80").is_dir())
            self.assertTrue(incomplete.is_dir())

    def test_ephemeral_checkpoint_cleanup_removes_only_exact_actor_tree(self):
        with tempfile.TemporaryDirectory() as temporary_root:
            checkpoint_root = Path(temporary_root) / "custom-output" / "weights"
            step_dir = checkpoint_root / "global_step_10"
            actor_dir = step_dir / "actor"
            actor_dir.mkdir(parents=True)
            (actor_dir / "rank-shard.pt").write_bytes(b"checkpoint")

            result = _remove_ephemeral_checkpoint_tree(
                str(actor_dir), checkpoint_root=str(checkpoint_root)
            )

            self.assertTrue(result["existed"])
            self.assertTrue(result["removed"])
            self.assertTrue(result["empty_step_dir_removed"])
            self.assertFalse(actor_dir.exists())
            self.assertFalse(step_dir.exists())
            self.assertTrue(checkpoint_root.is_dir())

    def test_ephemeral_checkpoint_cleanup_rejects_broad_target(self):
        with tempfile.TemporaryDirectory() as temporary_root:
            checkpoint_root = Path(temporary_root) / "checkpoints"
            checkpoint_root.mkdir(parents=True)

            with self.assertRaises(ValueError):
                _remove_ephemeral_checkpoint_tree(
                    str(checkpoint_root), checkpoint_root=str(checkpoint_root)
                )
            unrelated = Path(temporary_root) / "elsewhere" / "global_step_10" / "actor"
            unrelated.mkdir(parents=True)
            with self.assertRaises(ValueError):
                _remove_ephemeral_checkpoint_tree(
                    str(unrelated), checkpoint_root=str(checkpoint_root)
                )
            self.assertTrue(unrelated.is_dir())

    def test_actor_update_uses_all_physical_rows_as_one_logical_minibatch(self):
        trainer = object.__new__(ReTermTrainer)
        trainer.config = OmegaConf.create({
            "algorithm": {"adv_estimator": "official_attempt_grpo"},
            "actor_rollout_ref": {
                "rollout": {"n": 1},
                "actor": {"ppo_mini_batch_size": 128},
            },
        })
        batch = DataProto()
        batch.batch = TensorDict({"x": torch.zeros(6, 1)}, batch_size=6)
        observed = {}

        def fake_parent_update(instance, value):
            observed["mini_batch_size"] = int(instance.config.actor_rollout_ref.actor.ppo_mini_batch_size)
            return value

        with mock.patch.object(RayPPOTrainer, "_update_actor", fake_parent_update):
            result = trainer._update_actor(batch)

        self.assertIs(result, batch)
        self.assertEqual(observed["mini_batch_size"], 6)
        self.assertEqual(int(trainer.config.actor_rollout_ref.actor.ppo_mini_batch_size), 128)


if __name__ == "__main__":
    unittest.main()
