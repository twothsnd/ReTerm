import os
import json
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import torch

from reterm_core.errors import TaskSkip
from reterm_rl.agents.terminal_policy import RolloutCancelledError
from reterm_rl.llm_agent.recovery_proxy import (
    ReTermAgentProxy,
    _DistributedSandboxEpisodeWorker,
)
from reterm_rl.task_adapter import TaskInfrastructureError
from verl.trainer.ppo.core_algos import compute_gae_advantage_return


class _FakeQwenTokenizer:
    pad_token_id = 0
    eos_token_id = 99
    all_special_ids = [99]

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=False, tokenize=False):
        if not tokenize:
            return "<|im_start|>user\nx<|im_end|>\n<|im_start|>assistant\n<think>\n"
        return [100, 101]

    def encode(self, text, add_special_tokens=False):
        return [200 + index for index, _char in enumerate(text)]

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "decoded"


class RecoveryProxyRefillTest(unittest.TestCase):
    @staticmethod
    def _task_stream_proxy() -> ReTermAgentProxy:
        proxy = object.__new__(ReTermAgentProxy)
        proxy.train_task_ids = ["new-a", "new-b"]
        proxy.val_task_ids = ["val-a"]
        proxy._task_sampling_seed = 42
        proxy._task_stream_lock = threading.Lock()
        proxy._task_streams = {
            False: {
                "epoch": 0,
                "cursor": 0,
                "order": ["new-a", "new-b"],
            },
            True: {
                "epoch": 0,
                "cursor": 0,
                "order": ["val-a"],
            },
        }
        return proxy

    def test_sampler_advances_to_next_task(self):
        proxy = self._task_stream_proxy()
        self.assertEqual(proxy._next_task_id(val=False), "new-a")
        self.assertEqual(proxy._next_task_id(val=False), "new-b")

    def test_task_cursor_is_checkpointed(self):
        proxy = self._task_stream_proxy()
        self.assertEqual(proxy._next_task_id(val=False), "new-a")
        state = proxy.state_dict()
        self.assertEqual(state["version"], 1)
        restored = self._task_stream_proxy()
        restored.load_state_dict(state)
        self.assertEqual(restored._next_task_id(val=False), "new-b")

    def test_distributed_ray_wait_is_cancelled_without_waiting_for_actor_cleanup(self):
        import ray

        object_ref = object()
        worker = SimpleNamespace(
            run_episode=SimpleNamespace(remote=mock.Mock(return_value=object_ref))
        )
        cancel_event = threading.Event()
        cancel_event.set()

        with (
            mock.patch("ray.get", side_effect=ray.exceptions.GetTimeoutError()),
            mock.patch("ray.cancel") as cancel,
        ):
            with self.assertRaises(RolloutCancelledError):
                ReTermAgentProxy._get_distributed_episode(
                    worker,
                    "rollout-a",
                    {"task_id": "task-a"},
                    cancel_event,
                )

        cancel.assert_not_called()

    def test_distributed_ray_wait_keeps_polling_before_cancellation(self):
        import ray

        object_ref = object()
        expected = {"reward": 1.0}
        worker = SimpleNamespace(
            run_episode=SimpleNamespace(remote=mock.Mock(return_value=object_ref))
        )
        with (
            mock.patch(
                "ray.get",
                side_effect=[ray.exceptions.GetTimeoutError(), expected],
            ) as get,
            mock.patch("ray.cancel") as cancel,
        ):
            actual = ReTermAgentProxy._get_distributed_episode(
                worker,
                "rollout-b",
                {"task_id": "task-b"},
                threading.Event(),
            )

        self.assertIs(actual, expected)
        self.assertEqual(get.call_count, 2)
        cancel.assert_not_called()

    def test_distributed_waiters_give_executor_a_bounded_shutdown(self):
        import ray

        cancel_event = threading.Event()
        refs = [object() for _ in range(8)]
        worker = SimpleNamespace(
            run_episode=SimpleNamespace(remote=mock.Mock(side_effect=refs))
        )

        def wait_until_cancelled(_ref, timeout):
            self.assertEqual(timeout, 0.25)
            raise ray.exceptions.GetTimeoutError()

        with (
            mock.patch("ray.get", side_effect=wait_until_cancelled),
            mock.patch("ray.cancel") as cancel,
        ):
            executor = ThreadPoolExecutor(max_workers=len(refs))
            futures = [
                executor.submit(
                    ReTermAgentProxy._get_distributed_episode,
                    worker,
                    "rollout-shutdown",
                    {"task_id": f"task-{index}"},
                    cancel_event,
                )
                for index in range(len(refs))
            ]
            time.sleep(0.05)
            started = time.monotonic()
            cancel_event.set()
            executor.shutdown(wait=True, cancel_futures=True)
            elapsed = time.monotonic() - started

        self.assertLess(elapsed, 1.0)
        cancel.assert_not_called()
        self.assertTrue(all(future.done() for future in futures))
        self.assertTrue(
            all(isinstance(future.exception(), RolloutCancelledError) for future in futures)
        )

    def test_worker_returns_normal_sentinel_for_expected_tail_cancellation(self):
        worker = object.__new__(_DistributedSandboxEpisodeWorker)
        worker.node_ip = "10.0.0.1"
        worker.node_index = 0
        worker._episode_slots = threading.BoundedSemaphore(1)
        worker._rollout_events = {}
        worker._rollout_active_calls = {}
        worker._rollout_finishing = set()
        worker._rollout_events_lock = threading.Lock()
        worker._wait_for_memory = mock.Mock(return_value=128.0)
        worker._proxy = SimpleNamespace(
            _run_one_episode=mock.Mock(
                side_effect=RolloutCancelledError("accepted batch already complete")
            )
        )

        result = worker.run_episode(rollout_id="rollout-tail", kwargs={"task_id": "task-a"})

        self.assertTrue(result["cancelled"])
        self.assertEqual(result["metrics"], {"rollout_cancelled_tail": 1.0})
        self.assertEqual(result["sandbox_node_ip"], "10.0.0.1")
        self.assertNotIn("rollout-tail", worker._rollout_active_calls)

    def test_worker_rollout_event_lives_until_last_active_call_exits(self):
        worker = object.__new__(_DistributedSandboxEpisodeWorker)
        worker._rollout_events = {}
        worker._rollout_active_calls = {}
        worker._rollout_finishing = set()
        worker._rollout_events_lock = threading.Lock()

        first = worker._begin_rollout_call("rollout-c")
        second = worker._begin_rollout_call("rollout-c")
        self.assertIs(first, second)
        self.assertEqual(worker._rollout_active_calls["rollout-c"], 2)

        worker.cancel_rollout("rollout-c")
        status = worker.finish_rollout("rollout-c")
        self.assertTrue(first.is_set())
        self.assertEqual(status, {"finished": False, "active_calls": 2})

        worker._end_rollout_call("rollout-c")
        self.assertIn("rollout-c", worker._rollout_events)
        worker._end_rollout_call("rollout-c")
        self.assertNotIn("rollout-c", worker._rollout_events)
        self.assertNotIn("rollout-c", worker._rollout_active_calls)
        self.assertNotIn("rollout-c", worker._rollout_finishing)

    def test_trajectory_writer_serializes_dataclass_directly_as_jsonl(self):
        @dataclass
        class Result:
            status: str
            values: list[int]

        proxy = object.__new__(ReTermAgentProxy)
        proxy._trajectory_lock = threading.Lock()
        with TemporaryDirectory() as tmp:
            proxy.trajectory_dir = Path(tmp)
            proxy._save_trajectory(
                dataset_name="task",
                episode={
                    "env_id": 1,
                    "group_id": 2,
                    "task_id": "任务",
                    "reward": 0.7,
                    "messages": [{"role": "assistant", "content": "完成"}],
                },
                benchmark_result=Result(status="ok", values=[1, 2, 3]),
            )
            raw = (Path(tmp) / "episodes.jsonl").read_bytes()
            self.assertEqual(raw.count(b"\n"), 1)
            payload = json.loads(raw)
            self.assertEqual(payload["task_id"], "任务")
            self.assertEqual(payload["benchmark_result"], {"status": "ok", "values": [1, 2, 3]})

    def test_sampled_im_end_keeps_its_logprob_and_is_trainable(self):
        proxy = object.__new__(ReTermAgentProxy)
        proxy.tokenizer = _FakeQwenTokenizer()
        proxy.config = SimpleNamespace(
            recovery_env=SimpleNamespace(agent=SimpleNamespace()),
            actor_rollout_ref=SimpleNamespace(actor=SimpleNamespace(policy_loss={})),
        )

        content, token_ids, log_probs = proxy._decoded_generation_with_log_probs(
            token_ids=[7, 99],
            log_probs=[-0.25, -0.5],
        )

        # Special tokens are stripped only from the text passed to the protocol
        # parser.  The sampled action and its rollout logprob stay untouched.
        self.assertEqual(content, "decoded")
        self.assertEqual(token_ids, [7, 99])
        self.assertEqual(log_probs, [-0.25, -0.5])

    def test_qwen_tool_schema_matches_sft(self):
        proxy = object.__new__(ReTermAgentProxy)
        proxy.config = SimpleNamespace(
            recovery_env=SimpleNamespace(agent=SimpleNamespace())
        )

        self.assertEqual(
            proxy._qwen_tool_schemas(),
            [
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "description": "Execute exactly one non-interactive shell command in the task environment.",
                        "parameters": {
                            "type": "object",
                            "properties": {"command": {"type": "string"}},
                            "required": ["command"],
                            "additionalProperties": False,
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "done",
                        "description": "Submit the current task state for official evaluation.",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "additionalProperties": False,
                        },
                    },
                },
            ],
        )

    def test_empty_protocol_result_fallback_contains_user_query(self):
        proxy = object.__new__(ReTermAgentProxy)

        messages, start_index = proxy._messages_and_current_attempt_start(SimpleNamespace(attempts=[]))

        self.assertEqual([message["role"] for message in messages], ["system", "user", "assistant"])
        self.assertEqual(start_index, 2)

    def test_refill_attempt_limit_is_bounded_by_batches(self):
        with mock.patch.dict(os.environ, {"RETERM_RL_MAX_REFILL_BATCHES_PER_TASK": "2"}):
            self.assertEqual(ReTermAgentProxy._max_refill_attempts_per_task(16), 32)

    def test_task_replacement_resets_group_examples(self):
        proxy = object.__new__(ReTermAgentProxy)
        proxy._next_task_id = lambda *, val: "task_new"

        task_by_group = {0: "task_old"}
        attempts_by_group = {0: 16}
        replacements_by_group = {0: 0}
        accepted = [{"env_id": 1}]
        group_examples = {0: accepted.copy()}
        group_candidates = {0: [{"env_id": 2}]}

        replaced = proxy._replace_group_task(
            group_id=0,
            task_by_group=task_by_group,
            attempts_by_group=attempts_by_group,
            replacements_by_group=replacements_by_group,
            group_examples=group_examples,
            group_candidates=group_candidates,
            max_replacement_tasks=3,
            val=False,
            reason="test",
        )

        self.assertTrue(replaced)
        self.assertEqual(task_by_group[0], "task_new")
        self.assertEqual(attempts_by_group[0], 0)
        self.assertEqual(replacements_by_group[0], 1)
        self.assertEqual(group_examples[0], [])
        self.assertEqual(group_candidates[0], [])

    def test_replacement_limit_returns_false_without_mutating_group(self):
        proxy = object.__new__(ReTermAgentProxy)
        proxy._cleanup_group_task_resources = mock.Mock()
        proxy._next_task_id = mock.Mock(return_value="task_new")
        task_by_group = {0: "task_old"}
        attempts_by_group = {0: 16}
        replacements_by_group = {0: 3}
        group_examples = {0: [{"env_id": 1}]}
        group_candidates = {0: []}

        replaced = proxy._replace_group_task(
            group_id=0,
            task_by_group=task_by_group,
            attempts_by_group=attempts_by_group,
            replacements_by_group=replacements_by_group,
            group_examples=group_examples,
            group_candidates=group_candidates,
            max_replacement_tasks=3,
            val=False,
            reason="limit",
        )

        self.assertFalse(replaced)
        self.assertEqual(task_by_group[0], "task_old")
        proxy._cleanup_group_task_resources.assert_not_called()

    def test_active_sampling_is_explicit_and_not_tied_to_advantage_estimator(self):
        proxy = object.__new__(ReTermAgentProxy)
        proxy.config = SimpleNamespace(
            recovery_env=SimpleNamespace(active_sampling={"enabled": True, "max_replacement_rounds": 4}),
            algorithm=SimpleNamespace(adv_estimator="gae"),
        )

        self.assertTrue(proxy._active_sampling_enabled())
        self.assertEqual(proxy._max_replacement_tasks(), 4)

    def test_validation_generation_seed_is_stable_per_task(self):
        first = ReTermAgentProxy._stable_generation_seed(
            rollout_seed=10_000,
            task_id="task_a",
            sample_index=0,
            val=True,
        )
        second = ReTermAgentProxy._stable_generation_seed(
            rollout_seed=10_000,
            task_id="task_a",
            sample_index=0,
            val=True,
        )
        other_sample = ReTermAgentProxy._stable_generation_seed(
            rollout_seed=10_000,
            task_id="task_a",
            sample_index=1,
            val=True,
        )

        self.assertEqual(first, second)
        self.assertNotEqual(first, other_sample)

    def test_train_and_validation_use_separate_task_manifests(self):
        proxy = object.__new__(ReTermAgentProxy)
        proxy.config = SimpleNamespace(
            recovery_env=SimpleNamespace(
                dataset={
                    "name": "task",
                    "factory": "example.tasks:create_dataset",
                    "tasks_root": "/tasks",
                    "task_ids_file": "/all.txt",
                    "train_task_ids_file": "/train.txt",
                    "val_task_ids_file": "/val.txt",
                }
            )
        )

        with mock.patch("reterm_rl.llm_agent.recovery_proxy.create_task_dataset") as dataset_cls:
            proxy._make_dataset(val=False)
            train_config = dataset_cls.call_args.args[0]
            proxy._make_dataset(val=True)
            val_config = dataset_cls.call_args.args[0]

        self.assertEqual(train_config["task_ids_file"], "/train.txt")
        self.assertEqual(val_config["task_ids_file"], "/val.txt")

    def test_infrastructure_failure_is_not_trainable(self):
        proxy = object.__new__(ReTermAgentProxy)
        proxy._official_segments_for_example = mock.Mock()

        is_trainable = proxy._is_trainable_example(
            {
                "task_id": "task_a",
                "env_id": 1,
                "discard_reason": "infrastructure_failure",
            }
        )

        self.assertFalse(is_trainable)
        proxy._official_segments_for_example.assert_not_called()

    def test_infrastructure_failure_categories_are_diagnostic(self):
        self.assertEqual(
            ReTermAgentProxy._infrastructure_failure_category(
                TaskInfrastructureError("task setup failed")
            ),
            "environment",
        )
        self.assertEqual(
            ReTermAgentProxy._infrastructure_failure_category(
                TaskInfrastructureError("evaluation unavailable", category="evaluation")
            ),
            "evaluation",
        )
        self.assertEqual(
            ReTermAgentProxy._infrastructure_failure_category(TaskSkip("task", "unavailable")),
            "environment",
        )
        self.assertIsNone(
            ReTermAgentProxy._infrastructure_failure_category(
                RuntimeError("unexpected internal state")
            )
        )
        self.assertIsNone(
            ReTermAgentProxy._infrastructure_failure_category(
                FileNotFoundError("unrecognized missing file")
            )
        )
        self.assertIsNone(
            ReTermAgentProxy._infrastructure_failure_category(
                RuntimeError("failed to initialize tokenizer")
            )
        )

    def test_infrastructure_metrics_include_zeros_and_aggregate_counts(self):
        normal = ReTermAgentProxy._infrastructure_metrics()
        environment_failure = ReTermAgentProxy._infrastructure_metrics("environment")

        self.assertEqual(normal["infrastructure_failures"], 0.0)
        self.assertEqual(environment_failure["infrastructure_failures"], 1.0)
        self.assertEqual(environment_failure["infrastructure_environment"], 1.0)
        self.assertEqual(environment_failure["infrastructure_evaluation"], 0.0)

        metrics = {
            f"task/{key}": [normal[key], environment_failure[key], normal[key]]
            for key in normal
        }
        aggregated = ReTermAgentProxy._aggregate_rollout_metrics(metrics, "task")

        self.assertAlmostEqual(aggregated["task/infrastructure_failures"], 1.0 / 3.0)
        self.assertAlmostEqual(aggregated["task/infrastructure_failure_rate"], 1.0 / 3.0)
        self.assertEqual(aggregated["task/infrastructure_failure_count"], 1.0)
        self.assertEqual(aggregated["task/rollout_episode_count"], 3.0)
        self.assertEqual(aggregated["task/infrastructure_environment_count"], 1.0)

    def test_steady_state_sampling_decouples_concurrency_from_train_batch(self):
        proxy = object.__new__(ReTermAgentProxy)
        # object.__new__ deliberately bypasses the production constructor;
        # mirror its no-distributed-sandbox default for this unit fixture.
        proxy._sandbox_workers = []
        proxy.config = SimpleNamespace(
            es_manager=SimpleNamespace(
                train=SimpleNamespace(env_groups=1, group_size=2, max_workers=4),
                val=SimpleNamespace(env_groups=1, group_size=1, max_workers=1),
            ),
            recovery_env=SimpleNamespace(
                active_sampling={
                    "enabled": True,
                    "steady_state": {
                        "enabled": True,
                        "prefetch_factor": 2.0,
                        "max_candidate_groups": 8,
                    },
                },
                dataset={"name": "task"},
            ),
        )
        proxy._task_sampling_seed = 42
        proxy._task_stream_lock = threading.Lock()
        proxy._task_streams = {
            False: {"epoch": 0, "cursor": 0, "order": []},
            True: {"epoch": 0, "cursor": 0, "order": []},
        }
        tasks = ["task_mixed", "task_all_fail", "task_all_success", "task_mixed_2"]
        task_cursor = {"value": 0}

        def next_task(*, val):
            del val
            index = task_cursor["value"]
            task_cursor["value"] += 1
            return tasks[index] if index < len(tasks) else f"task_mixed_{index}"

        proxy._next_task_id = next_task
        proxy._is_trainable_example = lambda example, *, allow_discard=False: True
        proxy._is_over_context_example = lambda example: False
        proxy._cleanup_group_task_resources = lambda task_id: None
        proxy.trajectory_dir = None
        proxy._resolve_buffer_group = lambda **_kwargs: None

        release = threading.Event()
        state_lock = threading.Lock()
        state = {"active": 0, "peak": 0, "started": 0}

        def run_episode(**kwargs):
            with state_lock:
                state["active"] += 1
                state["started"] += 1
                state["peak"] = max(state["peak"], state["active"])
            release.wait(timeout=5)
            with state_lock:
                state["active"] -= 1
            task_id = kwargs["task_id"]
            if task_id.startswith("task_mixed"):
                reward = float(kwargs["sample_index"] == 1)
            elif task_id == "task_all_success":
                reward = 1.0
            else:
                reward = 0.0
            return {
                "env_id": kwargs["env_id"],
                "group_id": kwargs["group_id"],
                "task_id": task_id,
                "reward": reward,
                "metrics": {},
            }

        captured = []
        proxy._run_one_episode = run_episode
        proxy._formulate_rollouts = lambda examples: (
            captured.extend(examples) or SimpleNamespace(meta_info={}),
            {},
        )

        result_holder = []
        error_holder = []

        def run_rollout():
            try:
                result_holder.append(proxy._rollout_steady_state(
                    mode_cfg=proxy.config.es_manager.train,
                    target_groups=1,
                    group_size=2,
                    rollout_seed=1,
                    val=False,
                    sandbox_rollout_id="test",
                ))
            except Exception as exc:  # pragma: no cover - surfaced below
                error_holder.append(exc)

        thread = threading.Thread(target=run_rollout)
        thread.start()
        for _ in range(100):
            with state_lock:
                if state["started"] >= 4:
                    break
            threading.Event().wait(0.01)
        release.set()
        thread.join(timeout=10)

        self.assertFalse(thread.is_alive())
        self.assertEqual(error_holder, [])
        self.assertEqual(len(result_holder), 1)
        self.assertEqual(state["peak"], 4)
        # The optimizer receives one two-attempt group even though four rollout
        # workers and an eight-attempt prefetch pool were active.
        self.assertEqual(len(captured), 2)
        self.assertEqual(sorted(example["reward"] for example in captured), [0.0, 1.0])
        self.assertEqual(result_holder[0].meta_info["metrics"]["rollout_target_train_attempts"], 2.0)
        self.assertEqual(result_holder[0].meta_info["metrics"]["rollout_sampling_workers"], 4.0)
        self.assertEqual(result_holder[0].meta_info["metrics"]["rollout_pipeline_groups"], 4.0)
        self.assertGreater(result_holder[0].meta_info["metrics"]["rollout_cancelled_candidate_groups"], 0.0)

    def test_complete_group_metrics_include_zero_std_and_trainable_groups(self):
        proxy = object.__new__(ReTermAgentProxy)
        proxy._sandbox_workers = []
        proxy.config = SimpleNamespace(
            es_manager=SimpleNamespace(
                train=SimpleNamespace(env_groups=1, group_size=2, max_workers=2),
                val=SimpleNamespace(env_groups=1, group_size=1, max_workers=1),
            ),
            recovery_env=SimpleNamespace(
                active_sampling={
                    "enabled": True,
                    "steady_state": {
                        "enabled": True,
                        "prefetch_factor": 1.0,
                        "max_candidate_groups": 4,
                    },
                },
                dataset={"name": "task"},
            ),
        )
        proxy._task_sampling_seed = 42
        tasks = iter(["task_all_fail", "task_mixed"])
        proxy._next_task_id = lambda *, val: next(tasks)
        proxy._is_trainable_example = lambda example, *, allow_discard=False: True
        proxy._is_over_context_example = lambda example: False
        proxy._cleanup_group_task_resources = lambda task_id: None
        proxy.trajectory_dir = None
        proxy._resolve_buffer_group = lambda **_kwargs: None

        def run_episode(**kwargs):
            reward = 0.0
            if kwargs["task_id"] == "task_mixed":
                reward = float(kwargs["sample_index"] == 1)
            return {
                "env_id": kwargs["env_id"],
                "group_id": kwargs["group_id"],
                "task_id": kwargs["task_id"],
                "reward": reward,
                "metrics": {},
            }

        proxy._run_one_episode = run_episode
        proxy._formulate_rollouts = lambda examples: (SimpleNamespace(meta_info={}), {})

        result = proxy._rollout_steady_state(
            mode_cfg=proxy.config.es_manager.train,
            target_groups=1,
            group_size=2,
            rollout_seed=1,
            val=False,
            sandbox_rollout_id="test",
        )
        metrics = result.meta_info["metrics"]
        self.assertEqual(metrics["task/complete_groups_total"], 2.0)
        self.assertEqual(metrics["task/complete_groups_trainable_total"], 1.0)
        self.assertEqual(metrics["task/complete_groups_zero_std_total"], 1.0)
        self.assertEqual(metrics["task/complete_groups_all_success_total"], 0.0)
        self.assertEqual(metrics["task/complete_groups_all_fail_total"], 1.0)
        self.assertEqual(metrics["task/complete_groups_success_rate"], 0.25)
        self.assertEqual(metrics["task/all_complete_groups_reward_mean"], 0.25)
        self.assertEqual(metrics["task/complete_groups_reward_mean"], 0.25)
        self.assertEqual(metrics["task/complete_groups_trainable_rate"], 0.5)
        self.assertEqual(metrics["task/complete_groups_all_fail_rate"], 0.5)

        tasks = iter(["task_all_fail", "task_mixed"])
        examples, raw_metrics = proxy._rollout_steady_state(
            mode_cfg=proxy.config.es_manager.train,
            target_groups=1,
            group_size=2,
            rollout_seed=2,
            val=False,
            sandbox_rollout_id="test-raw",
            return_examples=True,
        )
        self.assertEqual(len(examples), 2)
        self.assertEqual(raw_metrics["task/complete_groups_success_rate"], 0.25)
        self.assertIn("rollout_collect_seconds", raw_metrics)

    def test_gae_propagates_terminal_reward_across_observation_gaps(self):
        response_mask = torch.tensor(
            [
                [0, 1, 1, 0, 0, 1, 0, 0, 1, 0],
                [0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
            ],
            dtype=torch.bool,
        )
        rewards = torch.zeros((2, 10), dtype=torch.float32)
        rewards[0, 8] = 1.0
        rewards[1, 8] = 0.7
        values = torch.zeros_like(rewards)

        advantages, returns = compute_gae_advantage_return(
            token_level_rewards=rewards,
            values=values,
            response_mask=response_mask,
            gamma=1.0,
            lam=1.0,
        )

        self.assertTrue(torch.allclose(returns[0][response_mask[0]], torch.ones(4)))
        self.assertTrue(torch.allclose(returns[1][response_mask[1]], torch.full((4,), 0.7)))
        self.assertTrue(bool(torch.all(advantages[0][response_mask[0]] > 0)))
        self.assertTrue(bool(torch.all(advantages[1][response_mask[1]] < 0)))


if __name__ == "__main__":
    unittest.main()
