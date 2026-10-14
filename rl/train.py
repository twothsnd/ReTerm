"""Launch ReTerm-GRPO training with veRL workers."""

import os
from pathlib import Path

import hydra
import ray
from omegaconf import OmegaConf

from reterm_rl.bootstrap import REPO_ROOT, ensure_local_paths

ensure_local_paths()

from reterm_rl.trainer import ReTermTrainer  # noqa: E402


def register_resolvers() -> None:
    resolvers = {
        "mul": lambda x, y: x * y,
        "int_div": lambda x, y: int(float(x) / float(y)),
        "not": lambda x: not x,
    }
    for name, resolver in resolvers.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, resolver)


register_resolvers()


def normalize_node_local_paths(config):
    """Resolve paths before Ray sends the config to workers on other nodes.

    Persistent Ray daemons can have different working directories on each
    host. Checkpoint paths must therefore never retain cwd-relative semantics.
    Relative config paths are interpreted from the ReTerm workspace root.
    """

    checkpoint_dir = Path(str(config.trainer.default_local_dir)).expanduser()
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = (REPO_ROOT / checkpoint_dir).resolve()
    config.trainer.default_local_dir = str(checkpoint_dir)
    return config


def resolve_batch_sizes(config):
    if int(config.trainer.total_training_steps) == 0:
        config.data.train_batch_size = max(
            config.es_manager.val.env_groups * config.es_manager.val.group_size,
            config.ppo_mini_batch_size if config.ppo_mini_batch_size is not None else 16,
        )
    else:
        config.data.train_batch_size = config.es_manager.train.env_groups * config.es_manager.train.group_size

    if config.ppo_mini_batch_size is None:
        config.ppo_mini_batch_size = config.data.train_batch_size // 4
        print(f"config.ppo_mini_batch_size: {config.ppo_mini_batch_size}")

    if config.actor_rollout_ref.actor.ppo_mini_batch_size is None:
        config.actor_rollout_ref.actor.ppo_mini_batch_size = config.ppo_mini_batch_size
    if config.critic.ppo_mini_batch_size is None:
        config.critic.ppo_mini_batch_size = config.ppo_mini_batch_size

    if config.micro_batch_size_per_gpu is None:
        config.micro_batch_size_per_gpu = (
            config.actor_rollout_ref.actor.ppo_mini_batch_size // config.trainer.n_gpus_per_node
        )
        print(f"config.micro_batch_size_per_gpu: {config.micro_batch_size_per_gpu}")

    config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu = config.micro_batch_size_per_gpu
    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu = config.micro_batch_size_per_gpu
    config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu = config.micro_batch_size_per_gpu
    config.critic.ppo_micro_batch_size_per_gpu = config.micro_batch_size_per_gpu

    return config


@hydra.main(version_base=None, config_path="configs", config_name="reterm_grpo")
def main(config):
    config = resolve_batch_sizes(config)
    config = normalize_node_local_paths(config)
    run_ppo(config)


def run_ppo(config) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.system.CUDA_VISIBLE_DEVICES)
    print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")

    if not ray.is_initialized():
        ray_address = os.environ.get("RAY_ADDRESS")
        runtime_env_vars = {
            "TOKENIZERS_PARALLELISM": "true",
            "NCCL_DEBUG": os.environ.get("NCCL_DEBUG", "WARN"),
            "VLLM_LOGGING_LEVEL": "WARN",
        }
        for env_key in (
            'NCCL_SOCKET_IFNAME',
            'GLOO_SOCKET_IFNAME',
            'NCCL_IB_HCA',
            'NCCL_IB_DISABLE',
            'NCCL_IB_GID_INDEX',
            'NCCL_IB_TIMEOUT',
            'NCCL_IB_RETRY_CNT',
            'NCCL_CUMEM_ENABLE',
            'NCCL_P2P_DISABLE',
            'NCCL_SHM_DISABLE',
            'TORCH_NCCL_AVOID_RECORD_STREAMS',
            'TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC',
            'TORCH_NCCL_ENABLE_MONITORING',
            'XDG_CACHE_HOME',
            'TORCHINDUCTOR_CACHE_DIR',
            'TRITON_CACHE_DIR',
            'VLLM_CACHE_ROOT',
            'CUDA_CACHE_PATH',
            'HF_HOME',
            'HUGGINGFACE_HUB_CACHE',
            'TRANSFORMERS_CACHE',
            'VLLM_CONFIG_ROOT',
            'TORCH_HOME',
            'PYTORCH_ALLOC_CONF',
            'NCCL_SOCKET_NTHREADS',
            'NCCL_NSOCKS_PERTHREAD',
            'TRACKING_ROOT',
            'TENSORBOARD_DIR',
            # Keep weight transfer consistent across Ray workers.
            'VERL_WEIGHT_TRANSFER_USE_SHM',
            # Training libraries share the same import and linker paths.
            'PYTHONPATH',
            'LD_LIBRARY_PATH',
        ):
            env_value = os.environ.get(env_key)
            if env_value is not None:
                runtime_env_vars[env_key] = env_value
        # Actors attached to a persistent Ray cluster inherit the Ray node's
        # startup environment unless job-level runtime_env variables are set.
        # Forward rl controls so rollout concurrency and timeouts do
        # not silently depend on when the Ray head was launched.
        for env_key, env_value in os.environ.items():
            if env_key.startswith('RETERM_RL_'):
                runtime_env_vars[env_key] = env_value
        ray_init_kwargs = {
            "runtime_env": {
                "env_vars": runtime_env_vars,
            }
        }
        if ray_address:
            ray_init_kwargs["address"] = ray_address
        ray.init(**ray_init_kwargs)

    runner_options = {}
    if os.environ.get("RETERM_RL_PIN_TASKRUNNER_TO_DRIVER", "0") == "1":
        # Keep the coordinator on the driver's node when requested.
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        driver_node_id = ray.get_runtime_context().get_node_id()
        runner_options["scheduling_strategy"] = NodeAffinitySchedulingStrategy(
            node_id=driver_node_id,
            soft=False,
        )
        print(f"Pinning rl TaskRunner to driver node {driver_node_id}", flush=True)
    runner = TaskRunner.options(**runner_options).remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:

    def run(self, config):
        print(
            "rl runtime controls: "
            f"rollout_workers={os.environ.get('RETERM_RL_ROLLOUT_WORKERS', '<unset>')} "
            f"rollout_timeout={os.environ.get('RETERM_RL_ROLLOUT_WALL_TIME_LIMIT_SECONDS', '<unset>')} "
            f"llm_timeout={os.environ.get('RETERM_RL_LLM_TIMEOUT', '<unset>')}",
            flush=True,
        )
        from verl.utils.fs import copy_to_local
        is_eval_only = int(config.trainer.total_training_steps) == 0

        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        from verl.utils import hf_tokenizer, hf_processor
        tokenizer = hf_tokenizer(local_path)
        processor = hf_processor(local_path, use_fast=True)

        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager
        from verl.trainer.ppo.utils import Role, need_critic, need_reference_policy
        from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker
        from verl.utils.config import validate_config

        ray_worker_group_cls = RayWorkerGroup

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        ref_in_actor = bool(config.actor_rollout_ref.model.get("lora_adapter_path", None))
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        ref_in_actor = ref_in_actor or lora_rank > 0
        actor_role = Role.ActorRolloutRef if need_reference_policy(config) and not ref_in_actor else Role.ActorRollout

        role_worker_mapping = {actor_role: ray.remote(ActorRolloutRefWorker)}
        if is_eval_only:
            config.actor_rollout_ref.actor.use_kl_loss = False
        else:
            if need_critic(config):
                role_worker_mapping[Role.Critic] = ray.remote(TrainingWorker)
            if not need_reference_policy(config):
                config.actor_rollout_ref.actor.use_kl_loss = False
        
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }

        mapping = {actor_role: global_pool_id}
        if not is_eval_only and need_critic(config):
            mapping[Role.Critic] = global_pool_id

        if config.reward_model.enable:
            if config.reward_model.strategy == "fsdp":
                from verl.workers.engine_workers import TrainingWorker as RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        trainer = ReTermTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
        )

        if is_eval_only:
            trainer.init_eval_only_workers()
            trainer.evaluate()
        else:
            trainer.init_workers()
            trainer.init_agent_proxy()
            trainer.fit()


if __name__ == "__main__":
    main()
