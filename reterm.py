"""Validated entry point for ReTerm SFT and RL training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # RL's validated environment uses Python 3.10.
    import tomli as tomllib


ROOT = Path(__file__).resolve().parent
RL_CONFIG_DIR = ROOT / "rl" / "configs"
RL_TRAIN = ROOT / "rl" / "train.py"
SFT_DEEPSPEED = ROOT / "sft" / "configs" / "deepspeed_zero3_optimizer_offload.json"


class ProfileError(ValueError):
    """A training profile is incomplete or inconsistent."""


def _table(profile: dict[str, Any], key: str) -> dict[str, Any]:
    value = profile.get(key, {})
    if not isinstance(value, dict):
        raise ProfileError(f"[{key}] must be a TOML table")
    return value


def _required(table: dict[str, Any], key: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ProfileError(f"missing nonempty string: {key}")
    return value


def _path(table: dict[str, Any], key: str, *, must_exist: bool = True) -> Path:
    result = Path(os.path.expandvars(_required(table, key))).expanduser().resolve()
    if must_exist and not result.exists():
        raise ProfileError(f"{key} does not exist: {result}")
    return result


def _positive_int(table: dict[str, Any], key: str, *, minimum: int = 1) -> int:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProfileError(f"{key} must be an integer >= {minimum}")
    return value


def _python(profile: dict[str, Any]) -> Path:
    run = _table(profile, "run")
    # The interpreter in a virtual environment is commonly a symlink to the
    # system binary. Resolving it would silently discard that environment.
    python = Path(os.path.expandvars(_required(run, "python"))).expanduser().absolute()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ProfileError(f"python is not executable: {python}")
    return python


def _env(python: Path, *, rl: bool) -> dict[str, str]:
    env = os.environ.copy()
    paths = [ROOT / "src", ROOT / "rl" if rl else ROOT / "sft"]
    if rl:
        paths.extend(
            [
                ROOT / "rl" / "verl",
                ROOT / "rl" / "compat" / "py310_megatron_bridge",
                ROOT / "rl" / "external" / "Megatron-Bridge" / "src",
                ROOT / "rl" / "external" / "Megatron-Bridge" / "3rdparty" / "Megatron-LM",
            ]
        )
    # Do not inherit PYTHONPATH from another training checkout. Importing a
    # stale reterm_rl or veRL copy can silently change the algorithm.
    env["PYTHONPATH"] = os.pathsep.join(str(p) for p in paths)
    env["PATH"] = str(python.parent) + os.pathsep + env.get("PATH", "")
    return env


def _sft(profile: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    python = _python(profile)
    paths = _table(profile, "paths")
    training = _table(profile, "training")
    model = _path(paths, "model")
    if not (model / "config.json").is_file():
        raise ProfileError(f"model is missing config.json: {model}")
    data = _path(paths, "train_data")
    output = _path(paths, "output", must_exist=False)
    deepspeed = _path(paths, "deepspeed") if "deepspeed" in paths else SFT_DEEPSPEED
    if not deepspeed.is_file():
        raise ProfileError(f"DeepSpeed config does not exist: {deepspeed}")
    gpus = _positive_int(training, "gpus")
    epochs = training.get("epochs", 3)
    max_length = _positive_int(training, "max_length")
    if isinstance(epochs, bool) or not isinstance(epochs, (int, float)) or epochs <= 0:
        raise ProfileError("epochs must be positive")
    command = [
        str(python), "-m", "torch.distributed.run", "--standalone",
        f"--nproc_per_node={gpus}", "-m", "reterm_sft.train",
        "--model", str(model), "--data", str(data), "--output-dir", str(output),
        "--deepspeed", str(deepspeed), "--epochs", str(epochs),
        "--max-length", str(max_length), "--over-length-policy", "drop",
    ]
    if "validation_data" in paths:
        command.extend(["--validation-data", str(_path(paths, "validation_data"))])
    if "learning_rate" in training:
        command.extend(["--learning-rate", str(training["learning_rate"])])
    if "gradient_accumulation_steps" in training:
        command.extend(["--gradient-accumulation-steps", str(_positive_int(training, "gradient_accumulation_steps"))])
    return command, _env(python, rl=False)


def _hydra_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, list):
        return "[" + ",".join(_hydra_value(item) for item in value) + "]"
    if not isinstance(value, str):
        raise ProfileError("Hydra override values must be scalar values or lists")
    return json.dumps(value)


def _rl(profile: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    python = _python(profile)
    run = _table(profile, "run")
    groups = _table(profile, "groups")
    overrides = _table(profile, "hydra")
    config_name = _required(run, "config_name")
    if not (RL_CONFIG_DIR / f"{config_name}.yaml").is_file():
        raise ProfileError(f"unknown RL config: {config_name}")
    initial = _positive_int(groups, "initial")
    recovery = _positive_int(groups, "recovery")
    group_size = _positive_int(groups, "size")
    if group_size != 16:
        raise ProfileError("the current recovery sampler requires group size 16")
    total = initial + recovery
    required = {
        "actor_rollout_ref.model.path", "critic.model.path",
        "recovery_env.dataset.factory",
        "recovery_env.recovery_buffer_root", "trainer.default_local_dir",
    }
    missing = sorted(required - overrides.keys())
    if missing:
        raise ProfileError("missing required [hydra] overrides: " + ", ".join(missing))
    output_keys = {
        "trainer.default_local_dir",
        "recovery_env.recovery_buffer_root",
    }
    for key in required - output_keys - {"recovery_env.dataset.factory"}:
        value = overrides[key]
        if not isinstance(value, str):
            raise ProfileError(f"{key} does not exist: {value}")
        location = Path(os.path.expandvars(value)).expanduser()
        if not location.is_dir():
            raise ProfileError(f"{key} has the wrong type or does not exist: {value}")
    factory = overrides["recovery_env.dataset.factory"]
    if not isinstance(factory, str) or ":" not in factory:
        raise ProfileError("recovery_env.dataset.factory must be module:factory")
    model_paths = [Path(overrides[key]).expanduser().resolve() for key in (
        "actor_rollout_ref.model.path", "critic.model.path",
    )]
    if len(set(model_paths)) != 1 or not (model_paths[0] / "config.json").is_file():
        raise ProfileError("all RL model paths must point to one local model with config.json")
    fixed = {
        "recovery_env.recovery_buffer_initial_groups": initial,
        "recovery_env.recovery_buffer_recovery_groups": recovery,
        "es_manager.train.env_groups": total,
        "data.train_batch_size": total * group_size,
        "actor_rollout_ref.actor.ppo_mini_batch_size": total * group_size,
    }
    conflicting = sorted({key.lstrip("+") for key in fixed} & overrides.keys())
    if conflicting:
        raise ProfileError("[hydra] must not override derived group settings: " + ", ".join(conflicting))
    command = [str(python), str(RL_TRAIN), "--config-name", config_name]
    for key, value in {**overrides, **fixed}.items():
        command.append(f"{key}={_hydra_value(value)}")
    env = _env(python, rl=True)
    env["RETERM_RL_USE_MEGATRON"] = "1"
    for key, value in _table(profile, "environment").items():
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key) or not isinstance(value, (str, int, float, bool)):
            raise ProfileError(f"invalid [environment] setting: {key}")
        env[key] = str(value)
    if "ray_address" in run:
        env["RAY_ADDRESS"] = _required(run, "ray_address")
    return command, env


def _check_rl_composition(command: list[str], env: dict[str, str]) -> None:
    """Catch missing Hydra keys and example paths before GPU work begins."""
    check = r'''
import json
import sys
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

allowed = set(json.loads(sys.argv[3]))
with initialize_config_dir(config_dir=sys.argv[1], version_base=None):
    cfg = compose(config_name=sys.argv[2], overrides=sys.argv[4:])

stale = []
def walk(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            walk(item, f"{prefix}.{key}" if prefix else key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            walk(item, f"{prefix}[{index}]")
    elif isinstance(value, str) and prefix not in allowed and "/path/to/" in value:
        stale.append(prefix)

walk(OmegaConf.to_container(cfg, resolve=False))
if stale:
    raise SystemExit("unresolved example paths: " + ", ".join(stale))
'''
    probe = subprocess.run(
        [command[0], "-c", check, str(RL_CONFIG_DIR), command[3],
         json.dumps([item.split("=", 1)[0].lstrip("+") for item in command[4:]]),
         *command[4:]],
        cwd=ROOT, env=env, text=True, capture_output=True, check=False,
    )
    if probe.returncode:
        message = (probe.stderr or probe.stdout).strip().splitlines()
        raise ProfileError(message[-1] if message else "Hydra config composition failed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validated ReTerm training entry point")
    parser.add_argument("stage", choices=("sft", "rl"))
    parser.add_argument("--profile", required=True, type=Path, help="machine-local TOML profile")
    parser.add_argument("--run", action="store_true", help="execute after validation; default is dry-run")
    args = parser.parse_args(argv)
    try:
        with args.profile.open("rb") as source:
            profile = tomllib.load(source)
        builders = {"sft": _sft, "rl": _rl}
        command, env = builders[args.stage](profile)
        if args.stage == "rl":
            _check_rl_composition(command, env)
    except (OSError, ProfileError, tomllib.TOMLDecodeError) as exc:
        parser.exit(2, f"ReTerm preflight failed: {exc}\n")
    print(shlex.join(command))
    if not args.run:
        print("Dry run complete. Pass --run to start training.")
        return 0
    return subprocess.call(command, cwd=ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
