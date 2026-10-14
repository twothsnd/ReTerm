"""CPU-only contract tests for the public-facing training entry point."""

from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reterm  # noqa: E402


def test_rl_imports_bundled_training_libraries() -> None:
    env = reterm._env(Path(sys.executable), rl=True)
    command = (
        "import json, verl, megatron.core, megatron.bridge; "
        "print(json.dumps([verl.__file__, megatron.core.__file__, megatron.bridge.__file__]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", command],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    paths = [Path(path).resolve() for path in json.loads(result.stdout.splitlines()[-1])]
    expected = [
        reterm.ROOT / "rl" / "verl",
        reterm.ROOT / "rl" / "external" / "Megatron-Bridge" / "3rdparty" / "Megatron-LM",
        reterm.ROOT / "rl" / "external" / "Megatron-Bridge" / "src",
    ]
    assert all(path.is_relative_to(root) for path, root in zip(paths, expected, strict=True))


def _files(tmp_path: Path) -> dict[str, str]:
    root = tmp_path / "assets"
    root.mkdir()
    model = root / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    train = root / "train.jsonl"
    train.touch()
    task_ids = root / "task-ids.txt"
    task_ids.touch()
    return {
        "model": str(model),
        "train": str(train),
        "task_ids": str(task_ids),
        "root": str(root),
    }


def test_sft_command_preserves_virtualenv_python(tmp_path: Path) -> None:
    assets = _files(tmp_path)
    profile = {
        "run": {"python": sys.executable},
        "paths": {
            "model": assets["model"],
            "train_data": assets["train"],
            "output": str(tmp_path / "output"),
        },
        "training": {"gpus": 2, "epochs": 1.5, "max_length": 65536},
    }
    command, env = reterm._sft(profile)
    assert command[0] == str(Path(sys.executable).absolute())
    assert "reterm_sft.train" in command
    assert command[command.index("--model") + 1] == assets["model"]
    assert str(reterm.ROOT / "sft") in env["PYTHONPATH"]


def test_rl_ab_composition_and_masked_paths(tmp_path: Path) -> None:
    assets = _files(tmp_path)
    directory_keys = {
        "actor_rollout_ref.model.path", "critic.model.path",
    }
    output_keys = {
        "recovery_env.recovery_buffer_root",
        "trainer.default_local_dir",
    }
    hydra = {key: assets["model"] for key in directory_keys}
    hydra.update({key: str(tmp_path / "output") for key in output_keys})
    hydra["recovery_env.dataset.factory"] = "example.tasks:create_dataset"
    hydra["recovery_env.dataset.state_root"] = str(tmp_path / "state")
    profile = {
        "run": {
            "python": sys.executable,
            "config_name": "reterm_grpo",
        },
        "groups": {"initial": 7, "recovery": 3, "size": 16},
        "hydra": hydra,
    }
    command, _ = reterm._rl(profile)
    assert "recovery_env.recovery_buffer_initial_groups=7" in command
    assert "recovery_env.recovery_buffer_recovery_groups=3" in command
    assert "data.train_batch_size=160" in command
    assert "es_manager.train.env_groups=10" in command
    pytest.importorskip("hydra")
    _, env = reterm._rl(profile)
    reterm._check_rl_composition(command, env)


def test_sft_cli_dry_run_never_creates_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assets = _files(tmp_path)
    profile = tmp_path / "sft.toml"
    output = tmp_path / "output"
    profile.write_text(
        f'[run]\npython = "{sys.executable}"\n'
        f'[paths]\nmodel = "{assets["model"]}"\n'
        f'train_data = "{assets["train"]}"\noutput = "{output}"\n'
        '[training]\ngpus = 1\nepochs = 1\nmax_length = 1024\n',
        encoding="utf-8",
    )
    assert reterm.main(["sft", "--profile", str(profile)]) == 0
    assert "Dry run complete" in capsys.readouterr().out
    assert not output.exists()


def test_rl_rejects_inconsistent_batch_override(tmp_path: Path) -> None:
    assets = _files(tmp_path)
    profile = {
        "run": {"python": sys.executable, "config_name": "reterm_grpo"},
        "groups": {"initial": 7, "recovery": 3, "size": 16},
        "hydra": {"data.train_batch_size": 128},
    }
    with pytest.raises(reterm.ProfileError):
        reterm._rl(profile)
