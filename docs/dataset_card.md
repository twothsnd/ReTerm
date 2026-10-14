---
pretty_name: ReTerm-10K
license: odc-by
language:
- en
task_categories:
- text-generation
tags:
- agents
- tool-use
- terminal-agent
- supervised-fine-tuning
---

# ReTerm-10K

This dataset provides SFT training data for **ReTerm**, a self-improving terminal agent that learns to recover from errors to improve task completion, as presented in [ReTerm: Recovery Training for Self-Improving Terminal Agents](https://arxiv.org/pdf/2610.18484).

[**Paper**](https://arxiv.org/pdf/2610.18484) · [**Code**](https://github.com/twothsnd/ReTerm) · [**ReTerm-9B model**](https://huggingface.co/twothsnd/ReTerm-9B)

## About ReTerm-10K

**ReTerm-10K** is a high-quality terminal-task recovery trajectory dataset for SFT cold start. A dedicated data pipeline generates the demonstrations: a base model first executes terminal tasks, and a stronger teacher then continues from the environment states left by failed attempts, using the preceding interaction history to perform recovery.

<div align="center">
  <img src="assets/recovery_pipeline-white.png" alt="ReTerm-10K construction: collect failed states, generate teacher recoveries, and filter the continuations" width="80%">
</div>

Task verification, format, task-constraint, and efficiency filtering yield **9,712 recovery trajectories**. These demonstrations include inspecting failed execution states, repairing existing implementations, and verifying task outcomes, teaching the agent to build on useful progress and resolve the problems that prevented completion.

## Dataset Files

The data is provided in JSONL format. Each example contains the original task, the base model's failed interaction history, and the teacher's successful recovery trajectory.

| Field | Description |
| --- | --- |
| `id`, `task_id` | Trajectory and source-task identifiers |
| `messages` | Task instruction, failed interaction history, and teacher recovery trajectory in message order |
| `train_from_message_index` | Starting message index of the teacher recovery trajectory |
| `target_attempt_index` | Index of the recovery attempt |
| `trajectory_sha256` | Source-trajectory integrity hash |

During SFT, the original task and failed history provide context, while loss is applied only to assistant tokens generated during the teacher's recovery. Training code and example configurations are available in the [ReTerm repository](https://github.com/twothsnd/ReTerm).

## Acknowledgements and License

We thank [TMax-15K](https://huggingface.co/datasets/allenai/tmax-15k-open-instruct) for the task resources. The dataset is released under **ODC-BY**.
