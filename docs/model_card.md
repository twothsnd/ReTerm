---
library_name: transformers
license: apache-2.0
base_model: Qwen/Qwen3.5-9B
datasets:
- twothsnd/ReTerm-10K
pipeline_tag: image-text-to-text
tags:
- agents
- tool-use
- reinforcement-learning
- terminal-agent
---

# ReTerm-9B

This repository contains **ReTerm-9B**, a self-improving terminal agent built on Qwen3.5-9B through SFT and ReTerm-GRPO. The model uses the Qwen3.5 architecture and chat template.

[**Training code**](https://github.com/twothsnd/ReTerm) · [**ReTerm-10K dataset**](https://huggingface.co/datasets/twothsnd/ReTerm-10K) · [**Base model**](https://huggingface.co/Qwen/Qwen3.5-9B)

## 👀 About ReTerm

We introduce **ReTerm**, a self-improving terminal agent trained to recover from errors produced during its own execution.

We construct **ReTerm-10K**, a high-quality recovery trajectory dataset generated through a dedicated data pipeline for SFT cold start. Moreover, we propose **ReTerm-GRPO**, an online RL algorithm that branches into multiple independent self-recovery attempts from the agent's own failure states. Their verifier outcomes provide group-relative learning signals, enabling the agent to self-improve through its own recovery experience.

Built on Qwen3.5-9B, **ReTerm-9B** demonstrates strong performance on challenging terminal and software engineering tasks, with generalization across out-of-domain benchmarks and different agent harnesses.

## 🏆 Performance

ReTerm-9B achieves strong results on terminal and software engineering benchmarks:

| Benchmark | Pass@1 |
| --- | ---: |
| OpenThoughts-TBLite | **60.6%** |
| Terminal-Bench 2.0 | **34.8%** |
| SWE-bench Verified | **58.4%** |
| AIME 2024 | **86.7%** |
| AIME 2025 | **93.3%** |
| Aider Polyglot | **20.4%** |
| SpreadsheetBench Verified | **48.8%** |

Beyond these core evaluations, ReTerm also generalizes to out-of-domain tasks, including mathematical reasoning and spreadsheet processing. The improvement trend persists across different agent harnesses, suggesting transferable capabilities across interaction setups.

## 🎥 Recovery Example

<div align="center">
  <img src="assets/figure1-white.png" alt="Recovery behavior of GRPO and ReTerm-GRPO on a PyStan installation task" width="95%">
</div>

## 🔧 Using the model

ReTerm-9B can be loaded with Qwen3.5-compatible inference frameworks. For terminal-agent use, apply the Qwen3.5 chat template and tool-call protocol with an agent harness that executes commands and returns environment observations. Training code and configurations are available in the [ReTerm repository](https://github.com/twothsnd/ReTerm).
