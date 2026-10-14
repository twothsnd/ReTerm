# Third-party training code

ReTerm-GRPO includes the locally modified training dependencies used by the reported training path. Reproduction uses these bundled versions and the source commits listed below.

| Component | Upstream | Source commit | Local changes |
| --- | --- | --- | --- |
| veRL | https://github.com/volcengine/verl | `6fef6a7a699435cad84e8907e9121457e41eed04` | Qwen3.5 agentic rollout, training, and loss integration. |
| Megatron-Bridge | https://github.com/NVIDIA-NeMo/Megatron-Bridge | `b864189731ae15c5f891f5d543f638d2213348cc` | Model conversion and training integration. |
| Megatron-LM (under Megatron-Bridge) | https://github.com/NVIDIA/Megatron-LM | `cf2f07d7b1315c96c05554c670c43207c6783e5e` | Attention integration. |

Original license notices remain in each bundled source tree. ReTerm-specific integration lives in `rl/reterm_rl/`, while changes to the underlying training engine remain beside the upstream files so the executable code path is visible.
