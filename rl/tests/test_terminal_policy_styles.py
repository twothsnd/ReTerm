from types import SimpleNamespace

from reterm_core.types import AgentContext, Task
from reterm_rl.agents.terminal_policy import (
    RolloutTimeoutError,
    TerminalPolicyAgent,
    TerminalPolicyAgentConfig,
)
from reterm_rl.agents.terminal_io import CommandResult


def _agent(**kwargs):
    return TerminalPolicyAgent(policy=lambda messages, context: "", config=TerminalPolicyAgentConfig(**kwargs))


def _attempt(metadata):
    return SimpleNamespace(agent_result=SimpleNamespace(metadata=metadata))


def test_qwen_recovery_continues_exact_native_history_without_sft_teacher_prompts():
    agent = _agent()
    previous_messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "original task"},
        {
            "role": "assistant",
            "content": "<think>inspect</think><tool_call>call</tool_call>",
            "token_ids": [10, 11],
            "log_probs": [-0.1, -0.2],
        },
        {"role": "user", "content": "<tool_response>\nfailed\n</tool_response>"},
    ]
    attempt = _attempt({"messages": previous_messages})
    recovery_prompt = "Your previous attempt failed.\nThe environment has NOT been reset.\n"

    messages = agent._initial_messages(
        Task(task_id="task", prompt="original task"),
        recovery_prompt,
        AgentContext(
            benchmark="test",
            task_id="task",
            attempt_index=2,
            previous_attempts=(attempt,),
        ),
    )

    assert messages[:-1] == previous_messages
    assert messages[:-1] is not previous_messages
    assert messages[-1] == {"role": "user", "content": recovery_prompt}
    rendered = "\n".join(str(message["content"]) for message in messages)
    assert "Recovery tool-call constraint" not in rendered
    assert "Inherited State Audit" not in rendered


def test_qwen_tool_observation_matches_sft_user_wrapped_transport():
    responses = iter(
        [
            "<think>inspect</think><tool_call>\n<function=bash>\n<parameter=command>\npwd\n"
            "</parameter>\n</function>\n</tool_call>"
        ]
    )
    agent = TerminalPolicyAgent(
        policy=lambda messages, context: next(responses),
        config=TerminalPolicyAgentConfig(max_turns_per_attempt=1),
    )
    environment = SimpleNamespace(exec=lambda command, timeout=None: CommandResult(True, "/tmp", 0))

    result = agent.run(
        Task(task_id="task", prompt="task"),
        "task",
        environment,
        AgentContext(benchmark="test", task_id="task", attempt_index=1),
    )

    observation = result.metadata["messages"][-1]
    assert observation["role"] == "user"
    assert observation["content"].startswith("<tool_response>\n")
    assert observation["content"].endswith("\n</tool_response>")


def test_qwen_parser_ignores_tool_shaped_text_inside_think():
    agent = _agent()
    response = (
        "<think>an old idea: <tool_call><function=bash><parameter=command>bad</parameter>"
        "</function></tool_call></think>"
        "<tool_call><function=bash><parameter=command>good</parameter></function></tool_call>"
    )

    action = agent._parse_bash_action(response)

    assert action.command == "good"


def test_agent_exposes_its_single_wall_deadline_to_policy_and_clears_it():
    deadlines = []

    def policy(messages, context):
        assert deadlines[-1] is not None
        return "<think>finished</think><tool_call><function=done></function></tool_call>"

    agent = TerminalPolicyAgent(
        policy=policy,
        config=TerminalPolicyAgentConfig(
            max_turns_per_attempt=1,
            max_wall_time_seconds=600,
        ),
        deadline_setter=deadlines.append,
    )
    environment = SimpleNamespace(
        exec=lambda command, timeout=None: CommandResult(True, "done", 0),
    )

    result = agent.run(
        Task(task_id="task", prompt="task"),
        "task",
        environment,
        AgentContext(benchmark="test", task_id="task", attempt_index=1),
    )

    assert len(result.actions) == 1
    assert deadlines[0] is not None
    assert deadlines[-1] is None


def test_agent_preserves_messages_when_policy_generation_times_out():
    def policy(messages, context):
        raise RolloutTimeoutError("")

    agent = TerminalPolicyAgent(
        policy=policy,
        config=TerminalPolicyAgentConfig(
            max_turns_per_attempt=2,
            max_wall_time_seconds=600,
        ),
    )
    environment = SimpleNamespace(exec=lambda command, timeout=None: CommandResult(True, "done", 0))

    result = agent.run(
        Task(task_id="task", prompt="task"),
        "task",
        environment,
        AgentContext(benchmark="test", task_id="task", attempt_index=1),
    )

    assert [message["role"] for message in result.metadata["messages"]] == ["system", "user"]
    assert result.metadata["stopped_by_wall_time"]
    assert result.metadata["turns"][0]["stopped_by_wall_time"]
