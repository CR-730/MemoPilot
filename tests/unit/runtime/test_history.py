from memopilot.runtime.contracts import ChatMessage, FunctionCall
from memopilot.runtime.history import build_tool_chain, expand_history
from memopilot.tasks.operational import MessageRecord


def test_history_expands_assistant_tool_calls_results_and_final_reply() -> None:
    call = FunctionCall("call-1", "list_dir", {"path": "."})
    messages = (
        ChatMessage.assistant(content="我来查看", tool_calls=(call,)),
        ChatMessage.tool(call_id="call-1", name="list_dir", content="目录"),
        ChatMessage.assistant(content="已查看"),
    )
    records = (
        MessageRecord("u", "s", "user", "列出目录", "t", 1, "now"),
        MessageRecord("a", "s", "assistant", "已查看", "t", 2, "now", build_tool_chain(messages)),
    )

    history = expand_history(records)

    assert [(item.role, item.content) for item in history] == [
        ("user", "列出目录"),
        ("assistant", "我来查看"),
        ("tool", "目录"),
        ("assistant", "已查看"),
    ]
    assert history[1].tool_calls == (call,)
    assert history[2].tool_call_id == "call-1"


def test_history_truncates_large_tool_results_from_both_ends() -> None:
    result = "a" * 6_000 + "b" * 6_000
    records = (
        MessageRecord("u", "s", "user", "问题", "t", 0, "now"),
        MessageRecord(
            "a",
            "s",
            "assistant",
            "完成",
            "t",
            1,
            "now",
            (
                {
                    "calls": [
                        {"call_id": "call-1", "name": "tool", "arguments": {}, "result": result}
                    ]
                },
            ),
        ),
    )

    tool = expand_history(records)[2]

    assert len(tool.content or "") <= 10_050
    assert (tool.content or "").startswith("Total output lines: 1\n\n" + "a")
    assert (tool.content or "").endswith("b")
    assert "truncated" in (tool.content or "")


def test_history_skips_leading_assistant_until_user_boundary() -> None:
    records = (
        MessageRecord("a1", "s", "assistant", "旧半轮", "t1", 1, "now", ({"calls": []},)),
        MessageRecord("u", "s", "user", "新问题", "t2", 2, "now"),
        MessageRecord("a2", "s", "assistant", "新回答", "t2", 3, "now"),
    )

    history = expand_history(records)

    assert [(item.role, item.content) for item in history] == [
        ("user", "新问题"),
        ("assistant", "新回答"),
    ]


def test_tool_chain_does_not_pair_duplicate_call_ids_across_groups() -> None:
    call = FunctionCall("call-1", "tool", {})
    messages = (
        ChatMessage.assistant(content="第一组", tool_calls=(call,)),
        ChatMessage.tool(call_id="call-1", name="tool", content="结果一"),
        ChatMessage.assistant(content="第二组", tool_calls=(call,)),
        ChatMessage.tool(call_id="call-1", name="tool", content="结果二"),
    )

    groups = build_tool_chain(messages)

    assert [group["calls"][0]["result"] for group in groups] == ["结果一", "结果二"]
