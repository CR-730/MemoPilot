from __future__ import annotations

import json

from memopilot.channels.feishu_cards import (
    ToolLiveLine,
    build_live_card,
    build_summary_card,
)


def test_live_card_exposes_thinking_tools_and_partial_reply() -> None:
    card = json.loads(
        build_live_card(
            "正在分析用户问题",
            [
                ToolLiveLine(
                    call_id="call-1",
                    tool_name="web_search",
                    intent="查找资料",
                    target='"MemoPilot"',
                )
            ],
            "目前找到了一部分信息",
        )
    )

    elements = card["body"]["elements"]
    rendered = "\n".join(element.get("content", "") for element in elements)
    assert card["schema"] == "2.0"
    assert card["config"] == {"update_multi": True}
    assert "思考过程" in rendered
    assert "正在分析用户问题" in rendered
    assert "web_search" in rendered
    assert "目前找到了一部分信息" in rendered


def test_summary_card_collapses_thinking_and_keeps_completed_tool_timeline() -> None:
    card = json.loads(
        build_summary_card(
            "已经完成分析",
            [
                ToolLiveLine(
                    call_id="call-1",
                    tool_name="web_search",
                    intent="查找资料",
                    target="",
                    status="done",
                )
            ],
        )
    )

    thinking = card["body"]["elements"][0]
    tools = card["body"]["elements"][1]["content"]
    assert thinking["tag"] == "collapsible_panel"
    assert thinking["expanded"] is False
    assert "已经完成分析" in thinking["elements"][0]["content"]
    assert "✅" in tools
    assert "Done · 1 tools" in tools
