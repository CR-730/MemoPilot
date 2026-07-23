from __future__ import annotations

from datetime import UTC, datetime

import pytest

from memopilot.memory.contracts import MemoryQueryResult
from memopilot.proactive.content_turn import (
    CONTENT_TOOL_SCHEMAS,
    AgentTick,
    AgentTickDeps,
    AgentTickInput,
    ContentTurn,
    ContentTurnDeps,
    build_ack_instructions,
)
from memopilot.proactive.investigation import ContentCandidate
from memopilot.runtime.contracts import FunctionCall, ModelResponse

NOW = datetime(2026, 7, 22, tzinfo=UTC)


class _Provider:
    def __init__(self, calls: list[FunctionCall]) -> None:
        self.calls = list(calls)
        self.messages = []
        self.tool_names: list[tuple[str, ...]] = []

    async def complete(self, *, messages, tools=()):
        self.messages.append(tuple(messages))
        self.tool_names.append(tuple(str(tool["function"]["name"]) for tool in tools))
        call = self.calls.pop(0)
        return ModelResponse(None, (call,), finish_reason="tool_calls")


class _Memory:
    def __init__(self) -> None:
        self.requests = []

    async def query(self, request):
        self.requests.append(request)
        return MemoryQueryResult(text_block="用户关注 Agent", records=())


class _Fetcher:
    def __init__(self) -> None:
        self.urls: list[str] = []

    async def fetch(self, url: str) -> str:
        self.urls.append(url)
        return f"正文:{url}"


def _call(index: int, name: str, arguments: dict) -> FunctionCall:
    return FunctionCall(f"call-{index}", name, arguments)


def _candidate(name: str) -> ContentCandidate:
    return ContentCandidate(
        item_id=f"feed:{name}",
        title=f"标题 {name}",
        source_id="feed",
        published_at=NOW.isoformat(),
        payload={"url": f"https://example.com/{name}"},
    )


def test_content_tools_match_old_prototype_contract() -> None:
    names = tuple(tool["function"]["name"] for tool in CONTENT_TOOL_SCHEMAS)

    assert names == (
        "get_alert_events",
        "get_content_events",
        "get_context_data",
        "recall_memory",
        "get_content",
        "web_fetch",
        "web_search",
        "get_recent_chat",
        "message_push",
        "mark_interesting",
        "mark_not_interesting",
        "finish_turn",
    )


@pytest.mark.asyncio
async def test_agent_tick_batches_every_alert_and_skips_content_classification() -> None:
    provider = _Provider(
        [
            _call(1, "get_alert_events", {}),
            _call(
                2,
                "message_push",
                {"message": "两条提醒", "evidence": ["alerts:a1", "alerts:a2"]},
            ),
            _call(3, "finish_turn", {"decision": "reply"}),
        ]
    )
    tick = AgentTick(provider, AgentTickDeps(), max_steps=6)

    result = await tick.run(
        AgentTickInput(
            alerts=(
                {"source_id": "alerts", "event_id": "a1", "title": "提醒一"},
                {"source_id": "alerts", "event_id": "a2", "title": "提醒二"},
            ),
            contents=(_candidate("content"),),
            contexts=(),
            context_as_fallback_open=False,
        ),
        session_key="feishu:u1",
        now=NOW,
    )

    assert result.action == "reply"
    assert result.cited_item_ids == ("alerts:a1", "alerts:a2")
    assert result.interesting_item_ids == frozenset()
    assert result.discarded_item_ids == frozenset()


def test_agent_tick_renders_raw_context_as_background_and_derives_awake_probability() -> None:
    messages = AgentTick._initial_messages(
        AgentTickInput(
            contexts=(
                {
                    "_source": "fitbit",
                    "available": True,
                    "sleep_prob": 0.8,
                    "custom_metric": "kept",
                },
            ),
            context_as_fallback_open=True,
        ),
        memory_text="",
        proactive_context="",
        recent_context="",
    )

    rendered = "\n".join(message.content or "" for message in messages)
    assert "Alert > Content > Context-fallback" in rendered
    assert '"_source": "fitbit"' in rendered
    assert '"available": true' in rendered
    assert '"awake_prob": 0.2' in rendered
    assert '"custom_metric": "kept"' in rendered


@pytest.mark.asyncio
async def test_content_turn_prefetches_bodies_then_runs_old_react_tool_chain() -> None:
    provider = _Provider(
        [
            _call(1, "recall_memory", {"query": "如果标题 a 对用户有价值"}),
            _call(2, "mark_interesting", {"item_ids": ["feed:a"]}),
            _call(3, "get_content", {"item_ids": ["feed:a"]}),
            _call(4, "mark_not_interesting", {"item_ids": ["feed:b"]}),
            _call(5, "get_recent_chat", {}),
            _call(
                6,
                "message_push",
                {"message": "值得看", "evidence": ["feed:a"]},
            ),
            _call(7, "finish_turn", {"decision": "reply"}),
        ]
    )
    memory = _Memory()
    fetcher = _Fetcher()
    turn = ContentTurn(
        provider,
        ContentTurnDeps(
            memory=memory,
            content_fetcher=fetcher,
            recent_chat=lambda _session, _n: [{"role": "user", "content": "最近在看 Agent"}],
        ),
        max_steps=12,
    )

    result = await turn.run(
        (_candidate("a"), _candidate("b")),
        session_key="feishu:u1",
        now=NOW,
        memory_text="长期记忆",
        proactive_context="主动规则",
    )

    assert fetcher.urls == ["https://example.com/a", "https://example.com/b"]
    assert memory.requests[0].intent == "interest"
    assert memory.requests[0].limit == 2
    assert result.action == "reply"
    assert result.message == "值得看"
    assert result.cited_item_ids == ("feed:a",)
    assert result.interesting_item_ids == frozenset({"feed:a"})
    assert result.discarded_item_ids == frozenset({"feed:b"})
    assert provider.messages[0][-1].content is not None
    assert "正文:https://example.com/a" not in provider.messages[0][-1].content
    assert provider.messages[3][-1].role == "tool"
    assert "正文:https://example.com/a" in (provider.messages[3][-1].content or "")


@pytest.mark.asyncio
async def test_content_turn_reopens_skip_until_every_item_is_classified() -> None:
    provider = _Provider(
        [
            _call(1, "mark_not_interesting", {"item_ids": ["feed:a"]}),
            _call(2, "finish_turn", {"decision": "skip", "reason": "no_content"}),
            _call(3, "mark_not_interesting", {"item_ids": ["feed:b"]}),
            _call(4, "finish_turn", {"decision": "skip", "reason": "no_content"}),
        ]
    )
    turn = ContentTurn(provider, ContentTurnDeps(), max_steps=10)

    result = await turn.run(
        (_candidate("a"), _candidate("b")),
        session_key="feishu:u1",
        now=NOW,
    )

    assert result.action == "skip"
    assert result.discarded_item_ids == frozenset({"feed:a", "feed:b"})
    assert any(
        "尚未完成分类" in (message.content or "")
        for message in provider.messages[2]
        if message.role == "user"
    )


@pytest.mark.asyncio
async def test_tool_error_is_returned_to_llm_and_loop_can_recover() -> None:
    provider = _Provider(
        [
            _call(1, "finish_turn", {"decision": "reply"}),
            _call(2, "mark_not_interesting", {"item_ids": ["feed:a"]}),
            _call(3, "finish_turn", {"decision": "skip", "reason": "no_content"}),
        ]
    )
    turn = ContentTurn(provider, ContentTurnDeps(), max_steps=8)

    result = await turn.run((_candidate("a"),), session_key="feishu:u1", now=NOW)

    assert result.action == "skip"
    assert provider.messages[1][-1].role == "tool"
    assert "requires prior message_push" in (provider.messages[1][-1].content or "")


@pytest.mark.asyncio
async def test_context_fallback_cannot_reply_when_wake_policy_is_closed() -> None:
    provider = _Provider(
        [
            _call(1, "mark_not_interesting", {"item_ids": ["feed:a"]}),
            _call(2, "message_push", {"message": "仅根据 Context 发起聊天", "evidence": []}),
            _call(3, "finish_turn", {"decision": "reply"}),
            _call(4, "finish_turn", {"decision": "skip", "reason": "no_content"}),
        ]
    )
    tick = AgentTick(provider, AgentTickDeps(), max_steps=8)

    result = await tick.run(
        AgentTickInput(
            contents=(_candidate("a"),),
            contexts=({"_source": "presence", "available": True},),
            context_as_fallback_open=False,
        ),
        session_key="feishu:u1",
        now=NOW,
    )

    assert result.action == "skip"
    assert "未放行 Context fallback" in (provider.messages[2][-1].content or "")


def test_old_prototype_ack_ttls_are_preserved() -> None:
    instructions = build_ack_instructions(
        fetched_item_ids={"feed:a", "feed:b", "feed:c"},
        cited_item_ids={"feed:a"},
        interesting_item_ids={"feed:a", "feed:b"},
        discarded_item_ids={"feed:c"},
        delivery_succeeded=True,
    )

    assert {(item.item_id, item.ttl_hours) for item in instructions} == {
        ("feed:a", 168),
        ("feed:b", 24),
        ("feed:c", 720),
    }

    failed = build_ack_instructions(
        fetched_item_ids={"feed:a", "feed:b", "feed:c"},
        cited_item_ids={"feed:a"},
        interesting_item_ids={"feed:a", "feed:b"},
        discarded_item_ids={"feed:c"},
        delivery_succeeded=False,
    )
    assert [(item.item_id, item.ttl_hours) for item in failed] == [("feed:c", 720)]
