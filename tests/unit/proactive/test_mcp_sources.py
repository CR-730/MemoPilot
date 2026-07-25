from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from memopilot.extensions.mcp import McpCallResult
from memopilot.persistence.migrations import DatabaseKind, migrate_database
from memopilot.proactive.mcp_sources import (
    ProactiveEvent,
    ProactiveFetchResult,
    ProactiveSourceGateway,
    load_proactive_sources,
)
from memopilot.proactive.store import ProactiveRepository

NOW = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)


class _Caller:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def call_tool(self, name: str, arguments: dict[str, object]) -> McpCallResult:
        self.calls.append((name, arguments))
        if self.fail:
            raise RuntimeError("offline")
        if name.startswith("fetch"):
            return McpCallResult(
                content=(
                    {
                        "type": "text",
                        "text": json.dumps(
                            [
                                {
                                    "event_id": f"{name}-1",
                                    "kind": "content",
                                    "published_at": NOW.isoformat(),
                                    "title": name,
                                }
                            ]
                        ),
                    },
                )
            )
        return McpCallResult(
            content=(
                {
                    "type": "text",
                    "text": json.dumps(
                        {"acknowledged": arguments.get("event_ids", []), "failed": []}
                    ),
                },
            )
        )


class _PrototypeFeedCaller:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def call_tool(self, name: str, arguments: dict[str, object]) -> McpCallResult:
        self.calls.append((name, arguments))
        if name == "get_proactive_events":
            payload = [
                {
                    "event_id": "fmcp_article_1",
                    "kind": "content",
                    "source_type": "rss",
                    "source_name": "示例订阅",
                    "title": "示例文章",
                    "content": "正文摘要",
                    "url": "https://example.com/article-1",
                    "published_at": NOW.isoformat(),
                    "display_text": "示例文章\n正文摘要",
                }
            ]
            return McpCallResult(
                content=({"type": "text", "text": json.dumps(payload, ensure_ascii=False)},)
            )
        return McpCallResult(
            content=(
                {
                    "type": "text",
                    "text": json.dumps(
                        {"acknowledged": arguments.get("event_ids", []), "failed": []},
                        ensure_ascii=False,
                    ),
                },
            )
        )


def _repository(tmp_path: Path) -> ProactiveRepository:
    database = tmp_path / "proactive.db"
    migrate_database(database, DatabaseKind.PROACTIVE)
    return ProactiveRepository(database)


def test_load_proactive_sources_from_workspace_json(tmp_path: Path) -> None:
    path = tmp_path / "proactive_sources.json"
    path.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "id": "news",
                        "server": "feeds",
                        "channel": "content",
                        "get_tool": "fetch_news",
                        "ack_tool": "ack_news",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    sources = load_proactive_sources(path)

    assert len(sources) == 1
    assert sources[0].source_id == "news"
    assert sources[0].server == "feeds"
    assert sources[0].channel == "content"
    assert sources[0].get_tool == "fetch_news"
    assert sources[0].ack_tool == "ack_news"


def test_load_prototype_sources_without_id_derives_stable_channel_id(
    tmp_path: Path,
) -> None:
    path = tmp_path / "proactive_sources.json"
    path.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "server": "mi-fitness",
                        "channel": "content",
                        "get_tool": "get_proactive_events",
                        "ack_tool": "acknowledge_events",
                    },
                    {
                        "server": "mi-fitness",
                        "channel": "alert",
                        "get_tool": "get_proactive_events",
                        "ack_tool": "acknowledge_events",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    sources = load_proactive_sources(path)

    assert [source.source_id for source in sources] == [
        "mi-fitness:content",
        "mi-fitness:alert",
    ]


async def test_gateway_reads_configs_directly_and_isolates_one_failed_source(
    tmp_path: Path,
) -> None:
    path = tmp_path / "proactive_sources.json"
    path.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "id": "good",
                        "server": "good-server",
                        "channel": "content",
                        "get_tool": "fetch_good",
                        "ack_tool": "ack_good",
                    },
                    {
                        "id": "bad",
                        "server": "bad-server",
                        "channel": "content",
                        "get_tool": "fetch_bad",
                        "ack_tool": "ack_bad",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    callers = {
        "good-server": _Caller(),
        "bad-server": _Caller(fail=True),
    }
    repository = _repository(tmp_path)
    gateway = ProactiveSourceGateway(
        repository,
        config_path=path,
        caller_for_server=callers.__getitem__,
    )

    report = await gateway.collect(session_key="feishu:chat-1", fetched_at=NOW)

    assert report.succeeded == ("good",)
    assert set(report.failed) == {"bad"}
    assert report.inserted_events == 1
    assert callers["good-server"].calls == [("fetch_good", {})]


async def test_gateway_raises_when_every_configured_source_fails(tmp_path: Path) -> None:
    path = tmp_path / "proactive_sources.json"
    path.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "id": "bad",
                        "server": "bad-server",
                        "channel": "content",
                        "get_tool": "fetch_bad",
                        "ack_tool": "ack_bad",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    gateway = ProactiveSourceGateway(
        _repository(tmp_path),
        config_path=path,
        caller_for_server=lambda _server: _Caller(fail=True),
    )

    with pytest.raises(RuntimeError, match="全部 Proactive Source 拉取失败"):
        await gateway.collect(session_key="feishu:chat-1", fetched_at=NOW)


async def test_gateway_accepts_prototype_feed_text_json_contract(tmp_path: Path) -> None:
    path = tmp_path / "proactive_sources.json"
    path.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "id": "feed",
                        "server": "feed-mcp",
                        "channel": "content",
                        "get_tool": "get_proactive_events",
                        "ack_tool": "acknowledge_events",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    caller = _PrototypeFeedCaller()
    repository = _repository(tmp_path)
    gateway = ProactiveSourceGateway(
        repository,
        config_path=path,
        caller_for_server=lambda _server: caller,
    )

    report = await gateway.collect(session_key="feishu:chat-1", fetched_at=NOW)

    assert report.inserted_events == 1
    stored = repository.list_unconsumed("feishu:chat-1")
    assert [(item.source_event_id, item.kind) for item in stored] == [("fmcp_article_1", "content")]
    assert stored[0].occurred_at == NOW.isoformat()
    assert stored[0].payload["title"] == "示例文章"
    assert caller.calls == [("get_proactive_events", {})]


async def test_gateway_replays_persisted_ack_ttl_to_mcp(tmp_path: Path) -> None:
    path = tmp_path / "proactive_sources.json"
    path.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "id": "news",
                        "server": "feeds",
                        "channel": "content",
                        "get_tool": "fetch_news",
                        "ack_tool": "ack_news",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    repository = _repository(tmp_path)
    repository.commit_fetch(
        session_key="feishu:chat-1",
        source_id="news",
        result=ProactiveFetchResult(
            (ProactiveEvent("news", "e1", "content", NOW.isoformat(), {"title": "A"}),)
        ),
        fetched_at=NOW,
    )
    decision = repository.create_decision(
        decision_id="discard-e1",
        session_key="feishu:chat-1",
        trigger_kind="content",
        action="skip",
        source_events=(("news", "e1"),),
        activity_version=1,
        decided_at=NOW,
        ack_ttl_hours={"news:e1": 720},
    )
    repository.commit_skip(decision.decision_id, committed_at=NOW)
    caller = _PrototypeFeedCaller()
    gateway = ProactiveSourceGateway(
        repository,
        config_path=path,
        caller_for_server=lambda _server: caller,
    )
    report = await gateway.replay_pending_acknowledgements(now=NOW)

    assert report.acknowledged == 1
    assert caller.calls == [
        (
            "ack_news",
            {
                "event_ids": ["e1"],
                "ttl_hours": 720,
            },
        )
    ]


async def test_alert_ack_does_not_expand_old_mcp_contract_with_content_ttl(
    tmp_path: Path,
) -> None:
    path = tmp_path / "proactive_sources.json"
    path.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "id": "alerts",
                        "server": "monitor",
                        "channel": "alert",
                        "get_tool": "fetch_alerts",
                        "ack_tool": "ack_alert",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    repository = _repository(tmp_path)
    repository.commit_fetch(
        session_key="feishu:chat-1",
        source_id="alerts",
        result=ProactiveFetchResult(
            (ProactiveEvent("alerts", "a1", "alert", NOW.isoformat(), {}),)
        ),
        fetched_at=NOW,
    )
    decision = repository.create_decision(
        decision_id="alert-a1",
        session_key="feishu:chat-1",
        trigger_kind="alert",
        action="alert",
        source_events=(("alerts", "a1"),),
        activity_version=1,
        decided_at=NOW,
    )
    repository.finalize_confirmed(
        decision.decision_id,
        is_effect_confirmed=lambda _operation_id: True,
        committed_at=NOW,
    )
    caller = _PrototypeFeedCaller()
    gateway = ProactiveSourceGateway(
        repository,
        config_path=path,
        caller_for_server=lambda _server: caller,
    )

    await gateway.replay_pending_acknowledgements(now=NOW)

    assert caller.calls == [("ack_alert", {"event_ids": ["a1"]})]


def test_source_config_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "proactive_sources.json"
    source = {
        "id": "news",
        "server": "feeds",
        "channel": "content",
        "get_tool": "fetch_news",
        "ack_tool": "ack_news",
    }
    path.write_text(json.dumps({"sources": [source, source]}), encoding="utf-8")

    with pytest.raises(ValueError, match="重复"):
        load_proactive_sources(path)
