from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path

import pytest

from memopilot.builtin_plugins.citation.plugin import (
    extract_cited_ids,
    extract_cited_ids_from_tool_chain,
)
from memopilot.extensions.events import EventBus
from memopilot.extensions.plugin_manager import PluginManager
from memopilot.memory.contracts import MemoryQueryResult, MemoryRecord
from memopilot.persistence.conversation import ConversationRepository
from memopilot.persistence.migrations import DatabaseKind, migrate_database
from memopilot.runtime.contracts import (
    ChatMessage,
    FunctionCall,
    ModelResponse,
    ToolSchema,
)
from memopilot.runtime.engine import DefaultReasoner, TurnInput
from memopilot.runtime.passive_turn import PassiveTurnPipeline
from memopilot.runtime.phases import LifecyclePhase
from memopilot.runtime.providers import ChatProvider
from memopilot.runtime.tools import Tool, ToolRegistry

PLUGIN_DIR = (
    Path(__file__).resolve().parents[3] / "src" / "memopilot" / "builtin_plugins" / "citation"
)


@pytest.mark.parametrize(
    ("response", "expected_reply", "expected_ids"),
    [
        ("正文\n§cited:[m1,m-2]§", "正文", ("m1", "m-2")),
        ("正文\n§cited:[]§", "正文", ()),
        (
            "正文\n§cited:[m1]§ <meme:shy> <foo:bar>",
            "正文 <meme:shy> <foo:bar>",
            ("m1",),
        ),
        (
            "正文提到 §cited:[m1]§，但不是协议行。\n后面还有内容",
            "正文提到 §cited:[m1]§，但不是协议行。\n后面还有内容",
            (),
        ),
        (
            "正文\n§cited:[m1]§ 其他文字",
            "正文\n§cited:[m1]§ 其他文字",
            (),
        ),
    ],
)
def test_citation_parser_keeps_prototype_boundary_semantics(
    response: str,
    expected_reply: str,
    expected_ids: tuple[str, ...],
) -> None:
    reply, cited_ids = extract_cited_ids(response)

    assert reply == expected_reply
    assert tuple(cited_ids) == expected_ids


def test_citation_extracts_ids_from_json_result_envelope() -> None:
    assert extract_cited_ids_from_tool_chain(
        [
            {
                "name": "recall_memory",
                "status": "success",
                "result": '{"ok":true,"status":"success","result":{"cited_item_ids":["m1"]}}',
            }
        ]
    ) == ["m1"]


class _NoopOutbound:
    async def dispatch(self, dispatch):
        del dispatch
        return True


async def run_through_passive_pipeline(
    reasoner: DefaultReasoner, turn: TurnInput, **kwargs: object
):
    with tempfile.TemporaryDirectory() as directory:
        database = Path(directory) / "operational.db"
        migrate_database(database, DatabaseKind.OPERATIONAL)
        pipeline = PassiveTurnPipeline(
            reasoner,
            repository=ConversationRepository(database),
            outbound=_NoopOutbound(),
            event_bus=reasoner._event_bus or EventBus(),
            history_limit=50,
            after_turn_modules=tuple(
                module
                for module in getattr(reasoner, "_test_outer_modules", ())
                if module.phase is LifecyclePhase.AFTER_TURN
            ),
            outer_modules=tuple(
                module
                for module in getattr(reasoner, "_test_outer_modules", ())
                if module.phase
                in {
                    LifecyclePhase.BEFORE_TURN,
                    LifecyclePhase.BEFORE_REASONING,
                    LifecyclePhase.AFTER_REASONING,
                }
            ),
        )
        return await pipeline.execute_direct(turn, **kwargs)


class _Provider(ChatProvider):
    def __init__(self, responses: Sequence[ModelResponse]) -> None:
        self.responses = list(responses)
        self.requests: list[tuple[ChatMessage, ...]] = []

    async def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSchema],
    ) -> ModelResponse:
        del tools
        self.requests.append(tuple(messages))
        return self.responses.pop(0)


class _MemoryEngine:
    async def query(self, request) -> MemoryQueryResult:
        del request
        return MemoryQueryResult(
            text_block="[p1] 用户偏好简洁回答",
            records=(MemoryRecord("p1", "preference", "用户偏好简洁回答", 0.9),),
            trace={"injected_ids": ["p1"]},
        )


async def test_citation_plugin_owns_prompt_cleanup_filter_and_recall_fallback() -> None:
    manager = PluginManager([PLUGIN_DIR], tool_registry=ToolRegistry())
    await manager.load_all()

    explicit_provider = _Provider(
        [
            ModelResponse(
                content="记得这件事 [§p1]\n§cited:[p1,forged]§ <meme:shy>",
                tool_calls=(),
                finish_reason="stop",
            )
        ]
    )
    explicit_runtime = DefaultReasoner(
        explicit_provider,
        ToolRegistry(),
        modules=manager.phase_modules,
        memory_engine=_MemoryEngine(),  # type: ignore[arg-type]
    )
    explicit_runtime._test_outer_modules = manager.phase_modules  # type: ignore[attr-defined]
    explicit = await run_through_passive_pipeline(
        explicit_runtime,
        TurnInput("feishu:chat-1", "我的偏好是什么？"),
    )

    assert "§cited:[id1,id2,id3]§" in (explicit_provider.requests[0][0].content or "")
    assert explicit.reply == "记得这件事"
    assert explicit.messages[-1].content == "记得这件事"
    assert explicit.cited_memory_ids == ("p1",)

    async def recall_memory(query: str) -> dict[str, object]:
        del query
        return {"cited_item_ids": ["m1"]}

    recall_tools = ToolRegistry(
        [
            Tool(
                "recall_memory",
                "recall",
                {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                recall_memory,
            )
        ]
    )
    fallback_provider = _Provider(
        [
            ModelResponse(
                content="",
                tool_calls=(FunctionCall("c1", "recall_memory", {"query": "偏好"}),),
                finish_reason="tool_calls",
            ),
            ModelResponse(
                content="回退正文 [§m1]",
                tool_calls=(),
                finish_reason="stop",
            ),
        ]
    )
    fallback_runtime = DefaultReasoner(
        fallback_provider,
        recall_tools,
        modules=manager.phase_modules,
    )
    fallback_runtime._test_outer_modules = manager.phase_modules  # type: ignore[attr-defined]
    fallback = await run_through_passive_pipeline(
        fallback_runtime,
        TurnInput("feishu:chat-1", "继续"),
    )

    assert fallback.reply == "回退正文"
    assert fallback.cited_memory_ids == ("m1",)

    raw_reply = "裸回复\n§cited:[p1]§"
    plain_provider = _Provider(
        [ModelResponse(content=raw_reply, tool_calls=(), finish_reason="stop")]
    )
    plain = await run_through_passive_pipeline(
        DefaultReasoner(
            plain_provider,
            ToolRegistry(),
            memory_engine=_MemoryEngine(),  # type: ignore[arg-type]
        ),
        TurnInput("feishu:chat-1", "不加载插件"),
    )

    assert "§cited:[" not in (plain_provider.requests[0][0].content or "")
    assert plain.reply == raw_reply
    assert plain.cited_memory_ids == ()
