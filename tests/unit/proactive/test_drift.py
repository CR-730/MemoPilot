from __future__ import annotations

from collections.abc import Sequence

import pytest

from memopilot.extensions.skills import SkillCatalog, SkillDefinition
from memopilot.proactive.drift import DRIFT_SYSTEM_PROMPT, DriftSkillSelector
from memopilot.runtime.contracts import ChatMessage, FunctionCall, ModelResponse, ToolSchema


def _skill(name: str, *, background: bool = True, available: bool = True) -> SkillDefinition:
    return SkillDefinition(
        name=name,
        description=f"{name} 描述",
        background_allowed=background,
        required_tools=(),
        content=f"# {name}",
        source="test",
        path=None,
        available=available,
    )


class _Provider:
    def __init__(self, selected: str) -> None:
        self.selected = selected
        self.calls: list[tuple[Sequence[ChatMessage], Sequence[ToolSchema]]] = []

    async def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSchema],
    ) -> ModelResponse:
        self.calls.append((messages, tools))
        return ModelResponse(
            content=None,
            tool_calls=(
                FunctionCall("call-1", "select_drift_skill", {"skill_name": self.selected}),
            ),
        )


@pytest.mark.asyncio
async def test_selector_lets_llm_choose_from_background_candidates() -> None:
    provider = _Provider("beta")
    selector = DriftSkillSelector(
        provider,
        SkillCatalog(
            (
                _skill("alpha"),
                _skill("beta"),
                _skill("foreground", background=False),
                _skill("missing", available=False),
            )
        ),
    )

    selected = await selector.select()

    assert selected == "beta"
    prompt = provider.calls[0][0][-1].content or ""
    assert "alpha" in prompt and "beta" in prompt
    assert "foreground" not in prompt and "missing" not in prompt


@pytest.mark.asyncio
async def test_selector_rejects_name_outside_current_candidates() -> None:
    selector = DriftSkillSelector(_Provider("missing"), SkillCatalog((_skill("alpha"),)))

    with pytest.raises(ValueError, match="不可用"):
        await selector.select()


@pytest.mark.asyncio
async def test_selector_requires_exactly_one_selection_call() -> None:
    class _NoCallProvider(_Provider):
        async def complete(self, **kwargs) -> ModelResponse:
            return ModelResponse(content="alpha", tool_calls=())

    selector = DriftSkillSelector(_NoCallProvider("alpha"), SkillCatalog((_skill("alpha"),)))

    with pytest.raises(ValueError, match="select_drift_skill"):
        await selector.select()


def test_drift_prompt_keeps_prototype_finish_contract() -> None:
    assert "message_push 成功后禁止 recall_memory" in DRIFT_SYSTEM_PROMPT
    assert "必须调用 finish_drift" in DRIFT_SYSTEM_PROMPT
    assert "message_result" in DRIFT_SYSTEM_PROMPT
