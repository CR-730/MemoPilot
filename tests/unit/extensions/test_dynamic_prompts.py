from __future__ import annotations

from memopilot.extensions.prompts import (
    PromptBlock,
    PromptRenderer,
    PromptSectionRender,
)


def test_prompt_renderer_keeps_top_core_blocks_bottom_order_and_total_budget() -> None:
    renderer = PromptRenderer(
        (PromptBlock("plugin.block", "BLOCK", max_chars=5),)
    )

    rendered = renderer.render(
        scope="passive",
        base_prompt="CORE",
        max_chars=20,
        top_sections=(PromptSectionRender("top", "TOP", is_static=True),),
        bottom_sections=(PromptSectionRender("bottom", "BOTTOM", is_static=False),),
    )

    assert rendered == "TOP\n\nCORE\n\nBLOCK\n\nBO"
    assert len(rendered) == 20


def test_only_static_prompt_sections_are_cached_across_turns() -> None:
    renderer = PromptRenderer()

    first = renderer.prepare_sections(
        scope="passive",
        sections=(
            PromptSectionRender("policy", "固定协议", is_static=True),
            PromptSectionRender("user", "用户甲", is_static=False),
        ),
    )
    second = renderer.prepare_sections(
        scope="passive",
        sections=(
            PromptSectionRender("policy", "固定协议", is_static=True),
            PromptSectionRender("user", "用户乙", is_static=False),
        ),
    )

    assert [item.cache_hit for item in first] == [False, False]
    assert [item.cache_hit for item in second] == [True, False]
    assert second[1].content == "用户乙"
def test_static_prompt_cache_is_bounded() -> None:
    renderer = PromptRenderer(static_cache_max_entries=2)

    for index in range(3):
        renderer.prepare_sections(
            scope="passive",
            sections=(
                PromptSectionRender(
                    f"policy-{index}",
                    f"content-{index}",
                    is_static=True,
                ),
            ),
        )

    assert renderer.static_cache_size == 2


def test_top_sections_cannot_exhaust_core_prompt_budget() -> None:
    renderer = PromptRenderer()

    rendered = renderer.render(
        scope="passive",
        base_prompt="CORE-MUST-STAY",
        max_chars=40,
        top_sections=(
            PromptSectionRender("huge", "T" * 200, is_static=False),
        ),
    )

    assert len(rendered) <= 40
    assert "CORE-MUST-STAY" in rendered
    assert rendered.startswith("T")


def test_core_prompt_owns_the_entire_budget_when_it_fills_the_limit() -> None:
    renderer = PromptRenderer(
        (PromptBlock("plugin.block", "PLUGIN", max_chars=20),)
    )
    core = "C" * 40

    rendered = renderer.render(
        scope="passive",
        base_prompt=core,
        max_chars=40,
        top_sections=(
            PromptSectionRender("huge", "T" * 200, is_static=False),
        ),
        bottom_sections=(
            PromptSectionRender("bottom", "BOTTOM", is_static=False),
        ),
    )

    assert rendered == core
