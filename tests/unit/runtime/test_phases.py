from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from memopilot.runtime.contracts import ChatMessage
from memopilot.runtime.phases import (
    LifecyclePhase,
    PhaseContext,
    PhaseDefinitionError,
    PhasePipeline,
    estimate_messages_tokens,
)


@dataclass
class _Module:
    phase: LifecyclePhase
    slot: str
    requires: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()
    value: str | None = None

    async def run(self, context: PhaseContext) -> dict[str, Any]:
        if self.value is None:
            return {}
        return {name: self.value for name in self.produces}


async def test_pipeline_orders_modules_by_dependencies_not_registration_order() -> None:
    pipeline = PhasePipeline(
        [
            _Module(
                LifecyclePhase.BEFORE_TURN,
                "before_turn.render",
                requires=("turn.context",),
                produces=("turn.prompt",),
                value="ready",
            ),
            _Module(
                LifecyclePhase.BEFORE_TURN,
                "before_turn.context",
                requires=("turn.input",),
                produces=("turn.context",),
                value="context",
            ),
        ],
        initial_slots={"turn.input"},
    )

    context = await pipeline.run(PhaseContext(slots={"turn.input": "hello"}))

    assert pipeline.module_slots == (
        "before_turn.context",
        "before_turn.render",
    )
    assert context.slots["turn.prompt"] == "ready"
    assert context.trace == [
        (LifecyclePhase.BEFORE_TURN, "before_turn.context"),
        (LifecyclePhase.BEFORE_TURN, "before_turn.render"),
    ]


@pytest.mark.parametrize(
    ("modules", "kind", "slots"),
    [
        (
            [
                _Module(LifecyclePhase.BEFORE_TURN, "duplicate"),
                _Module(LifecyclePhase.BEFORE_TURN, "duplicate"),
            ],
            "duplicate_slot",
            ("duplicate",),
        ),
        (
            [
                _Module(
                    LifecyclePhase.BEFORE_TURN,
                    "consumer",
                    requires=("missing",),
                )
            ],
            "missing_dependency",
            ("consumer", "missing"),
        ),
        (
            [
                _Module(
                    LifecyclePhase.BEFORE_TURN,
                    "left",
                    requires=("right",),
                ),
                _Module(
                    LifecyclePhase.BEFORE_TURN,
                    "right",
                    requires=("left",),
                ),
            ],
            "dependency_cycle",
            ("left", "right"),
        ),
    ],
)
def test_pipeline_reports_invalid_phase_definitions(
    modules: list[_Module],
    kind: str,
    slots: tuple[str, ...],
) -> None:
    with pytest.raises(PhaseDefinitionError) as caught:
        PhasePipeline(modules)

    assert caught.value.kind == kind
    assert set(slots).issubset(set(caught.value.slots))


def test_pipeline_does_not_expose_later_runtime_slots_to_earlier_phases() -> None:
    module = _Module(
        LifecyclePhase.BEFORE_TURN,
        "before_turn.invalid",
        requires=("reasoning.result",),
    )

    with pytest.raises(PhaseDefinitionError) as caught:
        PhasePipeline(
            [module],
            provided_slots={
                LifecyclePhase.AFTER_REASONING: {"reasoning.result"},
            },
        )

    assert caught.value.kind == "missing_dependency"


@pytest.mark.parametrize("collision", ["turn.input", "other.module"])
def test_pipeline_rejects_outputs_that_shadow_external_or_module_slots(
    collision: str,
) -> None:
    modules = [
        _Module(
            LifecyclePhase.BEFORE_TURN,
            "producer",
            produces=(collision,),
            value="x",
        ),
        _Module(LifecyclePhase.BEFORE_TURN, "other.module"),
    ]

    with pytest.raises(PhaseDefinitionError) as caught:
        PhasePipeline(modules, initial_slots={"turn.input"})

    assert caught.value.kind == "slot_collision"


async def test_pipeline_rejects_module_that_does_not_return_declared_output() -> None:
    pipeline = PhasePipeline(
        [
            _Module(
                LifecyclePhase.PROMPT_RENDER,
                "prompt_render.empty",
                produces=("prompt.messages",),
            )
        ]
    )

    with pytest.raises(RuntimeError, match="prompt.messages"):
        await pipeline.run(PhaseContext())


def test_estimate_messages_tokens_uses_prototype_json_budget() -> None:
    messages = (
        ChatMessage.system("rule"),
        ChatMessage.user("hi"),
    )

    assert estimate_messages_tokens(messages) == 24


def test_estimate_messages_tokens_includes_provider_fields_used_by_react() -> None:
    message = ChatMessage.assistant(
        content="answer",
        provider_fields={"reasoning_content": "trace"},
    )

    assert estimate_messages_tokens((message,)) == 24
