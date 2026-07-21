"""外层 Agent 生命周期与 PhaseModule 依赖编排。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol


class LifecyclePhase(StrEnum):
    """一次 Turn 的固定生命周期顺序。"""

    BEFORE_TURN = "before_turn"
    BEFORE_REASONING = "before_reasoning"
    PROMPT_RENDER = "prompt_render"
    BEFORE_STEP = "before_step"
    AFTER_STEP = "after_step"
    AFTER_REASONING = "after_reasoning"
    AFTER_TURN = "after_turn"


@dataclass
class PhaseContext:
    """PhaseModule 之间显式传递的产物和执行轨迹。"""

    slots: dict[str, Any] = field(default_factory=dict)
    trace: list[tuple[LifecyclePhase, str]] = field(default_factory=list)


@dataclass
class PluginPhaseFrame:
    """原型式插件 Phase 的独立可变帧。"""

    input: Any
    slots: dict[str, Any] = field(default_factory=dict)
    output: Any = None


class PhaseModule(Protocol):
    phase: LifecyclePhase
    slot: str
    requires: tuple[str, ...]
    produces: tuple[str, ...]

    async def run(self, context: PhaseContext) -> Mapping[str, Any]: ...


class PhaseDefinitionError(ValueError):
    """Phase 图在启动编译时即可定位的定义错误。"""

    def __init__(self, kind: str, slots: Sequence[str]) -> None:
        self.kind = kind
        self.slots = tuple(slots)
        joined = ", ".join(self.slots)
        super().__init__(f"Phase 定义错误 [{kind}]: {joined}")


class PhasePipeline:
    """按固定生命周期和模块依赖编译、执行 PhaseModule。"""

    def __init__(
        self,
        modules: Sequence[PhaseModule],
        *,
        initial_slots: set[str] | frozenset[str] = frozenset(),
        provided_slots: Mapping[
            LifecyclePhase,
            set[str] | frozenset[str],
        ]
        | None = None,
    ) -> None:
        self._initial_slots = frozenset(initial_slots)
        self._provided_slots = {
            phase: frozenset(slots)
            for phase, slots in (provided_slots or {}).items()
        }
        self._modules = self._compile(modules)

    @property
    def module_slots(self) -> tuple[str, ...]:
        return tuple(module.slot for module in self._modules)

    async def run(self, context: PhaseContext) -> PhaseContext:
        for phase in LifecyclePhase:
            await self.run_phase(phase, context)
        return context

    async def run_phase(
        self,
        phase: LifecyclePhase,
        context: PhaseContext,
    ) -> PhaseContext:
        for module in self._modules:
            if module.phase is not phase:
                continue
            missing_inputs = set(module.requires).difference(context.slots)
            if missing_inputs:
                raise RuntimeError(
                    f"PhaseModule {module.slot} 运行时缺少 slot: "
                    + ", ".join(sorted(missing_inputs))
                )
            outputs = dict(await module.run(context))
            missing_outputs = set(module.produces).difference(outputs)
            if missing_outputs:
                raise RuntimeError(
                    f"PhaseModule {module.slot} 未产生声明的 slot: "
                    + ", ".join(sorted(missing_outputs))
                )
            context.slots.update(outputs)
            context.slots[module.slot] = True
            context.trace.append((module.phase, module.slot))
        return context

    def _compile(self, modules: Sequence[PhaseModule]) -> tuple[PhaseModule, ...]:
        by_slot: dict[str, PhaseModule] = {}
        registration_order: dict[str, int] = {}
        produced_by: dict[tuple[LifecyclePhase, str], str] = {}
        external_slots = set(self._initial_slots)
        for slots in self._provided_slots.values():
            external_slots.update(slots)
        for index, module in enumerate(modules):
            if not module.slot or module.slot in by_slot:
                raise PhaseDefinitionError("duplicate_slot", (module.slot,))
            if module.slot in external_slots:
                raise PhaseDefinitionError("slot_collision", (module.slot,))
            by_slot[module.slot] = module
            registration_order[module.slot] = index
        module_slots = set(by_slot)
        for module in modules:
            for output in module.produces:
                if output in external_slots or output in module_slots:
                    raise PhaseDefinitionError(
                        "slot_collision",
                        (output, module.slot),
                    )
                output_key = (module.phase, output)
                previous = produced_by.get(output_key)
                if previous is not None:
                    raise PhaseDefinitionError(
                        "duplicate_output",
                        (output, previous, module.slot),
                    )
                produced_by[output_key] = module.slot

        compiled: list[PhaseModule] = []
        available = set(self._initial_slots)
        for phase in LifecyclePhase:
            available.update(self._provided_slots.get(phase, ()))
            phase_modules = [module for module in modules if module.phase is phase]
            if not phase_modules:
                continue
            same_phase_producers = {
                produced: module.slot
                for module in phase_modules
                for produced in (module.slot, *module.produces)
            }
            indegree = {module.slot: 0 for module in phase_modules}
            dependents: dict[str, list[str]] = {
                module.slot: [] for module in phase_modules
            }
            for module in phase_modules:
                for requirement in module.requires:
                    if requirement in available:
                        continue
                    producer = same_phase_producers.get(requirement)
                    if producer is None:
                        raise PhaseDefinitionError(
                            "missing_dependency",
                            (module.slot, requirement),
                        )
                    indegree[module.slot] += 1
                    dependents[producer].append(module.slot)

            ready = [slot for slot, degree in indegree.items() if degree == 0]
            phase_order: list[PhaseModule] = []
            while ready:
                ready.sort(
                    key=lambda slot: (
                        _is_builtin_slot(slot, phase),
                        registration_order[slot],
                    )
                )
                slot = ready.pop(0)
                module = by_slot[slot]
                phase_order.append(module)
                for dependent in dependents[slot]:
                    indegree[dependent] -= 1
                    if indegree[dependent] == 0:
                        ready.append(dependent)

            if len(phase_order) != len(phase_modules):
                cycle_slots = sorted(slot for slot, degree in indegree.items() if degree > 0)
                raise PhaseDefinitionError("dependency_cycle", cycle_slots)
            compiled.extend(phase_order)
            for module in phase_order:
                available.add(module.slot)
                available.update(module.produces)

        return tuple(compiled)


def _is_builtin_slot(slot: str, phase: LifecyclePhase) -> bool:
    return slot.startswith(f"{phase.value}.")


__all__ = [
    "LifecyclePhase",
    "PhaseContext",
    "PhaseDefinitionError",
    "PhaseModule",
    "PhasePipeline",
    "PluginPhaseFrame",
]
