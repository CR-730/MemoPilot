from memopilot.memory.procedures import build_procedure_rule_schema


def test_rule_schema_combines_explicit_and_inferred_tool_constraints() -> None:
    schema = build_procedure_rule_schema(
        "查 Steam 信息时不能直接使用 web_search，必须先使用 steam_mcp。",
        tool_requirement="steam_mcp",
        steps=["最后可以使用 web_search 补充验证"],
    )

    assert schema["required_tools"] == ["steam_mcp"]
    assert schema["forbidden_tools"] == ["web_search"]
    assert {"steam_mcp", "web_search"} <= set(schema["mentioned_tools"])


def test_rule_schema_supports_prototype_cues_and_adjacent_ascii_aliases() -> None:
    schema = build_procedure_rule_schema(
        "查资料时别直接用 web search，应先用 steam mcp，最后应该用 browser。"
    )

    assert "web_search" in schema["forbidden_tools"]
    assert {"steam_mcp", "browser"} <= set(schema["required_tools"])
    assert {"web_search", "steam_mcp"} <= set(schema["mentioned_tools"])

