from memopilot.proactive.content_turn import _CONTENT_SYSTEM_PROMPT


def test_prompt_keeps_prototype_operating_contract() -> None:
    required_fragments = (
        "Workspace 主动上下文代表用户当前对主动推送的明确要求",
        "当本轮 alert 和 content 都为空时",
        "get_recent_chat",
        "web_fetch 失败（404/超时/二进制图片）不能直接 mark_not_interesting",
        "message_push + finish_turn(decision=reply)",
        "evidence 格式",
    )
    for fragment in required_fragments:
        assert fragment in _CONTENT_SYSTEM_PROMPT
