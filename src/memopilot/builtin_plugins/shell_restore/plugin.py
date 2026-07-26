import logging
import os
import shlex
from pathlib import Path

from memopilot.extensions.decorators import on_tool_pre
from memopilot.extensions.plugin_base import Plugin
from memopilot.extensions.plugin_events import PreToolCtx

logger = logging.getLogger("plugin.shell_restore")


def _restore_dir() -> str:
    return os.environ.get("MEMOPILOT_RESTORE_DIR", str(Path.home() / "restore"))


class ShellRestore(Plugin):
    name = "shell_restore"

    @on_tool_pre(tool_name="shell")
    async def rewrite_rm_to_mv(self, event: PreToolCtx) -> dict[str, object] | None:
        command = str(event.arguments.get("command", "")).strip()
        rewritten = self._rewrite_command(command)
        if rewritten is None:
            return None
        restore_path = Path(_restore_dir())
        restore_path.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240
        logger.info("[%s:%s] rm → mv: %r", self.name, self.rewrite_rm_to_mv.__name__, rewritten)
        return dict(event.arguments, command=rewritten)

    def _rewrite_command(self, command: str) -> str | None:
        try:
            tokens = shlex.split(command, posix=True)
        except ValueError:
            return None
        if not tokens:
            return None
        prefix: list[str] = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if Path(token).name == "rm":
                break
            if token == "sudo" or token == "env" or "=" in token:
                prefix.append(token)
                i += 1
                continue
            return None
        if i >= len(tokens) or Path(tokens[i]).name != "rm":
            return None
        i += 1
        targets: list[str] = []
        parsing_options = True
        while i < len(tokens):
            token = tokens[i]
            i += 1
            if parsing_options and token == "--":
                parsing_options = False
                continue
            if parsing_options and token.startswith("-") and token != "-":
                continue
            parsing_options = False
            targets.append(token)
        if not targets:
            return None
        return shlex.join([*prefix, "mv", "--", *targets, _restore_dir()])
