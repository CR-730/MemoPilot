"""MemoPilot 的原型式启动入口。"""

from __future__ import annotations

import asyncio
import sys

from memopilot.cli import main as cli_main

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        from memopilot.channels.cli import CLIClient

        asyncio.run(CLIClient().run())
        raise SystemExit(0)
    # 与原型保持一致：不带子命令时直接启动完整异步服务。
    if len(sys.argv) == 1 or sys.argv[1].startswith("-"):
        sys.argv.insert(1, "run")
    cli_main()
