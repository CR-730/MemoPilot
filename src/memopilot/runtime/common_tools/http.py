"""公共联网工具使用的最小异步 HTTP 适配层。

接口保持旧公共工具的 ``HttpRequester`` 合同，底层复用 MemoPilot 已有
httpx 依赖；不改变 web_fetch 的 URL 校验、大小限制和内容转换逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class RequestBudget:
    total_timeout_s: float


class HttpRequester:
    async def get(self, url: str, **kwargs: Any) -> httpx.Response:
        timeout_s = float(kwargs.pop("timeout_s", 30))
        kwargs.pop("budget", None)
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            return await client.get(url, **kwargs)


def get_default_http_requester(_name: str = "external_default") -> HttpRequester:
    return HttpRequester()
