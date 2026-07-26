import asyncio

import httpx
import pytest

from memopilot.runtime.common_tools.http import (
    HttpRequester,
    RequestBudget,
    RetryPolicy,
    SharedHttpResources,
)


@pytest.mark.asyncio
async def test_http_requester_retries_timeout_then_succeeds() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadTimeout("timeout", request=request)
        return httpx.Response(200, request=request, text="ok")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    requester = HttpRequester(
        client=client,
        retry_policy=RetryPolicy(max_attempts=2, base_delay_s=0.0, max_delay_s=0.0),
        default_timeout_s=1.0,
        default_budget=RequestBudget(total_timeout_s=2.0),
        sleep=lambda _: asyncio.sleep(0),
    )

    response = await requester.get("https://example.com")

    assert response.status_code == 200
    assert calls == 2
    await client.aclose()


@pytest.mark.asyncio
async def test_shared_http_resources_close_is_idempotent() -> None:
    resources = SharedHttpResources()

    await resources.aclose()
    await resources.aclose()

    assert resources.closed is True
