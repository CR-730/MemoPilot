"""飞书私聊长连接 Channel。

第一版使用 lark-oapi 长连接，支持私聊消息解析、身份索引、资源收发与中断，
不包含 webhook、FastAPI 或 Uvicorn 接收分支。
"""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import threading
import time
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from memopilot.channels.base import AttachmentStore, MessageDeduper, SessionIdentityIndex
from memopilot.channels.contracts import (
    InboundMessage,
    InterruptController,
    MessageBus,
    SendReceipt,
)

logger = logging.getLogger(__name__)

_API_BASE = "https://open.feishu.cn/open-apis"
_TOKEN_URL = f"{_API_BASE}/auth/v3/tenant_access_token/internal"
_SEEN_EVENT_MAXSIZE = 500


@dataclass(frozen=True, slots=True)
class _TokenCache:
    token: str
    expires_at: float


class FeishuApiError(RuntimeError):
    def __init__(self, business_code: int, message: str) -> None:
        super().__init__(f"Feishu API request failed: code={business_code} msg={message}")
        self.business_code = business_code


class FeishuChannel:
    def __init__(
        self,
        *,
        app_id: str,
        app_secret: str,
        bus: MessageBus,
        identity_index: SessionIdentityIndex,
        attachment_store: AttachmentStore,
        allow_from: tuple[str, ...] = (),
        interrupt_controller: InterruptController | None = None,
        channel_name: str = "feishu",
        client: httpx.AsyncClient | None = None,
        ws_event_timeout_seconds: float = 30,
        ws_stop_timeout_seconds: float = 30,
    ) -> None:
        if ws_event_timeout_seconds <= 0 or ws_stop_timeout_seconds <= 0:
            raise ValueError("飞书长连接事件与停止超时必须大于 0")
        self._app_id = app_id
        self._app_secret = app_secret
        self._bus = bus
        self._identity_index = identity_index
        self._attachments = attachment_store
        self._allow_from = {str(value) for value in allow_from}
        self._interrupt_controller = interrupt_controller
        self._channel = channel_name
        self._client = client or httpx.AsyncClient(timeout=30.0)
        self._owns_client = client is None
        self._token: _TokenCache | None = None
        self._deduper = MessageDeduper(_SEEN_EVENT_MAXSIZE)
        self._main_loop: asyncio.AbstractEventLoop | None = None
        self._ws_client: Any | None = None
        self._ws_thread: threading.Thread | None = None
        self._ws_stop_requested = threading.Event()
        self._ws_ready = threading.Event()
        self._ws_start_error: BaseException | None = None
        self._ws_event_timeout_seconds = ws_event_timeout_seconds
        self._ws_stop_timeout_seconds = ws_stop_timeout_seconds

    async def start(self) -> None:
        self._identity_index.rebuild()
        self._start_ws_client()
        ready = await asyncio.to_thread(
            self._ws_ready.wait,
            self._ws_stop_timeout_seconds,
        )
        if not ready:
            await self._stop_ws_client()
            raise TimeoutError("飞书长连接初始化超时")
        if self._ws_start_error is not None:
            error = self._ws_start_error
            await self._stop_ws_client()
            raise RuntimeError("飞书长连接初始化失败") from error

    async def stop(self) -> None:
        try:
            await self._stop_ws_client()
        finally:
            if self._owns_client:
                await self._client.aclose()

    async def handle_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        header = _as_dict(payload.get("header"))
        if str(header.get("event_type") or "") != "im.message.receive_v1":
            return {"ok": True, "ignored": "event_type"}
        event = _as_dict(payload.get("event"))
        message = _as_dict(event.get("message"))
        event_id = str(header.get("event_id") or "")
        message_id = str(message.get("message_id") or "")
        dedupe_key = event_id or message_id
        if dedupe_key and not self._deduper.reserve(dedupe_key):
            return {"ok": True, "deduped": True}

        try:
            result = await self._handle_received_message(
                event,
                message,
                event_id=event_id,
                message_id=message_id,
            )
        except BaseException:
            if dedupe_key:
                self._deduper.release(dedupe_key)
            raise
        if dedupe_key:
            self._deduper.commit(dedupe_key)
        return result

    async def _handle_received_message(
        self,
        event: dict[str, Any],
        message: dict[str, Any],
        *,
        event_id: str,
        message_id: str,
    ) -> dict[str, Any]:

        sender_ids = _as_dict(_as_dict(event.get("sender")).get("sender_id"))
        sender = _sender_identity(sender_ids)
        if not sender:
            return {"ok": True, "ignored": "missing_sender"}
        if self._allow_from and not _is_allowed(self._allow_from, sender_ids, sender):
            return {"ok": True, "ignored": "unauthorized"}
        if str(message.get("chat_type") or "") != "p2p":
            return {"ok": True, "ignored": "non_private"}
        chat_id = str(message.get("chat_id") or "")
        if not chat_id:
            return {"ok": True, "ignored": "missing_chat_id"}

        text = _extract_message_text(message)
        media = await self._extract_message_media(message)
        metadata = {
            "message_id": message_id,
            "event_id": event_id,
            "chat_type": str(message.get("chat_type") or ""),
            "message_type": str(message.get("message_type") or ""),
            "open_id": str(sender_ids.get("open_id") or ""),
            "user_id": str(sender_ids.get("user_id") or ""),
            "union_id": str(sender_ids.get("union_id") or ""),
        }
        inbound = InboundMessage(
            channel=self._channel,
            sender=sender,
            chat_id=chat_id,
            content=text,
            media=tuple(media),
            metadata=metadata,
        )
        self._identity_index.remember(
            session_key=inbound.session_key,
            chat_id=chat_id,
            identities={key: metadata[key] for key in ("open_id", "user_id", "union_id")},
        )
        if text.strip() == "/stop":
            if self._interrupt_controller is None:
                return {"ok": True, "ignored": "interrupt_unavailable"}
            acknowledgement = await self._interrupt_controller.request_interrupt(inbound)
            await self.send(
                chat_id,
                acknowledgement.message,
                provider_uuid=acknowledgement.provider_uuid,
            )
            return {"ok": True}

        await self._bus.publish_inbound(inbound)
        return {"ok": True}

    async def _extract_message_media(self, message: dict[str, Any]) -> tuple[str, ...]:
        if str(message.get("message_type") or "") != "image":
            return ()
        message_id = str(message.get("message_id") or "")
        image_key = str(_parse_message_content(message).get("image_key") or "").strip()
        if not message_id or not image_key:
            return ()
        try:
            data, content_type = await self._download_message_resource(
                message_id=message_id,
                resource_key=image_key,
                resource_type="image",
            )
        except Exception as exc:
            logger.warning("[feishu] image download failed message_id=%s: %s", message_id, exc)
            return ()
        suffix = _suffix_from_content_type(content_type) or ".jpg"
        path = self._attachments.write_bytes(data, prefix="feishu_image_", suffix=suffix)
        return (str(path),)

    async def send(self, chat_id: str, message: str, *, provider_uuid: str) -> SendReceipt:
        token = await self._get_tenant_access_token()
        body = {
            "receive_id": str(chat_id),
            "msg_type": "text",
            "content": json.dumps({"text": message}, ensure_ascii=False),
            "uuid": provider_uuid,
        }
        data = await self._api_request(
            "POST",
            "/im/v1/messages",
            body,
            token=token,
            params={"receive_id_type": "chat_id"},
        )
        message_id = str(data.get("message_id") or "")
        if not message_id:
            raise RuntimeError("Feishu send response missing message_id")
        return SendReceipt(message_id=message_id)

    async def send_stream(
        self,
        chat_id: str,
        message: str,
        *,
        provider_uuid: str,
    ) -> SendReceipt:
        return await self.send(chat_id, message, provider_uuid=provider_uuid)

    async def send_card(
        self,
        chat_id: str,
        content: str,
        *,
        provider_uuid: str,
    ) -> SendReceipt:
        token = await self._get_tenant_access_token()
        body = {
            "receive_id": str(chat_id),
            "msg_type": "interactive",
            "content": content,
            "uuid": provider_uuid,
        }
        data = await self._api_request(
            "POST",
            "/im/v1/messages",
            body,
            token=token,
            params={"receive_id_type": "chat_id"},
        )
        return _send_receipt(data)

    async def patch_card(self, message_id: str, content: str) -> None:
        token = await self._get_tenant_access_token()
        await self._api_request(
            "PATCH",
            f"/im/v1/messages/{message_id}",
            {"content": content},
            token=token,
        )

    async def send_image(
        self,
        chat_id: str,
        image_path_or_url: str,
        *,
        provider_uuid: str,
    ) -> SendReceipt:
        token = await self._get_tenant_access_token()
        image_name, image_bytes = await self._read_local_or_url(image_path_or_url)
        image_key = await self._upload_image(token, image_name, image_bytes)
        body = {
            "receive_id": str(chat_id),
            "msg_type": "image",
            "content": json.dumps({"image_key": image_key}, ensure_ascii=False),
            "uuid": provider_uuid,
        }
        data = await self._api_request(
            "POST",
            "/im/v1/messages",
            body,
            token=token,
            params={"receive_id_type": "chat_id"},
        )
        return _send_receipt(data)

    async def send_file(
        self,
        chat_id: str,
        file_path: str,
        *,
        provider_uuid: str,
        name: str | None = None,
    ) -> SendReceipt:
        token = await self._get_tenant_access_token()
        path = Path(file_path)
        file_name = name or path.name
        file_bytes = await asyncio.to_thread(path.read_bytes)
        file_key = await self._upload_file(token, file_name, file_bytes)
        body = {
            "receive_id": str(chat_id),
            "msg_type": "file",
            "content": json.dumps(
                {"file_key": file_key, "file_name": file_name},
                ensure_ascii=False,
            ),
            "uuid": provider_uuid,
        }
        data = await self._api_request(
            "POST",
            "/im/v1/messages",
            body,
            token=token,
            params={"receive_id_type": "chat_id"},
        )
        return _send_receipt(data)

    def _start_ws_client(self) -> None:
        if self._ws_thread is not None and self._ws_thread.is_alive():
            return
        self._ws_stop_requested.clear()
        self._ws_ready.clear()
        self._ws_start_error = None
        self._main_loop = asyncio.get_running_loop()
        self._ws_thread = threading.Thread(
            target=self._run_ws_client,
            name=f"{self._channel}_ws_client",
            daemon=True,
        )
        self._ws_thread.start()

    def _run_ws_client(self) -> None:
        try:
            import lark_oapi as lark  # type: ignore[import-untyped]

            if self._ws_stop_requested.is_set():
                return
            handler = (
                lark.EventDispatcherHandler.builder("", "")
                .register_p2_im_message_receive_v1(self._on_ws_message)
                .build()
            )
            client = lark.ws.Client(
                self._app_id,
                self._app_secret,
                log_level=lark.LogLevel.WARNING,
                event_handler=handler,
                auto_reconnect=False,
            )
            self._ws_client = client
            original_connect = getattr(client, "_connect", None)
            if original_connect is None:
                raise RuntimeError("当前飞书 SDK 不支持长连接就绪探测")

            async def connect_and_mark_ready() -> None:
                await original_connect()
                if self._ws_stop_requested.is_set():
                    client._auto_reconnect = False
                    raise RuntimeError("飞书长连接在首次握手期间收到停止请求")
                client._auto_reconnect = True
                self._ws_ready.set()

            client._connect = connect_and_mark_ready
        except Exception as exc:
            self._ws_start_error = exc
            logger.warning("[feishu] long connection initialization failed: %s", exc)
            self._ws_ready.set()
            return
        if self._ws_stop_requested.is_set():
            return
        try:
            self._ws_client.start()
        except Exception as exc:
            self._ws_start_error = exc
            self._ws_ready.set()
            logger.warning("[feishu] long connection exited: %s", exc)

    def _on_ws_message(self, data: Any) -> None:
        payload = _marshal_ws_event(data)
        loop = self._main_loop
        if loop is None or loop.is_closed():
            raise RuntimeError("飞书事件到达时主 asyncio loop 不可用")
        future = asyncio.run_coroutine_threadsafe(self.handle_event(payload), loop)
        try:
            future.result(timeout=self._ws_event_timeout_seconds)
        except FutureTimeoutError:
            future.cancel()
            logger.error("[feishu] inbound persistence timed out")
            raise
        except Exception as exc:
            logger.warning("[feishu] ws event handling failed: %s", exc)
            raise

    async def _stop_ws_client(self) -> None:
        self._ws_stop_requested.set()
        thread = self._ws_thread
        if thread is None:
            return
        deadline = asyncio.get_running_loop().time() + self._ws_stop_timeout_seconds
        while thread.is_alive():
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError("飞书长连接线程未在停止超时内退出")
            try:
                await self._disconnect_ws_client_once(timeout_seconds=remaining)
            except TimeoutError as exc:
                raise TimeoutError("飞书长连接断开操作超时") from exc
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError("飞书长连接线程未在停止超时内退出")
            await asyncio.to_thread(thread.join, min(0.05, remaining))
        self._ws_thread = None
        self._ws_client = None

    async def _disconnect_ws_client_once(self, *, timeout_seconds: float) -> None:
        client = self._ws_client
        disconnect = getattr(client, "_disconnect", None) if client is not None else None
        if disconnect is None:
            return
        if client is not None and hasattr(client, "_auto_reconnect"):
            client._auto_reconnect = False
        try:
            ws_loop = self._get_ws_loop()
            if ws_loop is not None and ws_loop.is_running():
                future = asyncio.run_coroutine_threadsafe(disconnect(), ws_loop)
                await asyncio.wait_for(
                    asyncio.wrap_future(future),
                    timeout=timeout_seconds,
                )
                if ws_loop is not asyncio.get_running_loop():
                    ws_loop.call_soon_threadsafe(ws_loop.stop)
        except TimeoutError:
            raise
        except Exception as exc:
            logger.debug("[feishu] long connection disconnect skipped: %s", exc)

    @staticmethod
    def _get_ws_loop() -> Any:
        import lark_oapi.ws.client as ws_client_mod  # type: ignore[import-untyped]

        return getattr(ws_client_mod, "loop", None)

    async def _get_tenant_access_token(self) -> str:
        now = time.time()
        if self._token is not None and self._token.expires_at > now:
            return self._token.token
        response = await self._client.post(
            _TOKEN_URL,
            json={"app_id": self._app_id, "app_secret": self._app_secret},
        )
        response.raise_for_status()
        payload = response.json()
        if int(payload.get("code", -1)) != 0:
            raise RuntimeError(f"Feishu token request failed: {payload.get('msg') or payload}")
        token = str(payload.get("tenant_access_token") or "")
        if not token:
            raise RuntimeError("Feishu token response missing tenant_access_token")
        expires_in = int(payload.get("expire") or 7200)
        self._token = _TokenCache(token=token, expires_at=now + max(60, expires_in - 300))
        return token

    async def _api_request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        token: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        access_token = token or await self._get_tenant_access_token()
        response = await self._client.request(
            method,
            f"{_API_BASE}{path}",
            headers={"Authorization": f"Bearer {access_token}"},
            params=params,
            json=body,
        )
        response.raise_for_status()
        payload = response.json()
        business_code = int(payload.get("code", -1))
        if business_code != 0:
            raise FeishuApiError(business_code, str(payload.get("msg") or payload))
        return _as_dict(payload.get("data"))

    async def _download_message_resource(
        self,
        *,
        message_id: str,
        resource_key: str,
        resource_type: str,
    ) -> tuple[bytes, str]:
        token = await self._get_tenant_access_token()
        response = await self._client.get(
            f"{_API_BASE}/im/v1/messages/{message_id}/resources/{resource_key}",
            headers={"Authorization": f"Bearer {token}"},
            params={"type": resource_type},
        )
        response.raise_for_status()
        return response.content, response.headers.get("content-type", "")

    async def _upload_image(self, token: str, file_name: str, data: bytes) -> str:
        response = await self._client.post(
            f"{_API_BASE}/im/v1/images",
            headers={"Authorization": f"Bearer {token}"},
            data={"image_type": "message"},
            files={"image": (file_name, data, _content_type(file_name))},
        )
        response.raise_for_status()
        payload = response.json()
        if int(payload.get("code", -1)) != 0:
            raise RuntimeError(f"Feishu image upload failed: {payload.get('msg') or payload}")
        image_key = str(_as_dict(payload.get("data")).get("image_key") or "")
        if not image_key:
            raise RuntimeError("Feishu image upload response missing image_key")
        return image_key

    async def _upload_file(self, token: str, file_name: str, data: bytes) -> str:
        response = await self._client.post(
            f"{_API_BASE}/im/v1/files",
            headers={"Authorization": f"Bearer {token}"},
            data={"file_type": _file_type(file_name), "file_name": file_name},
            files={"file": (file_name, data, _content_type(file_name))},
        )
        response.raise_for_status()
        payload = response.json()
        if int(payload.get("code", -1)) != 0:
            raise RuntimeError(f"Feishu file upload failed: {payload.get('msg') or payload}")
        file_key = str(_as_dict(payload.get("data")).get("file_key") or "")
        if not file_key:
            raise RuntimeError("Feishu file upload response missing file_key")
        return file_key

    async def _read_local_or_url(self, value: str) -> tuple[str, bytes]:
        if value.startswith(("http://", "https://")):
            response = await self._client.get(value)
            response.raise_for_status()
            name = Path(value.split("?", 1)[0]).name or "image"
            return name, response.content
        path = Path(value)
        return path.name, await asyncio.to_thread(path.read_bytes)


def _marshal_ws_event(data: Any) -> dict[str, Any]:
    import lark_oapi as lark

    return _as_dict(json.loads(lark.JSON.marshal(data) or "{}"))


def _as_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _parse_message_content(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content")
    if isinstance(content, dict):
        return content
    if isinstance(content, str) and content.strip():
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return {}
        return _as_dict(parsed)
    return {}


def _extract_message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    parsed = _parse_message_content(message)
    if not parsed and isinstance(content, str) and content.strip():
        return content
    message_type = str(message.get("message_type") or "")
    if message_type == "text":
        return str(parsed.get("text") or "")
    if message_type == "post":
        return _flatten_post(parsed)
    if message_type == "image":
        return "[图片]"
    if message_type == "file":
        return str(parsed.get("file_name") or "[文件]")
    return str(parsed or content or "")


def _flatten_post(content: dict[str, Any]) -> str:
    chunks: list[str] = []
    for locale_block in _as_dict(content.get("post")).values():
        block = _as_dict(locale_block)
        title = str(block.get("title") or "").strip()
        if title:
            chunks.append(title)
        for line in block.get("content") or ():
            if not isinstance(line, list):
                continue
            text = "".join(str(_as_dict(part).get("text") or "") for part in line)
            if text.strip():
                chunks.append(text.strip())
    return "\n".join(chunks)


def _sender_identity(sender_ids: dict[str, Any]) -> str:
    return str(
        sender_ids.get("open_id")
        or sender_ids.get("user_id")
        or sender_ids.get("union_id")
        or ""
    )


def _is_allowed(allow_from: set[str], sender_ids: dict[str, Any], sender: str) -> bool:
    candidates = {
        sender,
        str(sender_ids.get("open_id") or ""),
        str(sender_ids.get("user_id") or ""),
        str(sender_ids.get("union_id") or ""),
    }
    return bool(allow_from.intersection(candidates))


def _suffix_from_content_type(content_type: str) -> str:
    clean = str(content_type or "").split(";", 1)[0].strip().lower()
    known = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/gif": ".gif",
        "image/webp": ".webp",
    }
    return known.get(clean) or (mimetypes.guess_extension(clean) if clean else "") or ""


def _send_receipt(data: dict[str, Any]) -> SendReceipt:
    message_id = str(data.get("message_id") or "")
    if not message_id:
        raise RuntimeError("Feishu send response missing message_id")
    return SendReceipt(message_id=message_id)


def _content_type(file_name: str) -> str:
    return mimetypes.guess_type(file_name)[0] or "application/octet-stream"


def _file_type(file_name: str) -> str:
    suffix = Path(file_name).suffix.lower().lstrip(".")
    return suffix if suffix in {"opus", "mp4", "pdf", "doc", "xls", "ppt"} else "stream"


__all__ = ["FeishuApiError", "FeishuChannel"]
