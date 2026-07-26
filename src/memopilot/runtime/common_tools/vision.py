"""视觉工具：使用独立的 VL 模型分析图片，返回文本描述。"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path

from memopilot.runtime.providers import VisionProvider
from memopilot.runtime.tools import Tool

from .filesystem import (
    _detect_supported_image_mime_from_header,
    _resolve_path,
)

_VL_MAX_FILE_BYTES = 20 * 1024 * 1024
_VL_MAX_DATA_URI_BYTES = 8 * 1024 * 1024
_VL_MAX_EDGE = 4096


def _encode_image_data_uri(file_path: Path) -> str:
    file_size = os.path.getsize(file_path)
    if file_size > _VL_MAX_FILE_BYTES:
        raise ValueError(
            f"图片文件过大（{file_size / 1024 / 1024:.1f}MB），"
            f"上限为 {_VL_MAX_FILE_BYTES / 1024 / 1024:.0f}MB。"
            "请压缩图片后重试，或裁剪到只包含需要分析的区域。"
        )

    raw = file_path.read_bytes()
    mime = _detect_supported_image_mime_from_header(raw[:4096])
    if mime is None:
        raise ValueError("不支持的图片格式。仅支持 PNG、JPEG、GIF、BMP、WebP。")

    try:
        from PIL import Image, ImageOps  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        raise ValueError(  # noqa: B904
            "当前环境未安装 Pillow，无法校验图片。请安装 Pillow 后重试。"
        )

    try:
        with Image.open(file_path) as image:
            image.verify()
    except Exception as exc:
        raise ValueError("图片文件无法解码或已损坏。请确认这是有效图片。") from exc

    with Image.open(file_path) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode not in ("RGB", "L"):
            canvas = Image.new("RGB", image.size, (255, 255, 255))
            alpha = image.getchannel("A") if "A" in image.getbands() else None
            canvas.paste(image.convert("RGB"), mask=alpha)
            image = canvas
        elif image.mode == "L":
            image = image.convert("RGB")

        raw_b64_len = len(base64.b64encode(raw).decode())
        if max(image.size) > _VL_MAX_EDGE or raw_b64_len > _VL_MAX_DATA_URI_BYTES:
            image.thumbnail((_VL_MAX_EDGE, _VL_MAX_EDGE))

        if (
            raw_b64_len <= _VL_MAX_DATA_URI_BYTES
            and max(image.size) <= _VL_MAX_EDGE
        ):
            buffer = io.BytesIO()
            if mime == "image/jpeg":
                image.save(buffer, format="JPEG", quality=95, optimize=True)
                clean_mime = "image/jpeg"
            else:
                image.save(buffer, format="PNG", optimize=True)
                clean_mime = "image/png"
            clean_b64 = base64.b64encode(buffer.getvalue()).decode()
            if len(clean_b64) <= _VL_MAX_DATA_URI_BYTES:
                return f"data:{clean_mime};base64,{clean_b64}"

        best: bytes | None = None
        for quality in (85, 75, 65, 55, 45):
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=quality, optimize=True)
            candidate = buffer.getvalue()
            candidate_b64 = base64.b64encode(candidate).decode()
            best = candidate
            if len(candidate_b64) <= _VL_MAX_DATA_URI_BYTES:
                return f"data:image/jpeg;base64,{candidate_b64}"

    if best is None:
        raise ValueError("图片压缩失败")
    best_b64 = base64.b64encode(best).decode()
    raise ValueError(
        f"图片压缩后仍然过大（{len(best_b64) / 1024 / 1024:.1f}MB base64），"
        f"上限为 {_VL_MAX_DATA_URI_BYTES / 1024 / 1024:.0f}MB。"
        "请继续压缩图片或裁剪到只包含需要分析的区域。"
    )


def build_read_image_vision_tool(
    provider: VisionProvider,
    *,
    workspace: Path,
) -> Tool:
    async def read_image_vision(path: str, prompt: str) -> str:
        try:
            file_path = _resolve_path(path, workspace)
            if not file_path.exists():
                return f"错误：文件不存在：{path}"
            if not file_path.is_file():
                return f"错误：路径不是文件：{path}"
            data_uri = _encode_image_data_uri(file_path)
        except ValueError as exc:
            return f"图片处理失败：{exc}"
        except Exception as exc:
            return f"读取图片文件失败：{exc}"

        try:
            return await provider.complete_vision(data_uri=data_uri, prompt=prompt)
        except Exception as exc:
            return f"调用视觉模型失败：{exc}"

    return Tool(
        name="read_image_vision",
        description=(
            "使用独立的视觉模型分析图片内容。主模型无法直接查看图片时使用此工具。"
            "你需要提供一个 prompt 来说明你想从图片中了解什么。\n\n"
            "参数说明：\n"
            "- path：图片文件的路径\n"
            "- prompt：描述你想从这张图片中了解什么内容，越具体越好。"
            "例如 '图中有什么文字？'、'描述这张图片中的物体和场景'、"
            "'这张表格中第3行的数据是什么？'\n\n"
            "限制：原始文件不超过20MB，超限图片会自动缩放至最宽/最高4096像素并压缩。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "图片文件的路径"},
                "prompt": {
                    "type": "string",
                    "description": "描述你想从图片中了解什么内容，越具体越好",
                },
            },
            "required": ["path", "prompt"],
        },
        handler=read_image_vision,
        source="builtin:common",
    )


__all__ = ["build_read_image_vision_tool"]
