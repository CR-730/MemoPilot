"""主模型不支持多模态时使用的独立视觉工具。"""

from __future__ import annotations

from pathlib import Path

from memopilot.runtime.providers import VisionProvider
from memopilot.runtime.tools import Tool

from .filesystem import (
    _detect_supported_image_mime_from_header,
    _encode_image_for_model,
    _resolve_path,
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
            mime = _detect_supported_image_mime_from_header(file_path.read_bytes()[:4096])
            if mime is None:
                return "图片处理失败：不支持的图片格式。仅支持 PNG、JPEG、GIF、BMP、WebP。"
            encoded_mime, payload, _ = _encode_image_for_model(file_path, mime)
            return await provider.complete_vision(
                data_uri=f"data:{encoded_mime};base64,{payload}",
                prompt=prompt,
            )
        except PermissionError as exc:
            return f"图片处理失败：{exc}"
        except Exception as exc:
            return f"调用视觉模型失败：{exc}"

    return Tool(
        name="read_image_vision",
        description=(
            "使用独立的视觉模型分析图片内容。主模型无法直接查看图片时使用此工具。"
            "path 是 workspace 内的图片路径；prompt 用于说明希望从图片中了解什么。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "图片文件路径"},
                "prompt": {"type": "string", "description": "希望分析的图片内容"},
            },
            "required": ["path", "prompt"],
            "additionalProperties": False,
        },
        handler=read_image_vision,
        source="builtin:common",
    )


__all__ = ["build_read_image_vision_tool"]
