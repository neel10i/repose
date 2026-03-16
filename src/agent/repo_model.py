"""
smolagents Model backend that calls the repose FastAPI server.

uses httpx to POST to the openai-compatible chat completions endpoint.
"""

from __future__ import annotations

from typing import Any

import httpx
from smolagents.models import ChatMessage, MessageRole, Model


def _flatten_content(content: str | list[dict[str, Any]] | None) -> str:
    """extracts plain text from message content (handles multimodal list format)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text", "")))
        elif isinstance(item, dict) and "text" in item:
            parts.append(str(item["text"]))
    return "\n".join(parts) if parts else ""


def _messages_to_api_format(messages: list[ChatMessage | dict[str, Any]]) -> list[dict[str, str]]:
    """converts smolagents message format to our API's messages list."""
    out: list[dict[str, str]] = []
    for m in messages:
        if isinstance(m, ChatMessage):
            role = m.role.value if hasattr(m.role, "value") else str(m.role)
            content = _flatten_content(m.content)
        else:
            role = str(m.get("role", "user"))
            content = _flatten_content(m.get("content"))
        out.append({"role": role, "content": content})
    return out


class ReposeAPIModel(Model):
    """
    smolagents Model that calls the repose FastAPI server.

    posts to the openai-compatible /v1/chat/completions endpoint.
    """

    def __init__(
        self,
        api_url: str = "http://localhost:8000/v1/chat/completions",
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.api_url = api_url
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(
        self,
        messages: list[ChatMessage | dict[str, Any]],
        stop_sequences: list[str] | None = None,
        **kwargs: Any,
    ) -> ChatMessage:
        api_messages = _messages_to_api_format(messages)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)

        payload = {
            "messages": api_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        with httpx.Client(timeout=120.0) as client:
            response = client.post(self.api_url, json=payload)
            response.raise_for_status()

        data = response.json()
        choice = data["choices"][0]
        content = choice["message"]["content"]

        return ChatMessage(role=MessageRole.ASSISTANT, content=content)
