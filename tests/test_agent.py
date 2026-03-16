"""tests for agent integration."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.agent.agent import create_repose_agent, read_file
from src.agent.repo_model import ReposeAPIModel


def test_repose_api_model_generate() -> None:
    """ReposeAPIModel.generate returns ChatMessage from API response."""
    from smolagents.models import ChatMessage, MessageRole

    model = ReposeAPIModel(api_url="http://localhost:9999/v1/chat/completions")

    mock_response = type("R", (), {"json": lambda s: {"choices": [{"message": {"content": "hello"}}]}, "raise_for_status": lambda s: None})()

    with patch("httpx.Client") as mock_client:
        mock_client.return_value.__enter__.return_value.post.return_value = mock_response
        result = model.generate([{"role": "user", "content": "hi"}])

    assert isinstance(result, ChatMessage)
    assert result.role == MessageRole.ASSISTANT
    assert result.content == "hello"


def test_repose_api_model_flattens_multimodal_content() -> None:
    """ReposeAPIModel handles content as list of dicts (multimodal format)."""
    from smolagents.models import ChatMessage, MessageRole

    model = ReposeAPIModel(api_url="http://localhost:9999/v1/chat/completions")

    mock_response = type("R", (), {"json": lambda s: {"choices": [{"message": {"content": "ok"}}]}, "raise_for_status": lambda s: None})()

    with patch("httpx.Client") as mock_client:
        mock_client.return_value.__enter__.return_value.post.return_value = mock_response
        result = model.generate([
            {"role": "user", "content": [{"type": "text", "text": "hello"}]}
        ])

    assert result.content == "ok"


def test_read_file_tool() -> None:
    """read_file tool returns file contents."""
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("hello world")
        path = f.name

    try:
        out = read_file(path)
        assert out == "hello world"
    finally:
        import os
        os.unlink(path)


def test_create_repose_agent_returns_tool_calling_agent() -> None:
    """create_repose_agent returns a ToolCallingAgent."""
    from smolagents import ToolCallingAgent

    agent = create_repose_agent()
    assert isinstance(agent, ToolCallingAgent)


def test_create_repose_agent_has_tools() -> None:
    """create_repose_agent configures read_file, web_search, python_interpreter."""
    agent = create_repose_agent()
    tools = agent.tools
    assert isinstance(tools, dict), "ToolCallingAgent.tools is expected to be a dict"
    tool_names = list(tools.keys())
    assert "read_file" in tool_names
    assert "web_search" in tool_names
    assert "python_interpreter" in tool_names


def test_create_repose_agent_uses_repose_model() -> None:
    """create_repose_agent uses ReposeAPIModel as the model."""
    agent = create_repose_agent(api_url="http://custom:9000/chat")
    assert isinstance(agent.model, ReposeAPIModel)
    assert agent.model.api_url == "http://custom:9000/chat"
