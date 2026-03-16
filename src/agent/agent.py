"""
repose agent definition.

ToolCallingAgent backed by ReposeAPIModel with read_file, web search, and python execution.
"""

from __future__ import annotations

from smolagents import ToolCallingAgent, tool
from smolagents.default_tools import DuckDuckGoSearchTool, PythonInterpreterTool

from src.agent.repo_model import ReposeAPIModel


@tool
def read_file(path: str) -> str:
    """
    read contents of a file from the filesystem.

    args:
        path: path to the file to read (relative or absolute)

    Returns:
        file contents as a string
    """
    with open(path, encoding="utf-8") as f:
        return f.read()


def create_repose_agent(
    api_url: str = "http://localhost:8000/v1/chat/completions",
    max_steps: int = 10,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> ToolCallingAgent:
    """
    creates a ToolCallingAgent that uses the repose FastAPI server as its model.

    tools: read_file, web_search (DuckDuckGo), python_interpreter.

    args:
        api_url: repose chat completions endpoint
        max_steps: maximum agent steps per run
        max_tokens: max tokens per model call
        temperature: sampling temperature

    returns:
        configured ToolCallingAgent
    """
    model = ReposeAPIModel(
        api_url=api_url,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    tools = [
        read_file,
        DuckDuckGoSearchTool(),
        PythonInterpreterTool(),
    ]
    return ToolCallingAgent(
        tools=tools,
        model=model,
        max_steps=max_steps,
    )
