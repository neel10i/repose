"""
fastapi server for REPO model serving.
"""

from src.server.main import app, ChatCompletionRequest

__all__ = ["app", "ChatCompletionRequest"]
