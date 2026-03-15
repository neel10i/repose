"""tests for the fastapi server."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator
from unittest.mock import MagicMock, patch

import pytest
import torch
from fastapi.testclient import TestClient


def _make_mock_model(input_len: int = 5, new_tokens: int = 3) -> MagicMock:
    """builds a mock model whose generate() returns a fixed token tensor."""
    mock = MagicMock()
    mock.parameters.return_value = iter([torch.zeros(1)])
    mock.generate.return_value = torch.zeros(1, input_len + new_tokens, dtype=torch.long)
    return mock


def _make_mock_tokenizer(input_len: int = 5) -> MagicMock:
    """builds a mock tokenizer whose encode/decode cycle is trivial."""
    from transformers import BatchEncoding

    mock = MagicMock()
    mock.apply_chat_template.return_value = "hello"
    # BatchEncoding is a dict subclass that supports .to(device) and ** unpacking
    mock.return_value = BatchEncoding({"input_ids": torch.zeros(1, input_len, dtype=torch.long)})
    mock.pad_token = "<pad>"
    mock.pad_token_id = 0
    mock.eos_token_id = 1
    mock.decode.return_value = "stub reply"
    return mock


@contextmanager
def _patched_client(override_model: bool = True) -> Iterator[TestClient]:
    """
    yields a TestClient with HuggingFace loading replaced by lightweight mocks.

    all tests must go through here or the lifespan will try to hit the network.
    """
    import src.server.main as server_mod

    mock_model = _make_mock_model()
    mock_tok = _make_mock_tokenizer()

    with (
        patch("src.server.main.patch_olmo2_with_repo", return_value=(mock_model, {})),
        patch("src.server.main.AutoTokenizer") as mock_auto_tok,
    ):
        mock_auto_tok.from_pretrained.return_value = mock_tok
        with TestClient(server_mod.app, raise_server_exceptions=False) as c:
            if not override_model:
                # simulate the case where model failed to load after startup
                server_mod._model = None
                server_mod._tokenizer = None
            yield c

    # reset after test so module state doesn't leak
    server_mod._model = None
    server_mod._tokenizer = None


@pytest.fixture()
def client() -> Iterator[TestClient]:
    """test client with a fully mocked loaded model."""
    with _patched_client(override_model=True) as c:
        yield c


def test_health_model_loaded(client: TestClient) -> None:
    """health endpoint reports model_loaded=True when mock model is injected."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["model_loaded"] is True


def test_health_model_not_loaded() -> None:
    """health endpoint reports model_loaded=False when nothing is set."""
    with _patched_client(override_model=False) as c:
        resp = c.get("/health")
    assert resp.status_code == 200
    assert resp.json()["model_loaded"] is False


def test_chat_completions_returns_200(client: TestClient) -> None:
    """endpoint returns 200 for a valid request."""
    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )
    assert resp.status_code == 200


def test_chat_completions_response_shape(client: TestClient) -> None:
    """response has the expected openai-compatible structure."""
    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "what is 2+2?"}]},
    )
    body = resp.json()
    assert "id" in body
    assert body["id"].startswith("chatcmpl-")
    assert body["object"] == "chat.completion"
    assert isinstance(body["choices"], list) and len(body["choices"]) == 1
    choice = body["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert isinstance(choice["message"]["content"], str)


def test_chat_completions_usage_tokens(client: TestClient) -> None:
    """response includes token usage counts that add up correctly."""
    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "count tokens"}]},
    )
    usage = resp.json()["usage"]
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0


def test_chat_completions_503_when_no_model() -> None:
    """endpoint returns 503 when the model is not loaded."""
    with _patched_client(override_model=False) as c:
        resp = c.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 503


def test_chat_completions_multi_turn(client: TestClient) -> None:
    """endpoint handles multi-turn message lists without error."""
    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "system", "content": "you are a helpful assistant"},
                {"role": "user", "content": "what is the capital of France?"},
                {"role": "assistant", "content": "Paris."},
                {"role": "user", "content": "and Germany?"},
            ]
        },
    )
    assert resp.status_code == 200


def test_chat_completions_calls_generate(client: TestClient) -> None:
    """endpoint actually calls model.generate() with the right kwargs."""
    import src.server.main as server_mod

    assert isinstance(server_mod._model, MagicMock)
    client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 64},
    )
    server_mod._model.generate.assert_called_once()
    call_kwargs = server_mod._model.generate.call_args.kwargs
    assert call_kwargs["max_new_tokens"] == 64


def test_request_model_has_messages() -> None:
    """ChatCompletionRequest requires a messages field."""
    from src.server.main import ChatCompletionRequest

    assert "messages" in ChatCompletionRequest.model_fields
