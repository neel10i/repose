"""
fastapi server with openai-compatible chat completions endpoint.

serves REPO-modified models for agent inference.
"""

from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, AsyncGenerator, List, cast

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.patch_olmo import patch_olmo2_with_repo


# module-level state persists across requests
_model: torch.nn.Module | None = None
_tokenizer: PreTrainedTokenizerBase | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """load REPO-patched model at startup, clean up on shutdown."""
    global _model, _tokenizer

    model_name = os.getenv("REPOSE_MODEL", "allenai/OLMo-2-0425-1B")
    lora_path = os.getenv("REPOSE_LORA_PATH", "")
    device = os.getenv("REPOSE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

    print(f"loading model {model_name} on {device}...")

    _model, _ = patch_olmo2_with_repo(
        model_name=model_name,
        start_layer=5,
        device=device,
    )

    # optionally load LoRA weights on top
    if lora_path:
        from peft import PeftModel
        from transformers import PreTrainedModel

        assert isinstance(_model, PreTrainedModel)
        _model = PeftModel.from_pretrained(_model, lora_path)
        print(f"loaded LoRA weights from {lora_path}")

    _model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _tokenizer = tokenizer

    print("model ready.")
    yield

    # free memory on shutdown
    del _model, _tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("model unloaded.")


app = FastAPI(
    title="repose",
    description="REPO-modified model serving",
    lifespan=lifespan,
)


class ChatMessage(BaseModel):
    """single message in a chat conversation."""

    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """openai-compatible chat completion request."""

    messages: List[ChatMessage]
    max_tokens: int = 256
    temperature: float = 0.7
    stream: bool = False


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> dict:
    """
    openai-compatible chat completions endpoint.

    tokenizes messages, runs generation, returns decoded response.
    streaming not yet implemented.
    """
    if _model is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    model = cast("PreTrainedModel", _model)

    # apply chat template to convert messages list into a single prompt string
    prompt = _tokenizer.apply_chat_template(
        [{"role": m.role, "content": m.content} for m in request.messages],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(prompt, str):
        prompt = str(prompt)

    inputs = _tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    input_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(  # type: ignore[union-attr]
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            do_sample=request.temperature > 0,
            pad_token_id=_tokenizer.pad_token_id,
            eos_token_id=_tokenizer.eos_token_id,
        )

    # decode only the newly generated tokens (skip the prompt)
    response_ids = outputs[0][input_len:]
    response_text = _tokenizer.decode(response_ids, skip_special_tokens=True)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": os.getenv("REPOSE_MODEL", "allenai/OLMo-2-0425-1B"),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": input_len,
            "completion_tokens": len(response_ids),
            "total_tokens": input_len + len(response_ids),
        },
    }


@app.get("/health")
async def health() -> dict:
    """health check — returns whether model is loaded."""
    return {"status": "ok", "model_loaded": _model is not None}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
