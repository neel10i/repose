"""
LoRA training script for REPO-modified models.

handles: model loading, patching, LoRA wrapping, training loop, checkpointing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.patch_olmo import patch_olmo2_with_repo
from src.lora_config import get_repo_lora_config

if TYPE_CHECKING:
    from transformers import AutoTokenizer, PreTrainedModel
    from peft import PeftModel


def train(
    model_name: str = "allenai/OLMo-2-0425-1B",
    output_dir: str = "./checkpoints",
    num_epochs: int = 3,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    device: str = "cuda",
    gradient_checkpointing: bool = True,
) -> None:
    """
    fine-tune OLMo-2 with REPO using LoRA.

    workflow:
        1. load base model from HF
        2. patch with REPO modules (layers >= 5)
        3. wrap with LoRA (attention projections only)
        4. train on agent-style traces
        5. save checkpoints

    args:
        model_name: HuggingFace model identifier
        output_dir: where to save checkpoints
        num_epochs: training epochs
        batch_size: per-device batch (usually 1 for 1B model)
        learning_rate: optimizer learning rate
        device: "cuda" or "cpu"
        gradient_checkpointing: enable to save memory
    """
    from peft import get_peft_model
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    # create output directory
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # load and patch model with REPO
    model, repo_modules = patch_olmo2_with_repo(
        model_name=model_name,
        start_layer=5,
        device=device,
    )

    # apply LoRA (only attention projections, not REPO modules)
    # model from patch_olmo2_with_repo is a PreTrainedModel (from AutoModelForCausalLM)
    lora_config = get_repo_lora_config()
    base_model: PreTrainedModel = model  # type: ignore[assignment]
    model = get_peft_model(base_model, lora_config)

    # enable gradient checkpointing for memory efficiency
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()  # type: ignore[operator]

    # setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # placeholder: load dataset
    # in practice, this would load synthetic agent traces
    # for now, just a stub
    print(f"training stub: model={model_name}, epochs={num_epochs}, lr={learning_rate}")
    print(f"REPO modules: {len(repo_modules)} layers patched")
    print(f"LoRA trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # training loop placeholder: this is where the system will load real agent traces,
    # run forward/backward passes, and step the optimizer once the data format is defined.
    for epoch in range(num_epochs):
        print(f"epoch {epoch + 1}/{num_epochs}")
        # save checkpoint
        checkpoint_path = out_path / f"checkpoint-{epoch}"
        model.save_pretrained(str(checkpoint_path))
        tokenizer.save_pretrained(str(checkpoint_path))
        print(f"saved checkpoint to {checkpoint_path}")

    print("training complete")


if __name__ == "__main__":
    # simple CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="train REPO + LoRA on OLMo-2")
    parser.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--output", default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    train(
        model_name=args.model,
        output_dir=args.output,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=args.device,
    )
