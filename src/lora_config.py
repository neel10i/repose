"""
LoRA configuration for REPO fine-tuning.

strategy: LoRA on attention projections, full training for REPO modules.
"""

from peft import LoraConfig, TaskType


def get_repo_lora_config(
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
) -> LoraConfig:
    """
    returns LoRA config for REPO fine-tuning.

    targets attention projections (q/k/v/o) with LoRA adapters.
    REPO modules are excluded from LoRA (they train fully).

    args:
        r: LoRA rank (default 8)
        lora_alpha: LoRA alpha scaling (default 16)
        lora_dropout: dropout for LoRA layers (default 0.05)

    returns:
        LoraConfig ready for get_peft_model()
    """
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
    )
