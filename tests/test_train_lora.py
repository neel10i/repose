"""
tests for LoRA training script.

verifies: train_lora module can be imported, key functions exist.
"""

from __future__ import annotations


def test_train_lora_module_imports():
    """train_lora module can be imported without error."""
    from src.training import train_lora

    assert hasattr(train_lora, "train")


def test_train_function_signature():
    """train function accepts expected arguments."""
    from src.training.train_lora import train
    import inspect

    sig = inspect.signature(train)
    params = list(sig.parameters.keys())

    # key params we expect
    assert "model_name" in params
    assert "output_dir" in params
    assert "num_epochs" in params
