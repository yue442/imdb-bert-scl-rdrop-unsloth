"""Strict loading of single-file BERT checkpoints, including legacy LayerNorm names."""
from pathlib import Path

from safetensors.torch import load_file


def normalized_bert_weights(checkpoint):
    path = Path(checkpoint) / "model.safetensors"
    weights = load_file(str(path), device="cpu")
    normalized = {}
    renamed = 0
    for key, value in weights.items():
        parts = key.split(".")
        if len(parts) >= 2 and parts[-2] == "LayerNorm":
            replacement = {"gamma": "weight", "beta": "bias"}.get(parts[-1])
            if replacement:
                parts[-1] = replacement
                renamed += 1
        target = ".".join(parts)
        if target in normalized:
            raise ValueError(f"Checkpoint key collision: {target}")
        normalized[target] = value
    return normalized, renamed


def load_bert_checkpoint_into(model, checkpoint):
    if model.config.model_type != "bert" or hasattr(model, "peft_config"):
        raise ValueError("Strict legacy loader supports ordinary BERT, not PEFT")
    weights, renamed = normalized_bert_weights(checkpoint)
    model.load_state_dict(weights, strict=True)
    return {"strict": True, "renamed_layernorm_keys": renamed,
            "checkpoint": str(Path(checkpoint).resolve())}
