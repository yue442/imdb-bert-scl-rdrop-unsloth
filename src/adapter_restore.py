"""Strictly restore every exported adapter and classification-head tensor."""
import torch
from safetensors.torch import load_file
from peft import get_peft_model_state_dict, set_peft_model_state_dict


def restore_adapter_exact(model, adapter_file):
    weights = load_file(str(adapter_file), device="cpu")
    expected = get_peft_model_state_dict(model)
    if set(weights) != set(expected):
        raise ValueError(f"Adapter key mismatch: missing={sorted(set(expected)-set(weights))}; "
                         f"unexpected={sorted(set(weights)-set(expected))}")
    if not any("lora_A" in key for key in weights) or not any("classifier" in key for key in weights):
        raise ValueError("Both LoRA and trained classifier weights are required")
    for key in weights:
        if weights[key].shape != expected[key].shape:
            raise ValueError(f"Adapter shape mismatch: {key}")
    # PEFT may rewrite dictionary keys in-place while expanding modules_to_save.
    # Preserve the exported-key dictionary for exact post-load comparison.
    result = set_peft_model_state_dict(model, dict(weights), adapter_name="default")
    if result.unexpected_keys:
        raise ValueError(f"Unexpected load keys: {result.unexpected_keys}")
    active_keys = {name for name, _ in model.named_parameters()
                   if "lora_" in name or ".modules_to_save.default." in name}
    if active_keys & set(result.missing_keys):
        raise ValueError(f"Trainable adapter/head not loaded: {active_keys & set(result.missing_keys)}")
    loaded = get_peft_model_state_dict(model)
    for key, value in weights.items():
        actual = loaded[key].detach().cpu()
        if actual.dtype != value.dtype or not torch.equal(actual, value):
            raise ValueError(f"Saved adapter tensor not restored exactly: {key}")
    return {"saved_tensor_count": len(weights), "all_saved_tensors_exact": True,
            "adapter_keys_shapes_and_classifier_checked": True,
            "note": "Frozen base weights are not stored in a LoRA adapter; loaded from original base revision."}
