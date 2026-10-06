"""Classification inference without Trainer's globally patched prediction_step."""
import torch


def classification_logits(model, dataset, collator, batch_size, precision, progress=False):
    device = next(model.parameters()).device
    amp = device.type == "cuda" and precision in ("bf16", "fp16")
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    logits = []
    model.eval()
    # Some patched modules enable grad internally. inference_mode prevents an
    # accidental retained graph even then. Scope AMP per batch; do not retain a
    # dataset-long cast cache. Store only detached CPU logits.
    with torch.inference_mode():
        for index, batch in enumerate(loader, 1):
            inputs = {key: value.to(device) for key, value in batch.items() if key != "labels"}
            with torch.autocast(device.type, dtype=dtype, enabled=amp, cache_enabled=False):
                result = model(**inputs, return_dict=True, output_hidden_states=False, output_attentions=False)
            output = result.logits.detach().float().cpu()
            del result, inputs, batch
            if output.ndim != 2 or output.shape[1] != 2 or not torch.isfinite(output).all():
                raise RuntimeError("Expected finite binary classification logits")
            logits.append(output)
            if progress and (index == 1 or index % 100 == 0 or index == len(loader)):
                memory = (f" GPU allocated={torch.cuda.memory_allocated(device)/1024**3:.2f}GiB"
                          if device.type == "cuda" else "")
                print(f"Inference batches {index}/{len(loader)}{memory}", flush=True)
    if not logits:
        raise ValueError("Empty prediction dataset")
    return torch.cat(logits)
