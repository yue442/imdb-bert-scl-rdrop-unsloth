"""Single-process experiments. Run --help; README contains the complete runbook."""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from importlib import metadata
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-file", required=True)
    p.add_argument("--test-file")
    p.add_argument("--sample-file", help="Official sampleSubmission.csv for ID/order validation")
    p.add_argument("--split-file", default="data/split_seed42.json")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", default="bert-base-uncased")
    p.add_argument("--revision", help="Pinned Hugging Face base-model commit")
    p.add_argument("--backend", choices=["standard", "unsloth"], default="standard")
    p.add_argument("--method", choices=["ce", "scl", "rdrop"], default="ce")
    p.add_argument("--lora", action="store_true")
    p.add_argument("--alpha", type=float, default=.2)
    p.add_argument("--beta", type=float, default=1.)
    p.add_argument("--temperature", type=float, default=.07)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--accumulation", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--limit-train", type=int, default=0)
    p.add_argument("--limit-val", type=int, default=0)
    p.add_argument("--scope", choices=["formal", "smoke", "benchmark"], default="formal")
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--skip-representations", action="store_true")
    return p.parse_args()


def dump(path, record):
    Path(path).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def run(args, out, summary):
    # Import Unsloth before transformers, torch and our Trainer subclass.
    fast_model = None
    if args.backend == "unsloth":
        os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"
        from unsloth import FastModel
        fast_model = FastModel
        if not args.lora:
            raise ValueError("This Unsloth stage explicitly tests LoRA; pass --lora")
    import hashlib
    import math
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, TrainingArguments, set_seed)
    from data_utils import TokenDataset, read_data, shared_split, subset
    from research_trainer import ResearchTrainer, ResearchCallback

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Use ordinary python on a single GPU, not torchrun")
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA unavailable: check AutoDL GPU image; CPU allowed only with --allow-cpu")
    if args.batch_size < 4 or args.batch_size % 2:
        raise ValueError("Use even batch size >=4 so SCL has same-class positives")
    if min(args.epochs, args.accumulation, args.max_length, args.learning_rate, args.temperature) <= 0:
        raise ValueError("epochs/accumulation/length/lr/temperature must be positive")
    if args.alpha < 0 or args.beta < 0:
        raise ValueError("loss weights cannot be negative")
    if args.scope == "formal" and (args.max_steps > 0 or args.limit_train or args.limit_val):
        raise ValueError("Limited data/steps must be marked smoke or benchmark, not formal")
    precision = args.precision
    if precision == "auto":
        precision = "bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
            "fp16" if torch.cuda.is_available() else "fp32")
    if precision == "bf16" and not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
        raise ValueError("GPU does not support BF16")
    set_seed(args.seed)
    raw = read_data(args.train_file)
    tr, va, split_hash = shared_split(raw, args.split_file, args.seed)
    tr, va = subset(tr, args.limit_train, args.seed), subset(va, args.limit_val, args.seed)
    selected_hash = hashlib.sha256(json.dumps(
        [tr.id.tolist(), va.id.tolist()], ensure_ascii=False).encode()).hexdigest()
    summary.update({"split_sha256": split_hash, "selected_ids_sha256": selected_hash,
                    "train_count": len(tr), "val_count": len(va), "precision": precision,
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                    "effective_batch_size": args.batch_size * args.accumulation,
                    "contrast_candidates_per_forward": args.batch_size,
                    "python": platform.python_version()})
    summary["versions"] = {}
    summary["runtime_flags"] = {key: os.environ.get(key, "0") for key in (
        "UNSLOTH_COMPILE_DISABLE", "TORCHDYNAMO_DISABLE", "TORCH_COMPILE_DISABLE")}
    for package in ["torch", "transformers", "accelerate", "peft", "unsloth", "unsloth_zoo",
                    "numpy", "pandas", "scikit-learn"]:
        try:
            summary["versions"][package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], text=True,
                            capture_output=True, check=True)
    (out / "environment.txt").write_text(freeze.stdout, encoding="utf-8")
    dump(out / "summary.json", summary)

    if fast_model:
        model, tokenizer = fast_model.from_pretrained(
            model_name=args.model, auto_model=AutoModelForSequenceClassification,
            num_labels=2, max_seq_length=args.max_length,
            dtype=torch.float32, load_in_4bit=False, full_finetuning=False,
            **({"revision": args.revision} if args.revision else {}),
        )
        model = fast_model.get_peft_model(
            model, r=16, lora_alpha=32, lora_dropout=0., bias="none",
            target_modules=["query", "value"], modules_to_save=["classifier"],
            task_type="SEQ_CLS", random_state=args.seed,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
        # Keep trainable weights FP32. Mixed precision is Trainer autocast, not .half().
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=2, torch_dtype=torch.float32,
            revision=args.revision,
        )
        if args.lora:
            from peft import LoraConfig, TaskType, get_peft_model
            model = get_peft_model(model, LoraConfig(
                task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=0.,
                target_modules=["query", "value"], modules_to_save=["classifier"], bias="none",
            ))
    trainable_names = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Optimizer parameters stay FP32; autocast controls activation precision.
            if param.dtype != torch.float32:
                param.data = param.data.float()
            trainable_names.append(name)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.lora and (not any("lora_" in n for n in trainable_names) or trainable >= total / 2):
        raise RuntimeError("LoRA was not applied correctly (or full fine-tuning unexpectedly enabled)")
    (out / "trainable_parameters.txt").write_text("\n".join(trainable_names), encoding="utf-8")
    summary.update({"total_parameters": total, "trainable_parameters": trainable,
                    "base_revision": getattr(model.config, "_commit_hash", None),
                    "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
                    "trainable_dtypes": sorted({str(p.dtype) for p in model.parameters() if p.requires_grad}),
                    "model_class": f"{type(model).__module__}.{type(model).__name__}",
                    "lora_config": {"r": 16, "lora_alpha": 32, "lora_dropout": 0.,
                        "target_modules": ["query", "value"], "modules_to_save": ["classifier"]}
                        if args.lora else None})
    if args.gradient_checkpointing and args.lora:
        model.enable_input_require_grads()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    train_dataset = TokenDataset(tr, tokenizer, args.max_length)
    val_dataset = TokenDataset(va, tokenizer, args.max_length)
    summary["tokenized_inputs_sha256"] = hashlib.sha256(json.dumps(
        [dict(train_dataset.tokens), dict(val_dataset.tokens)], sort_keys=True).encode()).hexdigest()
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    def metrics(pred):
        logits, labels = pred
        probs = torch.tensor(logits).float().softmax(-1).numpy()[:, 1]
        predictions = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, predictions),
                "f1": f1_score(labels, predictions), "auc": roc_auc_score(labels, probs)}

    batches = min(sum(np.array(train_dataset.labels) == c) for c in (0, 1)) // (args.batch_size // 2)
    total_steps = args.max_steps if args.max_steps > 0 else math.ceil(batches / args.accumulation) * args.epochs
    training_args = TrainingArguments(
        output_dir=str(out / "checkpoints"), per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size, gradient_accumulation_steps=args.accumulation,
        learning_rate=args.learning_rate, weight_decay=.01, warmup_steps=int(total_steps * .1),
        lr_scheduler_type="linear", optim="adamw_torch", num_train_epochs=args.epochs,
        max_steps=args.max_steps, bf16=precision == "bf16", fp16=precision == "fp16",
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="accuracy", greater_is_better=True, save_total_limit=1,
        logging_steps=25, report_to="none", seed=args.seed, data_seed=args.seed,
        remove_unused_columns=False, dataloader_num_workers=0, label_names=["labels"],
        dataloader_pin_memory=torch.cuda.is_available(),
    )
    cb = ResearchCallback(out)
    trainer = ResearchTrainer(
        model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset,
        data_collator=collator, processing_class=tokenizer, compute_metrics=metrics,
        callbacks=[cb], method=args.method, alpha=args.alpha, beta=args.beta,
        temperature=args.temperature,
    )
    cb.trainer = trainer
    dump(out / "training_arguments.json", training_args.to_dict())
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    trainer.train()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    summary["training_seconds_including_eval_save"] = time.perf_counter() - started
    summary["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else None
    summary["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() else None
    summary["steady_window_seconds"] = cb.window_seconds
    summary["steady_window_updates"] = cb.window_steps
    summary["best_checkpoint"] = trainer.state.best_model_checkpoint
    summary["completed_steps"] = trainer.state.global_step
    summary["epoch_metrics"] = cb.rows
    summary["best_epoch"] = max(cb.rows, key=lambda row: row["eval_accuracy"])["epoch"] if cb.rows else None
    summary["final_epoch_metrics"] = cb.rows[-1] if cb.rows else None
    summary["best_validation"] = trainer.evaluate()  # every method: CE, one deterministic forward
    trainer.save_model(str(out / "best_model"))
    tokenizer.save_pretrained(out / "best_model")
    dump(out / "trainer_log.json", trainer.state.log_history)
    prediction = trainer.predict(val_dataset)
    logits = prediction.predictions
    probabilities = torch.tensor(logits).float().softmax(-1).numpy()[:, 1]
    val_rows = va[["id", "sentiment"]].copy()
    val_rows["prediction"] = np.argmax(logits, axis=-1)
    val_rows["positive_probability"] = probabilities
    val_rows.to_csv(out / "validation_predictions.csv", index=False)
    if not args.skip_representations:
        # Fixed subset for every method; best model, no dropout. Exact pairwise
        # cosine means exclude self pairs, not a proxy for downstream quality.
        chosen = subset(va, min(1000, len(va)), args.seed)
        representations = TokenDataset(chosen, tokenizer, args.max_length)
        device = next(trainer.model.parameters()).device
        vectors = []
        trainer.model.eval()
        with torch.no_grad():
            for batch in torch.utils.data.DataLoader(representations, batch_size=args.batch_size, collate_fn=collator):
                batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                output = trainer.model(**batch, output_hidden_states=True, return_dict=True)
                vectors.append(output.hidden_states[-1][:, 0, :].float().cpu())
        z = torch.nn.functional.normalize(torch.cat(vectors), dim=-1)
        sim = z @ z.T
        labels = torch.tensor(chosen.sentiment.to_numpy())
        same = labels[:, None].eq(labels[None, :])
        same.fill_diagonal_(False)
        different = labels[:, None].ne(labels[None, :])
        summary["representations"] = {"n": len(chosen), "within_class_cosine": sim[same].mean().item(),
                                       "between_class_cosine": sim[different].mean().item()}
        np.save(out / "validation_cls.npy", z.numpy())
        chosen[["id", "sentiment"]].to_csv(out / "representation_ids.csv", index=False)
    if args.test_file:
        test = read_data(args.test_file, labeled=False)
        from inference_utils import classification_logits
        # Unsloth globally patches Trainer.prediction_step and assumes "prompt"
        # for unlabelled inputs. Tokenized classification data has no such field.
        test_logits = classification_logits(trainer.model,
            TokenDataset(test, tokenizer, args.max_length), collator, args.batch_size, precision)
        # Competition metric is ROC-AUC: submit positive probabilities, not argmax.
        from submission_io import write_submission
        summary["submission"] = write_submission(
            out / "submission.csv", test.id.tolist(),
            test_logits.softmax(-1).numpy()[:, 1].tolist(), args.sample_file)
    summary["status"] = "completed"
    dump(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    args = parse_args()
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output: {out}. Use a NEW run directory.")
    out.mkdir(parents=True, exist_ok=True)
    summary = {"status": "running", "config": vars(args),
               "note": "single seed; no statistical significance claim; validation loss is CE"}
    dump(out / "config.json", vars(args))
    dump(out / "summary.json", summary)
    try:
        run(args, out, summary)
    except Exception as error:
        summary.update({"status": "failed", "error": repr(error)})
        dump(out / "summary.json", summary)
        (out / "failure_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
