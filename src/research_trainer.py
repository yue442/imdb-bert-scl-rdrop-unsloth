import csv
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import Trainer, TrainerCallback

from data_utils import BalancedBatchSampler
from losses import SupConLoss, symmetric_kl


class ResearchTrainer(Trainer):
    """Same sampler and CE evaluation for CE / SCL / R-Drop / LoRA."""
    def __init__(self, *args, method="ce", alpha=0.2, beta=1., temperature=.07, **kwargs):
        super().__init__(*args, **kwargs)
        self.method, self.alpha, self.beta = method, alpha, beta
        self.scl = SupConLoss(temperature)
        self.model_accepts_loss_kwargs = False  # Trainer handles gradient accumulation.
        self.components = []
        self.epoch_components = []
        self.batch_sampler = None

    def _load_best_model(self):
        # Transformers 5.12 can save legacy gamma/beta names but restore them
        # with bare load_state_dict, skipping LayerNorm. Fail closed instead.
        if self.model.config.model_type == "bert" and not hasattr(self.model, "peft_config"):
            from checkpoint_utils import load_bert_checkpoint_into
            self.best_checkpoint_load_report = load_bert_checkpoint_into(
                self.model, self.state.best_model_checkpoint
            )
        else:
            super()._load_best_model()

    def get_train_dataloader(self):
        if self.args.world_size != 1:
            raise ValueError("This package uses one GPU/process; do not use torchrun")
        self.batch_sampler = BalancedBatchSampler(
            self.train_dataset.labels, self.args.per_device_train_batch_size, self.args.seed
        )
        loader = DataLoader(self.train_dataset, batch_sampler=self.batch_sampler,
                            collate_fn=self.data_collator, num_workers=0,
                            pin_memory=torch.cuda.is_available())
        return self.accelerator.prepare(loader)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        x = {k: v for k, v in inputs.items() if k != "labels"}
        training = model.training
        with_scl = training and self.method == "scl" and self.alpha != 0
        output = model(**x, output_hidden_states=with_scl, return_dict=True)
        logits = output.logits
        ce = F.cross_entropy(logits.float(), labels)
        extra = ce.new_zeros(())
        if training and self.method == "rdrop":
            second = model(**x, return_dict=True).logits
            ce = (ce + F.cross_entropy(second.float(), labels)) / 2
            extra = symmetric_kl(logits, second)
            loss = ce + self.beta * extra
        elif with_scl:
            cls = output.hidden_states[-1][:, 0, :]
            extra = self.scl(cls.unsqueeze(1), labels)
            loss = ce + self.alpha * extra
        else:
            loss = ce
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite {self.method} loss")
        if training:
            item = {"step": self.state.global_step, "epoch": self.state.epoch,
                    "samples": len(labels), "ce_loss": ce.detach().item(),
                    "scl_loss": extra.detach().item() if with_scl else 0.,
                    "kl_loss": extra.detach().item() if self.method == "rdrop" else 0.,
                    "total_loss": loss.detach().item()}
            self.components.append(item)
            self.epoch_components.append(item)
        # Do not return hidden states to Trainer evaluation/prediction collection.
        clean_output = {"loss": loss, "logits": logits}
        return (loss, clean_output) if return_outputs else loss


def write_rows(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class ResearchCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.path = Path(output_dir)
        self.trainer = None
        self.rows = []
        self.window_start = None
        self.window_seconds = None
        self.window_steps = 0

    def on_epoch_begin(self, args, state, control, **kwargs):
        if self.trainer.batch_sampler is not None:
            self.trainer.batch_sampler.set_epoch(int(state.epoch or 0))

    def on_step_begin(self, args, state, control, **kwargs):
        # Fixed 20-update warmup, measure next 100 updates. For short runs this
        # avoids first-step initialization; epoch evaluation is outside window.
        if state.global_step == 20:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.window_start = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 120 and self.window_start is not None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.window_seconds = time.perf_counter() - self.window_start
            self.window_steps = 100

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        samples = sum(x["samples"] for x in self.trainer.epoch_components)
        if not samples:  # final best-checkpoint evaluation is not a new epoch
            return
        row = {"epoch": state.epoch, "step": state.global_step}
        for key in ("ce_loss", "scl_loss", "kl_loss", "total_loss"):
            row[key] = sum(x[key] * x["samples"] for x in self.trainer.epoch_components) / samples
        row.update({k: v for k, v in (metrics or {}).items() if k.startswith("eval_")})
        self.rows.append(row)
        self.trainer.epoch_components.clear()
        write_rows(self.path / "epoch_metrics.csv", self.rows)
        write_rows(self.path / "train_components.csv", self.trainer.components)

    def on_train_end(self, args, state, control, **kwargs):
        write_rows(self.path / "train_components.csv", self.trainer.components)
