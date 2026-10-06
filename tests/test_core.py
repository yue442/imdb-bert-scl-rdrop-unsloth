import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
import torch
from losses import SupConLoss, symmetric_kl
from data_utils import BalancedBatchSampler, shared_split


class CoreTests(unittest.TestCase):
    def test_finite_loss_and_gradients(self):
        for views in (1, 2):
            x = torch.randn(8, views, 16, requires_grad=True)
            loss = SupConLoss()(x, torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]))
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreater(x.grad.abs().sum().item(), 0)

    def test_no_positive_anchors(self):
        for n in (1, 2, 5):
            x = torch.randn(n, 1, 16, requires_grad=True)
            loss = SupConLoss()(x, torch.arange(n))
            self.assertEqual(loss.item(), 0)
            loss.backward()
            self.assertTrue(torch.isfinite(x.grad).all())

    def test_autocast_similarity_stays_float32(self):
        x = torch.randn(8, 1, 16, requires_grad=True)
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        expected = SupConLoss()(x, labels)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            actual = SupConLoss()(x, labels)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected)

    def test_scl_reference_formula(self):
        x = torch.randn(8, 1, 16)
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        z = torch.nn.functional.normalize(x[:, 0], dim=-1)
        logits = z @ z.T / .07
        terms = []
        for i in range(8):
            denominator = torch.logsumexp(torch.cat([logits[i, :i], logits[i, i+1:]]), 0)
            positives = [j for j in range(8) if j != i and labels[j] == labels[i]]
            terms.append(torch.stack([-logits[i, j] + denominator for j in positives]).mean())
        torch.testing.assert_close(SupConLoss()(x, labels), torch.stack(terms).mean())

    def test_bad_shape_and_kl(self):
        with self.assertRaises(ValueError):
            SupConLoss()(torch.randn(8, 16), torch.zeros(8))
        a, b = torch.randn(8, 2, requires_grad=True), torch.randn(8, 2, requires_grad=True)
        torch.testing.assert_close(symmetric_kl(a, a), torch.tensor(0.), atol=1e-6, rtol=0)
        loss = symmetric_kl(a, b)
        self.assertGreaterEqual(loss.item(), -1e-6)
        loss.backward()
        self.assertTrue(torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all())

    def test_balanced_sampling(self):
        labels = [0]*32 + [1]*32
        sampler = BalancedBatchSampler(labels, 16)
        first = list(sampler)
        self.assertEqual(first, list(BalancedBatchSampler(labels, 16)))
        self.assertEqual(len(set(i for batch in first for i in batch)), 64)
        for batch in first:
            self.assertEqual(sum(labels[i] for i in batch), 8)
        sampler.set_epoch(1)
        self.assertNotEqual(first, list(sampler))

    def test_shared_split(self):
        df = pd.DataFrame({"id": range(40), "review": ["review"]*40, "sentiment": [0, 1]*20})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.json"
            tr, va, sha = shared_split(df, path)
            tr2, va2, sha2 = shared_split(df, path)
            self.assertEqual((len(tr), len(va)), (32, 8))
            self.assertFalse(set(tr.id) & set(va.id))
            self.assertEqual(sha, sha2)
            pd.testing.assert_frame_equal(tr, tr2)
            with self.assertRaises(ValueError):
                shared_split(df, path, seed=99)

    def test_tiny_bert_trainer_and_lora_reload(self):
        from transformers import BertConfig, BertForSequenceClassification, TrainingArguments
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
        from research_trainer import ResearchCallback, ResearchTrainer
        torch.set_num_threads(2)
        cfg = BertConfig(vocab_size=32, hidden_size=16, num_hidden_layers=1,
                         num_attention_heads=2, intermediate_size=32, num_labels=2)

        class Synthetic(torch.utils.data.Dataset):
            labels = [0, 1]*8
            def __len__(self):
                return len(self.labels)
            def __getitem__(self, i):
                return {"input_ids": torch.tensor([1, 2+i%10, 3, 4]),
                        "attention_mask": torch.ones(4, dtype=torch.long),
                        "labels": torch.tensor(self.labels[i])}

        def collate(rows):
            return {k: torch.stack([r[k] for r in rows]) for k in rows[0]}

        for method in ("ce", "scl", "rdrop"):
            with tempfile.TemporaryDirectory() as tmp:
                callback = ResearchCallback(tmp)
                model = BertForSequenceClassification(cfg)
                args = TrainingArguments(output_dir=tmp, max_steps=2,
                    per_device_train_batch_size=8, per_device_eval_batch_size=8,
                    eval_strategy="epoch", save_strategy="no", report_to="none",
                    use_cpu=True, disable_tqdm=True, label_names=["labels"])
                trainer = ResearchTrainer(model=model, args=args, train_dataset=Synthetic(),
                    eval_dataset=Synthetic(), data_collator=collate, callbacks=[callback],
                    method=method, compute_metrics=lambda p: {"accuracy": float((p.predictions.argmax(-1)==p.label_ids).mean())})
                callback.trainer = trainer
                trainer.train()
                evaluation = trainer.evaluate()
                self.assertTrue(np.isfinite(evaluation["eval_loss"]))
                self.assertEqual(len(callback.rows), 1)
                self.assertEqual(len(trainer.components), 2)
                # Evaluation loss is ordinary one-pass CE for every method.
                model.eval()
                batch = collate([Synthetic()[i] for i in range(8)])
                with torch.no_grad():
                    loss = trainer.compute_loss(model, batch)
                    expected = torch.nn.functional.cross_entropy(model(
                        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits, batch["labels"])
                torch.testing.assert_close(loss, expected)
        with tempfile.TemporaryDirectory() as tmp:
            base = BertForSequenceClassification(cfg)
            base.save_pretrained(Path(tmp) / "base")
            model = get_peft_model(base, LoraConfig(task_type=TaskType.SEQ_CLS,
                r=2, lora_alpha=4, target_modules=["query", "value"], modules_to_save=["classifier"]))
            model.eval()
            inputs = {"input_ids": torch.tensor([[1, 2, 3, 4]])}
            model.save_pretrained(Path(tmp) / "adapter")
            reloaded = PeftModel.from_pretrained(BertForSequenceClassification.from_pretrained(
                Path(tmp) / "base"), Path(tmp) / "adapter").eval()
            with torch.no_grad():
                torch.testing.assert_close(model(**inputs).logits, reloaded(**inputs).logits)


if __name__ == "__main__":
    unittest.main()
