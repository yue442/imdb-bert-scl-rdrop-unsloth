"""Regression tests for legacy BERT save/restore and collision safety."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from safetensors.torch import save_file
from transformers import BertConfig, BertForSequenceClassification, TrainingArguments
from checkpoint_utils import load_bert_checkpoint_into, normalized_bert_weights
from research_trainer import ResearchTrainer


class CheckpointTests(unittest.TestCase):
    def test_strict_legacy_checkpoint_and_trainer_restore(self):
        cfg = BertConfig(vocab_size=20, hidden_size=8, num_hidden_layers=1,
                         num_attention_heads=2, intermediate_size=16, num_labels=2)
        original = BertForSequenceClassification(cfg)
        original.eval()
        legacy = {k.replace("LayerNorm.weight", "LayerNorm.gamma").replace(
            "LayerNorm.bias", "LayerNorm.beta"): v.clone()
            for k, v in original.state_dict().items()}
        with tempfile.TemporaryDirectory() as tmp:
            save_file(legacy, str(Path(tmp) / "model.safetensors"))
            fresh = BertForSequenceClassification(cfg)
            report = load_bert_checkpoint_into(fresh, tmp)
            self.assertEqual(report["renamed_layernorm_keys"], 6)
            args = TrainingArguments(output_dir=tmp, use_cpu=True, report_to="none")
            trainer = ResearchTrainer(model=fresh, args=args)
            trainer.state.best_model_checkpoint = tmp
            with torch.no_grad():
                for p in fresh.parameters():
                    p.add_(1)
            trainer._load_best_model()
            for key, value in fresh.state_dict().items():
                torch.testing.assert_close(value, original.state_dict()[key], rtol=0, atol=0)
            broken = dict(legacy)
            broken.pop("classifier.weight")
            save_file(broken, str(Path(tmp) / "model.safetensors"))
            with self.assertRaises(RuntimeError):
                load_bert_checkpoint_into(fresh, tmp)

    def test_renaming_collision_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_file({"bert.embeddings.LayerNorm.gamma": torch.ones(8),
                       "bert.embeddings.LayerNorm.weight": torch.zeros(8)},
                      str(Path(tmp) / "model.safetensors"))
            with self.assertRaises(ValueError):
                normalized_bert_weights(tmp)


if __name__ == "__main__":
    unittest.main()
