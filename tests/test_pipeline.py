"""Offline tiny-model tests of the actual entry points, not formal experiments."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

class PipelineTests(unittest.TestCase):
    def test_train_reload_predict_all_standard_methods(self):
        import pandas as pd
        import torch
        from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
        import train
        import predict
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / 'tiny_bert'
            base.mkdir()
            words = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'good', 'bad', 'film', 'a']
            (base / 'vocab.txt').write_text('\n'.join(words), encoding='utf-8')
            tokenizer = BertTokenizer.from_pretrained(base, do_lower_case=True)
            tokenizer.save_pretrained(base)
            config = BertConfig(vocab_size=len(words), hidden_size=16, num_hidden_layers=1,
                                num_attention_heads=2, intermediate_size=32,
                                max_position_embeddings=64, num_labels=2)
            BertForSequenceClassification(config).save_pretrained(base)
            data = pd.DataFrame({'id': [f'review_{i}' for i in range(40)],
                                 'review': ['a good film' if i % 2 else 'a bad film' for i in range(40)],
                                 'sentiment': [i % 2 for i in range(40)]})
            train_file, test_file, sample_file = root / 'train.tsv', root / 'test.tsv', root / 'sample.csv'
            data.to_csv(train_file, sep='\t', index=False)
            data.iloc[:4].drop(columns='sentiment').to_csv(test_file, sep='\t', index=False)
            data.iloc[:4][['id', 'sentiment']].to_csv(sample_file, index=False)
            split = root / 'split.json'
            hashes = []
            for name, method, lora in [('bert', 'ce', False), ('bert_scl', 'scl', False),
                                       ('bert_rdrop', 'rdrop', False), ('bert_lora', 'ce', True)]:
                run = root / name
                argv = ['train.py', '--train-file', str(train_file), '--test-file', str(test_file),
                        '--sample-file', str(sample_file), '--split-file', str(split), '--model', str(base),
                        '--method', method, '--allow-cpu', '--precision', 'fp32', '--scope', 'smoke',
                        '--epochs', '1', '--max-steps', '2', '--batch-size', '8', '--max-length', '32',
                        '--output-dir', str(run), '--skip-representations'] + (['--lora'] if lora else [])
                with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                    train.main()
                summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
                self.assertEqual(summary['status'], 'completed')
                self.assertEqual(summary['submission']['rows'], 4)
                hashes.append(summary['split_sha256'])
                output = root / f'{name}_pred.csv'
                argv = ['predict.py', '--run-dir', str(run), '--test-file', str(test_file),
                        '--sample-file', str(sample_file), '--output', str(output),
                        '--train-file', str(train_file), '--split-file', str(split)]
                with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                    predict.main()
                audit = json.loads(Path(str(output) + '.audit.json').read_text())
                self.assertTrue(audit['passed'])
                self.assertLess(audit['max_probability_difference'], 1e-6)
                self.assertEqual(audit['prediction_agreement'], 1.)
                self.assertEqual(output.read_text(), (run / 'submission.csv').read_text())
            self.assertEqual(len(set(hashes)), 1)

if __name__ == '__main__':
    unittest.main()
