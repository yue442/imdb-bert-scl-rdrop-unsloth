"""Export a new submission without training; optionally audit full validation."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--backend', choices=['standard', 'unsloth'], default='standard')
    parser.add_argument('--test-file', required=True)
    parser.add_argument('--sample-file', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--train-file', help='Enables complete validation probability audit')
    parser.add_argument('--split-file', default='configs/split_seed42.json')
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    from model_io import load_classifier
    model, tokenizer, summary, load_report = load_classifier(args.run_dir, args.backend)
    import torch
    import pandas as pd
    from transformers import DataCollatorWithPadding
    from data_utils import read_data, shared_split, TokenDataset
    from inference_utils import classification_logits
    from submission_io import canonical_id, write_submission

    config = summary['config']
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    precision = summary['precision'] if torch.cuda.is_available() else 'fp32'
    batch_size = config['batch_size']

    def probabilities(frame):
        logits = classification_logits(model, TokenDataset(frame, tokenizer, config['max_length']),
                                       collator, batch_size, precision, progress=True)
        return logits.softmax(-1).numpy()[:, 1]

    audit = {'load': load_report, 'backend': args.backend, 'inference_batch_size': batch_size}
    if args.train_file:
        _, validation, split_hash = shared_split(read_data(args.train_file), args.split_file, config['seed'])
        if split_hash != summary['split_sha256']:
            raise ValueError('Validation split differs from saved run')
        saved = pd.read_csv(Path(args.run_dir) / 'validation_predictions.csv')
        if [canonical_id(x) for x in validation.id] != [canonical_id(x) for x in saved.id]:
            raise ValueError('Validation ID order differs')
        if validation.sentiment.tolist() != saved.sentiment.tolist():
            raise ValueError('Validation labels differ')
        current = probabilities(validation)
        delta = abs(current - saved.positive_probability.to_numpy())
        agreement = float(((current >= .5).astype(int) == saved.prediction.to_numpy()).mean())
        audit.update(samples=len(saved), max_probability_difference=float(delta.max()),
                     mean_probability_difference=float(delta.mean()), prediction_agreement=agreement,
                     tolerance=.02, passed=bool(delta.max() <= .02),
                     note='Engineering tolerance, not exact equivalence or a statistical test.')
        if not audit['passed']:
            raise ValueError(f'Loaded validation predictions differ: {audit}')
    test = read_data(args.test_file, labeled=False)
    audit['submission'] = write_submission(args.output, test.id, probabilities(test), args.sample_file)
    Path(str(args.output) + '.audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(audit, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
