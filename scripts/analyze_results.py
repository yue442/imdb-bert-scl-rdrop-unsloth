"""Plot saved metrics and inspect paired validation errors, without training."""
import argparse
import json
from pathlib import Path

def analyze(root, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    names = ['bert', 'bert_scl', 'bert_rdrop', 'bert_lora', 'bert_unsloth_lora']
    colors = ['#415A77', '#2A9D8F', '#B56576', '#D49A37', '#7967A4']
    records = [json.loads((root / name / 'summary.json').read_text(encoding='utf-8')) for name in names]
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for name, color, record in zip(names, colors, records):
        rows = record['epoch_metrics']
        axes[0].plot([r['epoch'] for r in rows], [r['eval_accuracy'] * 100 for r in rows],
                     '-o', color=color, label=name)
        axes[1].plot([r['epoch'] for r in rows], [r['eval_loss'] for r in rows], '-o', color=color)
    axes[0].set(ylabel='Validation accuracy (%)', xlabel='Epoch', xticks=[1, 2, 3])
    axes[1].set(ylabel='Validation CE loss', xlabel='Epoch', xticks=[1, 2, 3])
    axes[0].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=.2)
    fig.savefig(output / 'validation.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for name, color, record in zip(names, colors, records):
        rows = record['epoch_metrics']
        axes[0].plot([r['epoch'] for r in rows], [r['ce_loss'] for r in rows], '-o', color=color, label=name)
        axes[1].plot([r['epoch'] for r in rows], [r['total_loss'] for r in rows], '-o', color=color)
    axes[0].set(ylabel='Training CE component', xlabel='Epoch', xticks=[1, 2, 3])
    axes[1].set(ylabel='Training total loss (different objectives)', xlabel='Epoch', xticks=[1, 2, 3])
    axes[0].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=.2)
    fig.savefig(output / 'training.png', dpi=160)
    plt.close(fig)
    ce = pd.read_csv(root / 'bert/validation_predictions.csv').set_index('id')
    scl = pd.read_csv(root / 'bert_scl/validation_predictions.csv').set_index('id')
    if not ce.index.equals(scl.index) or not ce.sentiment.equals(scl.sentiment):
        raise ValueError('Paired analysis requires identical validation IDs/labels/order')
    ce_ok, scl_ok = ce.prediction.eq(ce.sentiment), scl.prediction.eq(scl.sentiment)
    cases = {}
    for key, mask in [('ce_wrong_scl_correct', ~ce_ok & scl_ok),
                      ('ce_correct_scl_wrong', ce_ok & ~scl_ok), ('both_wrong', ~ce_ok & ~scl_ok)]:
        selected = ce.loc[mask]
        cases[key] = {'count': int(mask.sum()), 'example_ids': selected.index[:10].tolist()}
    (output.parent / 'paired_errors.json').write_text(json.dumps(cases, indent=2), encoding='utf-8')
    print(json.dumps(cases, indent=2))
    return cases

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default='results')
    parser.add_argument('--output', default='results/figures')
    args = parser.parse_args()
    analyze(Path(args.results), Path(args.output))
