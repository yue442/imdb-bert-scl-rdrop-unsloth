"""Load a saved classifier using the same model path as training."""
import json
from importlib import metadata
from pathlib import Path


def load_classifier(run_dir, backend):
    # predict.py calls this before importing torch or transformers.
    if backend == 'unsloth':
        import os
        os.environ['UNSLOTH_COMPILE_DISABLE'] = '1'
        os.environ['UNSLOTH_DISABLE_FAST_GENERATION'] = '1'
        from unsloth import FastModel
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed
    from adapter_restore import restore_adapter_exact
    from checkpoint_utils import load_bert_checkpoint_into

    run_dir = Path(run_dir)
    summary = json.loads((run_dir / 'summary.json').read_text(encoding='utf-8'))
    config = summary['config']
    if backend != config['backend']:
        raise ValueError('Use the same backend as the saved training run')
    if backend == 'unsloth':
        for package in ['torch', 'transformers', 'peft', 'unsloth', 'unsloth_zoo', 'accelerate']:
            if metadata.version(package) != summary['versions'][package]:
                raise ValueError(f'Use original Unsloth environment: {package}')
    directory = run_dir / 'best_model'
    tokenizer = AutoTokenizer.from_pretrained(directory)
    set_seed(config['seed'])
    revision = summary.get('base_revision') or config.get('revision')
    if not config['lora']:
        from transformers import AutoConfig
        model = AutoModelForSequenceClassification.from_config(AutoConfig.from_pretrained(directory))
        report = load_bert_checkpoint_into(model, directory)
    else:
        from peft import LoraConfig, get_peft_model
        adapter_config = LoraConfig.from_pretrained(directory)
        if backend == 'unsloth':
            if (adapter_config.r != 16 or adapter_config.lora_alpha != 32 or adapter_config.lora_dropout != 0
                    or set(adapter_config.target_modules) != {'query', 'value'}
                    or 'classifier' not in set(adapter_config.modules_to_save or [])):
                raise ValueError('Saved adapter differs from this training configuration')
            model, _ = FastModel.from_pretrained(
                model_name=config['model'], revision=revision,
                auto_model=AutoModelForSequenceClassification, num_labels=2,
                max_seq_length=config['max_length'], dtype=torch.float32,
                load_in_4bit=False, full_finetuning=False)
            model = FastModel.get_peft_model(
                model, r=16, lora_alpha=32, lora_dropout=0., bias='none',
                target_modules=['query', 'value'], modules_to_save=['classifier'], task_type='SEQ_CLS',
                random_state=config['seed'], use_gradient_checkpointing=config['gradient_checkpointing'])
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                config['model'], revision=revision, num_labels=2, torch_dtype=torch.float32)
            model = get_peft_model(model, adapter_config)
        for parameter in model.parameters():
            if parameter.requires_grad and parameter.dtype != torch.float32:
                parameter.data = parameter.data.float()
        report = restore_adapter_exact(model, directory / 'adapter_model.safetensors')
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = False
    if sorted({str(p.dtype) for p in model.parameters()}) != summary['parameter_dtypes']:
        raise ValueError('Loaded parameter dtypes differ from saved run')
    if sum(p.numel() for p in model.parameters()) != summary['total_parameters']:
        raise ValueError('Loaded parameter count differs from saved run')
    model.to('cuda' if torch.cuda.is_available() else 'cpu').eval()
    return model, tokenizer, summary, report
