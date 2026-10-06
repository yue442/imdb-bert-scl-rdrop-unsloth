import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


def read_data(path, labeled=True):
    # compression='infer' reads both the competition .tsv and .tsv.zip.
    df = pd.read_csv(path, sep="\t", quoting=3, compression="infer")
    required = {"id", "review"} | ({"sentiment"} if labeled else set())
    if not required.issubset(df.columns) or df[list(required)].isna().any().any():
        raise ValueError(f"Required non-null columns: {sorted(required)}")
    if df.id.duplicated().any():
        raise ValueError("duplicate review IDs")
    if labeled and set(df.sentiment.unique()) != {0, 1}:
        raise ValueError("sentiment must contain exactly the two labels 0 and 1")
    return df.reset_index(drop=True)


def shared_split(df, manifest_path, seed=42):
    # Hash content rather than ZIP bytes: extracted and zipped sources share a split.
    content = df[["id", "sentiment", "review"]].to_json(orient="records", force_ascii=False)
    fingerprint = hashlib.sha256(content.encode("utf-8")).hexdigest()
    path = Path(manifest_path)
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["data_sha256"] != fingerprint or record["seed"] != seed:
            raise ValueError("split manifest belongs to different data/seed; use a new path")
        train, val = record["train_indices"], record["val_indices"]
        if set(train) & set(val) or sorted(train + val) != list(range(len(df))):
            raise ValueError("invalid/overlapping split indices")
    else:
        train, val = train_test_split(
            np.arange(len(df)), test_size=0.2, random_state=seed, stratify=df.sentiment
        )
        record = {"seed": seed, "data_sha256": fingerprint,
                  "train_indices": train.tolist(), "val_indices": val.tolist()}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    split_hash = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    return df.iloc[record["train_indices"]].reset_index(drop=True), \
        df.iloc[record["val_indices"]].reset_index(drop=True), split_hash


def subset(df, size, seed):
    if not size or size >= len(df):
        return df
    selected, _ = train_test_split(
        np.arange(len(df)), train_size=size, random_state=seed, stratify=df.sentiment
    )
    return df.iloc[selected].reset_index(drop=True)


class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.tokens = tokenizer(df.review.astype(str).tolist(), truncation=True,
                                max_length=max_length, padding=False)
        self.labels = df.sentiment.astype(int).tolist() if "sentiment" in df else None

    def __len__(self):
        return len(self.tokens["input_ids"])

    def __getitem__(self, index):
        row = {k: v[index] for k, v in self.tokens.items()}
        if self.labels is not None:
            row["labels"] = self.labels[index]
        return row


class BalancedBatchSampler(torch.utils.data.Sampler):
    """Equal-class batches, no replacement. Discard only incomplete leftovers.

    On the official 20,000 balanced train rows and B=16 this drops nothing.
    Epoch-specific shuffles are deterministic and shared by every method.
    """
    def __init__(self, labels, batch_size=16, seed=42):
        if batch_size < 4 or batch_size % 2:
            raise ValueError("balanced single-view SCL requires an even batch_size >= 4")
        labels = np.asarray(labels)
        self.groups = [np.flatnonzero(labels == x) for x in (0, 1)]
        self.half, self.seed, self.epoch = batch_size // 2, seed, 0
        if len(self) == 0:
            raise ValueError("too few examples for a balanced batch")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return min(len(g) for g in self.groups) // self.half

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        groups = [rng.permutation(g) for g in self.groups]
        for start in range(0, len(self) * self.half, self.half):
            indices = np.concatenate([g[start:start + self.half] for g in groups])
            yield rng.permutation(indices).tolist()
