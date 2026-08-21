"""Dataset construction and splitting.

Two things here differ deliberately from the original scripts:

1. Three-way train/val/test splits. The original used the test set as the
   validation set for early stopping and best-checkpoint selection, then
   reported metrics on that same set, which biases every reported number.
   Here the test set is touched exactly once, at the end of a run.

2. Fixed-width padding is retained for comparability with completed experiment
   artifacts, although it is not required for correctness. An earlier check
   appeared to show padding-dependent embeddings, but the model was redrawing
   its random attention features between calls. With deterministic evaluation
   enabled, 128-wide and batch-tight padding produce identical pooled outputs.

   The maximum length remains 128. Three of 4200 molecules (0.07%) exceed this
   limit and are truncated; this is recorded as a known limitation.
"""
import logging

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Fixed padding width retained for comparability with completed runs.
MAX_TOKENS = 128


class SmilesRegressionDataset(Dataset):
    """Holds raw SMILES; tokenization happens in the collate function so that
    each batch can be padded to its own longest sequence."""

    def __init__(self, smiles_list, targets):
        if len(smiles_list) != len(targets):
            raise ValueError(
                f"smiles/targets length mismatch: {len(smiles_list)} vs {len(targets)}"
            )
        self.smiles_list = list(smiles_list)
        self.targets = list(targets)

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        return self.smiles_list[idx], self.targets[idx]


def make_collate_fn(tokenizer, max_length=MAX_TOKENS, pad_to_max=True):
    """Returns a collate_fn producing input_ids / attention_mask / labels.

    pad_to_max defaults to True to preserve comparability with completed runs.
    Setting it False enables faster batch-tight padding; both layouts produce
    identical outputs when MoLFormer is loaded with deterministic evaluation.
    """

    def collate(batch):
        smiles = [b[0] for b in batch]
        targets = [b[1] for b in batch]
        kwargs = dict(truncation=True, max_length=max_length, return_tensors="pt")
        kwargs["padding"] = "max_length" if pad_to_max else True
        encoding = tokenizer(smiles, **kwargs)
        encoding["labels"] = torch.tensor(targets, dtype=torch.float)
        return encoding

    return collate


def stratified_three_way_split(df, split_seed, test_size=0.2, val_size=0.125, num_bins=10):
    """Split into train/val/test, stratified on binned regression targets.

    val_size is a fraction of the post-test remainder, so the defaults give
    70/10/20. split_seed controls the partition only; model initialization is
    seeded separately so the two variance sources can be told apart.
    """
    work = df.copy()
    work["_bin"] = pd.qcut(work["label"], q=num_bins, duplicates="drop")

    train_val_df, test_df = train_test_split(
        work, test_size=test_size, stratify=work["_bin"], random_state=split_seed
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        stratify=train_val_df["_bin"],
        random_state=split_seed,
    )

    out = [d.drop(columns=["_bin"]).reset_index(drop=True) for d in (train_df, val_df, test_df)]
    logger.info(
        "split_seed=%d -> train=%d val=%d test=%d", split_seed, len(out[0]), len(out[1]), len(out[2])
    )
    return out


def load_external(path):
    """Load the external dataset, normalizing its 'Label' column to 'label'."""
    ext = pd.read_csv(path)
    if "Label" in ext.columns:
        ext = ext.rename(columns={"Label": "label"})
    missing = {"SMILES", "label"} - set(ext.columns)
    if missing:
        raise ValueError(f"external dataset {path} missing columns: {missing}")
    return ext
