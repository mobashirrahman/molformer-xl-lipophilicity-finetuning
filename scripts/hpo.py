"""Stage A: distributed hyperparameter search, selected on validation.

The original project chose its "best" configuration by test-set performance,
so that configuration cannot be trusted and neither can the metrics reported
for it. Here every trial is scored on the validation split only; the test split
is never read during the search.

All 13 hosts join one study through an Optuna JournalFileStorage on the shared
NFS home. JournalFileStorage is used rather than SQLite because SQLite's
locking is unreliable over NFS and would corrupt the study.
"""
import argparse
import logging
import os
import sys

import optuna
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nnti.data import (  # noqa: E402
    SmilesRegressionDataset,
    make_collate_fn,
    stratified_three_way_split,
)
from nnti.models import build_regressor, load_checkpoint  # noqa: E402
from nnti.peft import METHODS, apply_lora  # noqa: E402
from nnti.train import evaluate, set_seed, train  # noqa: E402
from scripts.experiment import _lipo_dataframe, baseline_checkpoint_path  # noqa: E402

# BitFit trains only biases and needs a far larger step than methods that add
# capacity, so each method gets its own range rather than one shared prior.
# LoRA's range was originally [1e-6, 5e-4] and its optimum came back at
# 4.15e-4 -- 97% of the way up the log range, i.e. pinned to the ceiling. A
# search that terminates at a boundary has not found the optimum, only the edge
# of where it was allowed to look, so that result bounded LoRA's performance
# rather than measuring it. Widened to [1e-5, 1e-2] and re-run.
PEFT_LR_RANGE = {
    "bitfit": (1e-4, 2e-2),
    "lora":   (1e-5, 1e-2),
    "ia3":    (1e-5, 5e-3),
    "full":   (5e-6, 3e-4),
}

logger = logging.getLogger("hpo")

EVAL_BATCH = 128


def objective(trial, device, split_seeds, epochs_cap):
    params = {
        "lr": trial.suggest_float("lr", 5e-6, 3e-4, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "patience": trial.suggest_int("patience", 2, 5),
    }

    # Average over a couple of splits: stage B showed a single run carries
    # +/-16% noise, so scoring a trial on one run would mostly rank noise.
    scores = []
    for split_seed in split_seeds:
        set_seed(split_seed)
        tokenizer, model = build_regressor(dropout_rate=params["dropout"])
        train_df, val_df, _ = stratified_three_way_split(_lipo_dataframe(), split_seed)
        coll = make_collate_fn(tokenizer)

        def loader(df, bs, shuffle):
            return DataLoader(
                SmilesRegressionDataset(df["SMILES"].tolist(), df["label"].tolist()),
                batch_size=bs, shuffle=shuffle, collate_fn=coll)

        model, info = train(
            model,
            loader(train_df, params["batch_size"], True),
            loader(val_df, EVAL_BATCH, False),
            device,
            epochs=epochs_cap,
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            patience=params["patience"],
        )
        scores.append(info["best_val_mse"])
        del model
        torch.cuda.empty_cache()

        trial.report(sum(scores) / len(scores), len(scores))
        if trial.should_prune():
            raise optuna.TrialPruned()

    return sum(scores) / len(scores)


def peft_objective(trial, device, method, artifact_dir, split_seeds, epochs_cap):
    """Tune one PEFT method on validation, starting from the stage B checkpoint.

    Stage D compares methods against each other, so each needs a learning rate
    chosen on equal terms; carrying over the original project's per-method
    values would confound "which method is better" with "whose learning rate
    happened to suit it".
    """
    lo, hi = PEFT_LR_RANGE[method]
    lr = trial.suggest_float("lr", lo, hi, log=True)
    r = trial.suggest_categorical("lora_r", [4, 8, 16]) if method == "lora" else None

    scores = []
    for split_seed in split_seeds:
        set_seed(split_seed)
        tokenizer, model = build_regressor()
        load_checkpoint(model, baseline_checkpoint_path(artifact_dir, split_seed),
                        map_location=device)
        if method == "lora":
            apply_lora(model, r=r)
        else:
            METHODS[method](model)

        train_df, val_df, _ = stratified_three_way_split(_lipo_dataframe(), split_seed)
        coll = make_collate_fn(tokenizer)

        def loader(df, bs, shuffle):
            return DataLoader(
                SmilesRegressionDataset(df["SMILES"].tolist(), df["label"].tolist()),
                batch_size=bs, shuffle=shuffle, collate_fn=coll)

        model, info = train(model, loader(train_df, 32, True), loader(val_df, EVAL_BATCH, False),
                            device, epochs=epochs_cap, lr=lr, patience=3)
        scores.append(info["best_val_mse"])
        del model
        torch.cuda.empty_cache()
        trial.report(sum(scores) / len(scores), len(scores))
        if trial.should_prune():
            raise optuna.TrialPruned()
    return sum(scores) / len(scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--storage", default="experiments/hpo/journal.log")
    ap.add_argument("--study", default="stageA-baseline")
    ap.add_argument("--trials", type=int, default=25, help="trials THIS worker runs")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--splits", default="0,1")
    ap.add_argument("--method", default=None,
                    help="PEFT method to tune (bitfit|lora|ia3|full); omit for the baseline study")
    ap.add_argument("--artifacts", default="/scratch/mdra00001/nnti-artifacts")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(name)s - %(message)s")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    os.makedirs(os.path.dirname(args.storage), exist_ok=True)

    # NFS-safe: SQLite locking over NFS is unreliable and would corrupt the study.
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(args.storage)
    )
    study = optuna.create_study(
        study_name=args.study, storage=storage, direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=None),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_seeds = [int(s) for s in args.splits.split(",")]
    logger.info("joining study %s (%d trials done); running %d more",
                args.study, len(study.trials), args.trials)

    if args.method:
        fn = lambda t: peft_objective(t, device, args.method, args.artifacts,
                                      split_seeds, args.epochs)
    else:
        fn = lambda t: objective(t, device, split_seeds, args.epochs)
    study.optimize(fn, n_trials=args.trials,
                   catch=(RuntimeError, torch.OutOfMemoryError))

    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if done:
        logger.info("best so far: val_mse=%.4f params=%s",
                    study.best_value, study.best_params)
        # Namespaced by study: the baseline and the four PEFT studies share a
        # journal, and a fixed filename would let whichever finished last
        # silently overwrite the others.
        out = os.path.join(os.path.dirname(args.storage), f"best_{args.study}.json")
        import json
        with open(out, "w") as fh:
            json.dump({"value": study.best_value, "params": study.best_params,
                       "n_complete": len(done)}, fh, indent=2)


if __name__ == "__main__":
    main()
