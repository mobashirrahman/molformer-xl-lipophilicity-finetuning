"""Single-run executor. Every experiment in the study funnels through run_one.

A run is fully determined by its spec dict, so the driver can hash the spec to
get a stable run id, skip completed work on restart, and reproduce any row of
the results table from the id alone.
"""
import json
import logging
import os
import time

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from nnti import influence as inf
from nnti import selection as sel
from nnti.data import (
    SmilesRegressionDataset,
    load_external,
    make_collate_fn,
    stratified_three_way_split,
)
from nnti.models import DEFAULT_MODEL, DEFAULT_REVISION, build_regressor, load_checkpoint, save_checkpoint
from nnti.peft import METHODS, apply_lora, count_parameters
from nnti.train import evaluate, set_seed, train, train_and_test

logger = logging.getLogger(__name__)

# HVP does a double backward over the full parameter set; batches above ~32
# exhaust an 8 GB card.
HVP_BATCH = 8
EVAL_BATCH = 128

_DATA_CACHE = {}

# Stage C2 reference losses, keyed by everything they depend on. Populated
# lazily and reused across the 100 external samples sharing a seed.
_LOO_BASELINE_CACHE = {}


def _lipo_dataframe():
    if "lipo" not in _DATA_CACHE:
        ds = load_dataset("scikit-fingerprints/MoleculeNet_Lipophilicity")["train"]
        _DATA_CACHE["lipo"] = pd.DataFrame(ds)
    return _DATA_CACHE["lipo"]


def _loader(df, tokenizer, batch_size, shuffle, seed=None):
    ds = SmilesRegressionDataset(df["SMILES"].tolist(), df["label"].tolist())
    gen = None
    if shuffle and seed is not None:
        gen = torch.Generator()
        gen.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=make_collate_fn(tokenizer),
        generator=gen,
    )


def baseline_checkpoint_path(artifact_dir, split_seed, model_name=DEFAULT_MODEL):
    """Checkpoint path, namespaced by model.

    Stage F introduces a second backbone, and without the model in the filename
    a ChemBERTa run would happily be handed MoLFormer weights. The default
    model keeps its original unprefixed name so checkpoints already on disk
    from stage B still resolve.
    """
    if model_name == DEFAULT_MODEL:
        return os.path.join(artifact_dir, f"baseline_split{split_seed}.pt")
    tag = model_name.split("/")[-1].replace(".", "_")
    return os.path.join(artifact_dir, f"baseline_{tag}_split{split_seed}.pt")


def run_one(spec, artifact_dir, device=None):
    """Execute one run and return a flat dict of results."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kind = spec["kind"]
    t0 = time.time()

    handler = {
        "baseline": _run_baseline,
        "selection": _run_selection,
        "peft": _run_peft,
        "lissa_sensitivity": _run_lissa_sensitivity,
        "loo": _run_loo,
    }[kind]

    out = handler(spec, artifact_dir, device)
    out.update({
        "kind": kind,
        "stage": spec.get("stage"),
        "wall_seconds": time.time() - t0,
    })
    return out


# ---------------------------------------------------------------------------
# Stage B: baseline. Also produces the checkpoint later stages start from.
# ---------------------------------------------------------------------------
def _run_baseline(spec, artifact_dir, device):
    split_seed, init_seed = spec["split_seed"], spec["init_seed"]
    set_seed(init_seed)
    tokenizer, model = build_regressor(
        spec.get("model_name", DEFAULT_MODEL),
        spec.get("revision", DEFAULT_REVISION),
        dropout_rate=spec.get("dropout", 0.0),
    )
    train_df, val_df, test_df = stratified_three_way_split(_lipo_dataframe(), split_seed)

    model, info = train_and_test(
        model,
        _loader(train_df, tokenizer, spec.get("batch_size", 64), True, init_seed),
        _loader(val_df, tokenizer, EVAL_BATCH, False),
        _loader(test_df, tokenizer, EVAL_BATCH, False),
        device,
        epochs=spec.get("epochs", 20),
        lr=spec.get("lr", 3e-5),
        weight_decay=spec.get("weight_decay", 0.0),
        patience=spec.get("patience", 3),
    )

    if spec.get("save_checkpoint"):
        os.makedirs(artifact_dir, exist_ok=True)
        save_checkpoint(model, baseline_checkpoint_path(
            artifact_dir, split_seed, spec.get("model_name", DEFAULT_MODEL)))

    trainable, total = count_parameters(model)
    return {**_scalar_info(info), "trainable_params": trainable, "total_params": total,
            "n_train": len(train_df), "n_external": 0}


# ---------------------------------------------------------------------------
# Stage C3: does the selection strategy matter, at matched budget?
# ---------------------------------------------------------------------------
def _prepare_external(spec, tokenizer, model, train_df, val_df, device, external_df):
    """Compute whatever the requested strategy needs (embeddings / influence)."""
    strategy = spec["strategy"]
    extras = {}

    if strategy in ("target_alignment", "clustering"):
        emb = sel.embed_smiles(external_df["SMILES"].tolist(), model.base_model,
                               tokenizer, device, amp=False)
        extras["external_embeddings"] = emb
        if strategy == "target_alignment":
            sample = train_df.sample(n=min(200, len(train_df)), random_state=spec["split_seed"])
            tgt = sel.embed_smiles(sample["SMILES"].tolist(), model.base_model,
                                   tokenizer, device, amp=False)
            extras["target_embedding"] = tgt.mean(axis=0)

    if strategy in ("influence_top", "influence_bottom"):
        params = inf.select_parameters(model, spec.get("influence_scope", "head"))
        loss_fn = torch.nn.MSELoss()
        # Influence direction comes from validation, never from test.
        v = inf.average_gradient(model, _loader(val_df, tokenizer, EVAL_BATCH, False),
                                 loss_fn, device, params)
        ihvp = inf.lissa_inverse_hvp(
            model, _loader(train_df, tokenizer, HVP_BATCH, True, spec["init_seed"]),
            loss_fn, v, device, params,
            damping=spec.get("lissa_damping", 0.01),
            scale=spec.get("lissa_scale", 1e3),
            recursion_depth=spec.get("lissa_depth", 100),
        )
        ext_loader = DataLoader(
            SmilesRegressionDataset(external_df["SMILES"].tolist(), external_df["label"].tolist()),
            batch_size=1, shuffle=False, collate_fn=make_collate_fn(tokenizer))
        extras["influence"] = inf.influence_scores(model, ext_loader, loss_fn, ihvp, device, params)

    return extras


def _run_selection(spec, artifact_dir, device):
    split_seed, init_seed = spec["split_seed"], spec["init_seed"]
    model_name = spec.get("model_name", DEFAULT_MODEL)
    revision = spec.get("revision", DEFAULT_REVISION if model_name == DEFAULT_MODEL else None)
    set_seed(init_seed)
    tokenizer, model = build_regressor(model_name, revision, dropout_rate=spec.get("dropout", 0.0))
    train_df, val_df, test_df = stratified_three_way_split(_lipo_dataframe(), split_seed)

    ckpt = baseline_checkpoint_path(artifact_dir, split_seed, model_name)
    load_checkpoint(model, ckpt, map_location=device)
    model.to(device)

    external_df = load_external(spec.get("external_path", "External-Dataset_for_Task2.csv"))
    extras = _prepare_external(spec, tokenizer, model, train_df, val_df, device, external_df)

    n_select = int(round(spec["fraction"] * len(external_df)))
    chosen = sel.select(spec["strategy"], external_df, n_select, seed=init_seed, **extras)

    combined = pd.concat([train_df[["SMILES", "label"]], chosen[["SMILES", "label"]]],
                         ignore_index=True)

    # Retrain from the same starting point for every arm.
    set_seed(init_seed)
    tokenizer, model = build_regressor(model_name, revision, dropout_rate=spec.get("dropout", 0.0))
    load_checkpoint(model, ckpt, map_location=device)

    model, info = train_and_test(
        model,
        _loader(combined, tokenizer, spec.get("batch_size", 64), True, init_seed),
        _loader(val_df, tokenizer, EVAL_BATCH, False),
        _loader(test_df, tokenizer, EVAL_BATCH, False),
        device,
        epochs=spec.get("epochs", 15),
        lr=spec.get("lr", 2e-5),
        patience=spec.get("patience", 3),
    )

    result = {**_scalar_info(info), "n_train": len(train_df),
              "n_external": len(chosen), "n_combined": len(combined)}
    if "influence" in extras:
        s = np.asarray(extras["influence"])
        result.update({"influence_mean": float(s.mean()), "influence_std": float(s.std())})
    return result


# ---------------------------------------------------------------------------
# Stage D: PEFT methods, each with a real no-external-data control arm.
# ---------------------------------------------------------------------------
def _run_peft(spec, artifact_dir, device):
    split_seed, init_seed = spec["split_seed"], spec["init_seed"]
    model_name = spec.get("model_name", DEFAULT_MODEL)
    revision = spec.get("revision", DEFAULT_REVISION if model_name == DEFAULT_MODEL else None)
    set_seed(init_seed)
    tokenizer, model = build_regressor(model_name, revision, dropout_rate=spec.get("dropout", 0.0))
    train_df, val_df, test_df = stratified_three_way_split(_lipo_dataframe(), split_seed)

    ckpt = baseline_checkpoint_path(artifact_dir, split_seed, model_name)
    load_checkpoint(model, ckpt, map_location=device)
    model.to(device)

    external_df = load_external(spec.get("external_path", "External-Dataset_for_Task2.csv"))
    strategy = spec["strategy"]
    if strategy == "none":
        chosen = external_df.iloc[0:0]
    else:
        extras = _prepare_external(spec, tokenizer, model, train_df, val_df, device, external_df)
        n_select = int(round(spec.get("fraction", 0.5) * len(external_df)))
        chosen = sel.select(strategy, external_df, n_select, seed=init_seed, **extras)

    combined = pd.concat([train_df[["SMILES", "label"]], chosen[["SMILES", "label"]]],
                         ignore_index=True)

    set_seed(init_seed)
    tokenizer, model = build_regressor(model_name, revision, dropout_rate=spec.get("dropout", 0.0))
    load_checkpoint(model, ckpt, map_location=device)

    method = spec["method"]
    if method == "lora":
        apply_lora(model, r=spec.get("lora_r", 8), alpha=spec.get("lora_alpha"))
    else:
        METHODS[method](model)
    trainable, total = count_parameters(model)

    model, info = train_and_test(
        model,
        _loader(combined, tokenizer, spec.get("batch_size", 64), True, init_seed),
        _loader(val_df, tokenizer, EVAL_BATCH, False),
        _loader(test_df, tokenizer, EVAL_BATCH, False),
        device,
        epochs=spec.get("epochs", 15),
        lr=spec["lr"],
        patience=spec.get("patience", 3),
    )
    return {**_scalar_info(info), "method": method, "trainable_params": trainable,
            "total_params": total, "n_external": len(chosen), "n_combined": len(combined)}


# ---------------------------------------------------------------------------
# Stage C1: is the influence ranking stable across LiSSA hyperparameters?
# ---------------------------------------------------------------------------
def _run_lissa_sensitivity(spec, artifact_dir, device):
    split_seed = spec["split_seed"]
    set_seed(spec["init_seed"])
    tokenizer, model = build_regressor()
    train_df, val_df, _ = stratified_three_way_split(_lipo_dataframe(), split_seed)
    load_checkpoint(model, baseline_checkpoint_path(artifact_dir, split_seed), map_location=device)
    model.to(device)

    external_df = load_external(spec.get("external_path", "External-Dataset_for_Task2.csv"))
    params = inf.select_parameters(model, spec["influence_scope"])
    loss_fn = torch.nn.MSELoss()
    v = inf.average_gradient(model, _loader(val_df, tokenizer, EVAL_BATCH, False),
                             loss_fn, device, params)
    try:
        ihvp = inf.lissa_inverse_hvp(
            model, _loader(train_df, tokenizer, HVP_BATCH, True, spec["init_seed"]),
            loss_fn, v, device, params,
            damping=spec["lissa_damping"], scale=spec["lissa_scale"],
            recursion_depth=spec["lissa_depth"])
        ext_loader = DataLoader(
            SmilesRegressionDataset(external_df["SMILES"].tolist(), external_df["label"].tolist()),
            batch_size=1, shuffle=False, collate_fn=make_collate_fn(tokenizer))
        scores = inf.influence_scores(model, ext_loader, loss_fn, ihvp, device, params)
    except inf.InfluenceDivergedError as e:
        return {"converged": False, "error": str(e)[:200]}

    s = np.asarray(scores)
    os.makedirs(os.path.join(artifact_dir, "influence"), exist_ok=True)
    out = os.path.join(artifact_dir, "influence", f"{spec['id']}.csv")
    external_df.assign(influence=s).to_csv(out, index=False)
    return {"converged": True, "influence_mean": float(s.mean()), "influence_std": float(s.std()),
            "influence_min": float(s.min()), "influence_max": float(s.max()),
            "scores_path": out}


# ---------------------------------------------------------------------------
# Stage C2: ground-truth check. Does predicted influence match the measured
# effect of actually adding that sample and retraining?
# ---------------------------------------------------------------------------
def _run_loo(spec, artifact_dir, device):
    split_seed, init_seed = spec["split_seed"], spec["init_seed"]
    subset_n = spec.get("train_subset", 500)

    set_seed(init_seed)
    tokenizer, model = build_regressor()
    train_df, val_df, _ = stratified_three_way_split(_lipo_dataframe(), split_seed)
    # A small training set makes a single added sample's effect measurable
    # above seed noise; on the full 2940 it is far below the noise floor.
    train_df = train_df.sample(n=min(subset_n, len(train_df)), random_state=split_seed)

    external_df = load_external(spec.get("external_path", "External-Dataset_for_Task2.csv"))
    row = external_df.iloc[[spec["external_index"]]]
    augmented = pd.concat([train_df[["SMILES", "label"]], row[["SMILES", "label"]]],
                          ignore_index=True)

    ckpt = baseline_checkpoint_path(artifact_dir, split_seed)
    val_loader = _loader(val_df, tokenizer, EVAL_BATCH, False)
    common = dict(epochs=spec.get("epochs", 5), lr=spec.get("lr", 2e-5),
                  patience=spec.get("patience", 5))

    def fit(frame):
        set_seed(init_seed)
        _, m = build_regressor()
        load_checkpoint(m, ckpt, map_location=device)
        m, _ = train(m, _loader(frame, tokenizer, spec.get("batch_size", 32), True, init_seed),
                     val_loader, device, **common)
        mse = evaluate(m, val_loader, device)["mse"]
        del m
        torch.cuda.empty_cache()
        return mse

    # The "without" reference depends only on (split, seed, subset, hparams),
    # not on which external sample is being tested, so it is computed once per
    # seed instead of once per (sample, seed) pair. That halves Stage C2.
    cache_key = (split_seed, init_seed, subset_n, common["epochs"], common["lr"],
                 spec.get("batch_size", 32), common["patience"])
    if cache_key not in _LOO_BASELINE_CACHE:
        _LOO_BASELINE_CACHE[cache_key] = fit(train_df)
    without = _LOO_BASELINE_CACHE[cache_key]
    with_ = fit(augmented)

    return {"external_index": spec["external_index"],
            "val_mse_without": without, "val_mse_with": with_,
            # Negative delta = adding the sample reduced loss = helpful.
            "loo_delta": with_ - without,
            "n_train": len(train_df)}


def _scalar_info(info):
    """Move the per-epoch history aside; the driver logs it to W&B and keeps
    the CSV flat."""
    out = {k: v for k, v in info.items() if k != "history"}
    out["_history"] = info.get("history", [])
    return out
