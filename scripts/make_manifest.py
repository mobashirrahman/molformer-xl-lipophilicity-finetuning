"""Generate the full study manifest as newline-delimited JSON.

Runs are emitted in dependency and priority order, so that interrupting the
driver at any point still leaves a coherent, analyzable subset:

  B  baseline               -> the seed/split noise band, plus the checkpoints
                               every later stage starts from
  C1 LiSSA sensitivity      -> is the influence ranking an artifact of its
                               own hyperparameters?
  C2 leave-one-out          -> do the influence scores predict the measured
                               effect of actually adding a sample?
  C3 selection strategies   -> the main question, at matched sample budgets
  D  PEFT x selection       -> Task 3, with a real no-external-data control
"""
import argparse
import hashlib
import json
import os

SPLIT_SEEDS = list(range(10))
INIT_SEEDS = list(range(10))

# Fallback training hyperparameters, used only when no completed HPO study is
# available. The originals were selected against the test set, so they are a
# starting point rather than a trusted configuration.
FALLBACK = {"lr": 3e-5, "batch_size": 64, "weight_decay": 0.0,
            "dropout": 0.0, "patience": 3}
PEFT_FALLBACK_LR = {"bitfit": 5e-3, "lora": 2e-5, "ia3": 1e-4, "full": 2e-5}


def tuned(study="stageA-baseline", journal="experiments/hpo/journal.log"):
    """Best validation-selected hyperparameters, or the fallback."""
    if not os.path.exists(journal):
        return dict(FALLBACK), "fallback (no HPO study found)"
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        st = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(journal))
        st_study = optuna.load_study(study_name=study, storage=st)
        done = [t for t in st_study.trials
                if t.state == optuna.trial.TrialState.COMPLETE]
        if not done:
            return dict(FALLBACK), "fallback (study has no complete trials)"
        p = dict(FALLBACK)
        p.update(st_study.best_params)
        return p, f"HPO best of {len(done)} trials (val_mse={st_study.best_value:.4f})"
    except Exception as exc:
        return dict(FALLBACK), f"fallback ({type(exc).__name__})"


def peft_lr(method, journal="experiments/hpo/journal.log"):
    """Per-method learning rate from its own study, else the original value."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        st = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(journal))
        s = optuna.load_study(study_name=f"stageD-{method}", storage=st)
        if [t for t in s.trials if t.state == optuna.trial.TrialState.COMPLETE]:
            return s.best_params["lr"]
    except Exception:
        pass
    return PEFT_FALLBACK_LR[method]


def run_id(spec):
    payload = json.dumps({k: v for k, v in spec.items() if k != "id"}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def emit(specs, spec):
    spec["id"] = run_id(spec)
    specs.append(spec)


def build():
    specs = []
    global HP
    HP, provenance = tuned()
    print(f"training hyperparameters: {provenance}")
    print(f"  {HP}")

    # -- Stage B ------------------------------------------------------------
    # init_seed 0 of each split writes the checkpoint that C/D reuse.
    for s in SPLIT_SEEDS:
        for i in INIT_SEEDS:
            emit(specs, {
                "stage": "B", "kind": "baseline", "split_seed": s, "init_seed": i,
                "epochs": 20, "lr": 3e-5, "batch_size": 64, "patience": 3,
                "save_checkpoint": i == 0,
            })

    # -- Stage C1 -----------------------------------------------------------
    for s in SPLIT_SEEDS[:5]:
        for scope in ["head", "all"]:
            for scale in [1e3, 1e4, 1e5, 1e6]:
                for damping in [0.0, 0.01]:
                    for depth in [50, 200]:
                        emit(specs, {
                            "stage": "C1", "kind": "lissa_sensitivity",
                            "split_seed": s, "init_seed": 0,
                            "influence_scope": scope, "lissa_scale": scale,
                            "lissa_damping": damping, "lissa_depth": depth,
                        })

    # -- Stage C2 -----------------------------------------------------------
    # Deliberately a small training subset: the effect of one added sample on
    # the full 2940-sample training set sits far below seed noise.
    #
    # Spread over 5 splits x 4 init seeds rather than 20 seeds of a single
    # split. Same run count and the same number of cached reference fits, but
    # the LOO check is then validated across data splits instead of resting on
    # one partition -- and it parallelizes across hosts instead of pinning all
    # 2000 runs to whichever machine owns split 0.
    for idx in range(100):
        for s in SPLIT_SEEDS[:5]:
            for i in range(4):
                emit(specs, {
                    "stage": "C2", "kind": "loo", "split_seed": s, "init_seed": i,
                    "external_index": idx, "train_subset": 500,
                    "epochs": 5, "lr": 2e-5, "batch_size": 32, "patience": 5,
                })

    # -- Stage C2b ----------------------------------------------------------
    # C2 at train_subset=500 found the between-sample variance in true LOO
    # effect to be indistinguishable from zero (reliability ~0.00), so no
    # amount of averaging can validate the influence estimator there.
    #
    # This arm shrinks the training set to 50, where a single added molecule is
    # ~2% of the data and its effect should clear the retraining noise floor.
    # The point is not realism -- it is to check the estimator is correct in a
    # regime where it can be checked at all, so that "no measurable effect at
    # realistic scale" can be reported as a property of the problem rather than
    # a possible bug in our implementation.
    for idx in range(30):
        for s in SPLIT_SEEDS[:5]:
            for i in range(6):
                emit(specs, {
                    "stage": "C2b", "kind": "loo", "split_seed": s, "init_seed": i,
                    "external_index": idx, "train_subset": 50,
                    "epochs": 8, "lr": 2e-5, "batch_size": 16, "patience": 8,
                })

    # -- Stage C3 -----------------------------------------------------------
    strategies = ["random", "target_alignment", "clustering",
                  "influence_top", "influence_bottom"]
    for s in SPLIT_SEEDS[:5]:
        for i in INIT_SEEDS[:3]:
            for strategy in strategies:
                for fraction in [0.10, 0.25, 0.50]:
                    emit(specs, {
                        "stage": "C3", "kind": "selection", "split_seed": s, "init_seed": i,
                        "strategy": strategy, "fraction": fraction,
                        "influence_scope": "head", "lissa_scale": 1e3,
                        "lissa_damping": 0.01, "lissa_depth": 100,
                        "epochs": 20, "lr": HP["lr"], "batch_size": HP["batch_size"],
                        "weight_decay": HP["weight_decay"], "dropout": HP["dropout"],
                        "patience": HP["patience"],
                    })
            # Controls: no external data at all, and all of it.
            for strategy in ["none", "all"]:
                emit(specs, {
                    "stage": "C3", "kind": "selection", "split_seed": s, "init_seed": i,
                    "strategy": strategy, "fraction": 1.0,
                    "epochs": 20, "lr": HP["lr"], "batch_size": HP["batch_size"],
                    "weight_decay": HP["weight_decay"], "dropout": HP["dropout"],
                    "patience": HP["patience"],
                })

    # -- Stage D ------------------------------------------------------------
    # Learning rates are the originals; scripts/hpo.py refines them per method.
    lrs = {m: peft_lr(m) for m in ("bitfit", "lora", "ia3", "full")}
    for s in SPLIT_SEEDS[:5]:
        for i in INIT_SEEDS[:3]:
            for method in ["bitfit", "lora", "ia3", "full"]:
                for strategy in ["none", "random", "target_alignment",
                                 "clustering", "influence_top"]:
                    emit(specs, {
                        "stage": "D", "kind": "peft", "split_seed": s, "init_seed": i,
                        "method": method, "strategy": strategy, "fraction": 0.50,
                        "influence_scope": "head", "lissa_scale": 1e3,
                        "lissa_damping": 0.01, "lissa_depth": 100,
                        "epochs": 20, "lr": lrs[method],
                        "batch_size": HP["batch_size"],
                        "dropout": HP["dropout"], "patience": HP["patience"],
                    })
    # -- Stage F ------------------------------------------------------------
    # Does the headline finding depend on MoLFormer? ChemBERTa-zinc-base is
    # chosen because it has 44.1M parameters against MoLFormer's 44.4M, so a
    # difference cannot be written off as model capacity.
    #
    # Split in two: F1 writes the ChemBERTa baselines that F2 loads. Hash
    # sharding scatters runs across hosts, so the checkpoints must exist and be
    # distributed before any F2 run starts.
    CHEMBERTA = "seyonec/ChemBERTa-zinc-base-v1"
    for s in SPLIT_SEEDS[:5]:
        for i in INIT_SEEDS[:3]:
            emit(specs, {
                "stage": "F1", "kind": "baseline", "split_seed": s, "init_seed": i,
                "model_name": CHEMBERTA, "revision": None,
                "epochs": 20, "lr": 3e-5, "batch_size": 64, "patience": 3,
                "save_checkpoint": i == 0,
            })
    for s in SPLIT_SEEDS[:5]:
        for i in INIT_SEEDS[:3]:
            for strategy in ["none", "random", "target_alignment",
                             "influence_top", "influence_bottom"]:
                emit(specs, {
                    "stage": "F2", "kind": "selection", "split_seed": s, "init_seed": i,
                    "model_name": CHEMBERTA, "revision": None,
                    "strategy": strategy, "fraction": 0.50,
                    "influence_scope": "head", "lissa_scale": 1e3,
                    "lissa_damping": 0.01, "lissa_depth": 100,
                    "epochs": 15, "lr": 2e-5, "batch_size": 64, "patience": 3,
                })

    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/manifest.jsonl")
    ap.add_argument("--stages", default=None,
                    help="comma-separated subset, e.g. B,C1")
    args = ap.parse_args()

    specs = build()
    if args.stages:
        keep = set(args.stages.split(","))
        specs = [s for s in specs if s["stage"] in keep]

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        for spec in specs:
            fh.write(json.dumps(spec) + "\n")

    counts = {}
    for s in specs:
        counts[s["stage"]] = counts.get(s["stage"], 0) + 1
    print(f"wrote {len(specs)} runs to {args.out}")
    for stage in sorted(counts):
        print(f"  stage {stage}: {counts[stage]}")


if __name__ == "__main__":
    main()
