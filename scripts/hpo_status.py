"""Print the state of every Optuna study in the journal."""
import sys, optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
path = sys.argv[1] if len(sys.argv) > 1 else "experiments/hpo/journal.log"
st = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(path))
T = optuna.trial.TrialState
for name in sorted(optuna.study.get_all_study_names(storage=st)):
    s = optuna.load_study(study_name=name, storage=st)
    n = {k: 0 for k in ("COMPLETE", "PRUNED", "FAIL", "RUNNING", "WAITING")}
    for t in s.trials:
        n[t.state.name] = n.get(t.state.name, 0) + 1
    line = (f"{name:24s} total={len(s.trials):4d}  complete={n['COMPLETE']:4d} "
            f"pruned={n['PRUNED']:4d} running={n['RUNNING']:3d} fail={n['FAIL']:3d}")
    if n["COMPLETE"]:
        line += f"  best={s.best_value:.4f}"
    print(line)
    if n["COMPLETE"]:
        print(f"{'':24s} {s.best_params}")
