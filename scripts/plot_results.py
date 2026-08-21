"""Generate the figures embedded in the project README.

The task-results summary is derived only from committed per-run CSVs. The
distribution-shift figure is derived from the Lipophilicity and external
datasets, using the same split and descriptor definitions as Stage E.

Usage:
    python scripts/plot_results.py
"""
import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nnti.data import load_external, stratified_three_way_split


BLUE = "#2563EB"
LIGHT_BLUE = "#93C5FD"
ORANGE = "#EA580C"
RED = "#DC2626"
GREEN = "#059669"
SLATE = "#475569"
GRID = "#CBD5E1"


def load_results(pattern):
    frames = [pd.read_csv(path) for path in sorted(glob.glob(pattern))]
    if not frames:
        raise SystemExit(f"no result files match {pattern!r}")
    return pd.concat(frames, ignore_index=True).drop_duplicates("id", keep="first")


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    return values.mean(), 1.96 * values.std(ddof=1) / np.sqrt(len(values))


def selection_contrasts(data):
    """Return the three paired contrasts that answer Task 2 directly."""
    index = ["split_seed", "init_seed"]
    c3 = data[(data.stage == "C3") & (data.status == "ok")]

    none = c3[c3.strategy == "none"].set_index(index).test_mse.pow(0.5).sort_index()
    all_external = c3[c3.strategy == "all"].set_index(index).test_mse.pow(0.5).sort_index()

    partial = c3[c3.strategy.isin(["influence_top", "influence_bottom"])].copy()
    partial["rmse"] = np.sqrt(partial.test_mse)
    partial = partial.groupby(index + ["strategy"]).rmse.mean().unstack("strategy")

    f2 = data[(data.stage == "F2") & (data.status == "ok")].copy()
    f2["rmse"] = np.sqrt(f2.test_mse)
    f2 = f2.pivot(index=index, columns="strategy", values="rmse")

    contrasts = [
        ("Full external set − none\nMoLFormer", all_external - none, RED),
        ("Influence top − bottom\nMoLFormer", partial.influence_top - partial.influence_bottom, BLUE),
        ("Influence top − bottom\nChemBERTa", f2.influence_top - f2.influence_bottom, GREEN),
    ]
    return [(label, *mean_ci(values), color) for label, values, color in contrasts]


def peft_controls(data):
    """Return no-external-data PEFT performance and trainable-parameter share."""
    rows = data[(data.stage == "D") & (data.status == "ok")
                & (data.strategy == "none")].copy()
    rows["rmse"] = np.sqrt(rows.test_mse)
    labels = {"ia3": "iA3", "bitfit": "BitFit", "lora": "LoRA", "full": "Full"}
    colors = {"ia3": GREEN, "bitfit": ORANGE, "lora": BLUE, "full": SLATE}
    result = []
    for method in ["ia3", "bitfit", "lora", "full"]:
        group = rows[rows.method == method]
        mean, ci = mean_ci(group.rmse)
        share = 100 * group.trainable_params.iloc[0] / group.total_params.iloc[0]
        result.append((labels[method], share, mean, ci, colors[method]))
    return result


def task_results(data, output_path):
    baseline_mse = data[(data.stage == "B") & (data.status == "ok")].test_mse.dropna()
    baseline = np.sqrt(baseline_mse)
    if len(baseline) != 100:
        raise ValueError(f"expected 100 Stage B runs, found {len(baseline)}")

    plt.rcParams.update({
        "font.size": 10,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7))

    # A — Task 1 performance across repeated runs.
    ax = axes[0]
    ax.hist(baseline, bins=13, color=LIGHT_BLUE, edgecolor="white", linewidth=1)
    tuned_rows = data[(data.stage == "C3") & (data.status == "ok")
                      & (data.strategy == "none")]
    tuned = np.sqrt(tuned_rows.test_mse).mean()
    ax.axvline(baseline.mean(), color=BLUE, linewidth=2,
               label=f"Repeated baseline: {baseline.mean():.3f}")
    ax.axvline(tuned, color=ORANGE, linestyle="--", linewidth=2,
               label=f"Tuned downstream control: {tuned:.3f}")
    ax.set_title("A  Task 1 · Model performance", loc="left")
    ax.set_xlabel("Test RMSE (identical configuration)")
    ax.set_ylabel("Run count")
    ax.legend(frameon=False, fontsize=8, loc="upper right",
              title=f"n=100 · SD={baseline.std(ddof=1):.3f}",
              title_fontsize=8, alignment="left")

    # B — the decisive matched contrasts for external-data selection.
    ax = axes[1]
    effects = selection_contrasts(data)[::-1]
    labels = [row[0] for row in effects]
    means = np.array([row[1] for row in effects])
    cis = np.array([row[2] for row in effects])
    colors = [row[3] for row in effects]
    y = np.arange(len(labels))
    ax.axvline(0, color=SLATE, linewidth=1)
    ax.errorbar(means, y, xerr=cis, fmt="none", ecolor=SLATE,
                elinewidth=2, capsize=4)
    ax.scatter(means, y, c=colors, s=45, zorder=3)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Paired Δ test RMSE  ·  positive = worse")
    ax.set_title("B  Task 2 · Does selection help?", loc="left")
    ax.grid(axis="x", color=GRID, alpha=0.6, linewidth=0.7)

    # C — predictive quality against the fraction of weights being updated.
    ax = axes[2]
    methods = peft_controls(data)
    for label, share, mean, ci, color in methods:
        ax.errorbar(share, mean, yerr=ci, fmt="o", color=color, ecolor=color,
                    elinewidth=2, capsize=4, markersize=7)
        offset = {
            "iA3": (5, -18), "BitFit": (5, 8),
            "LoRA": (5, -18), "Full": (-5, 8),
        }[label]
        ax.annotate(label, (share, mean), xytext=offset, textcoords="offset points",
                    ha="right" if label == "Full" else "left", fontsize=9,
                    fontweight="bold", color=color)
    full_mean = next(row[2] for row in methods if row[0] == "Full")
    ax.axhline(full_mean, color=GRID, linestyle="--", linewidth=1)
    # Label the reference line where it sits, so it cannot be read as a gridline.
    ax.text(0.075, full_mean, "full fine-tuning ", ha="left", va="bottom",
            fontsize=8, color=SLATE, style="italic")
    ax.set_xscale("log")
    ax.set_xlim(0.06, 180)
    ax.set_xticks([0.1, 1, 10, 100], ["0.1", "1", "10", "100"])
    ax.set_xlabel("Trainable parameters (%) · log scale")
    ax.set_ylabel("Test RMSE")
    ax.set_title("C  Task 3 · PEFT efficiency", loc="left")
    ax.grid(axis="y", color=GRID, alpha=0.6, linewidth=0.7)

    fig.suptitle("Results across repeated experiments", fontsize=16,
                 fontweight="bold", x=0.04, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=2.5)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def molecular_descriptors(smiles):
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
    except ImportError as exc:
        raise SystemExit("distribution plot requires RDKit (`pip install rdkit`)") from exc

    RDLogger.DisableLog("rdApp.*")
    rows = []
    for value in smiles:
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            continue
        rows.append({
            "cLogP": Crippen.MolLogP(molecule),
            "MolWt": Descriptors.MolWt(molecule),
            "Rings": rdMolDescriptors.CalcNumRings(molecule),
            "TPSA": rdMolDescriptors.CalcTPSA(molecule),
            "HBA": rdMolDescriptors.CalcNumHBA(molecule),
            "HBD": rdMolDescriptors.CalcNumHBD(molecule),
            "RotB": rdMolDescriptors.CalcNumRotatableBonds(molecule),
        })
    return pd.DataFrame(rows)


def distribution_shift(external_path, output_path):
    dataset = load_dataset("scikit-fingerprints/MoleculeNet_Lipophilicity")["train"]
    target = pd.DataFrame(dataset)
    train, _, _ = stratified_three_way_split(target, split_seed=0)
    external = load_external(external_path)
    train_desc = molecular_descriptors(train.SMILES)
    external_desc = molecular_descriptors(external.SMILES)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.5), gridspec_kw={"width_ratios": [1, 1.25]})

    ax = axes[0]
    bins = np.linspace(min(target.label.min(), external.label.min()),
                       max(target.label.max(), external.label.max()), 24)
    ax.hist(train.label, bins=bins, density=True, alpha=0.66, color=BLUE,
            label=f"Training (n={len(train):,})")
    ax.hist(external.label, bins=bins, density=True, alpha=0.72, color=ORANGE,
            label=f"External (n={len(external):,})")
    ax.axvline(train.label.mean(), color=BLUE, linewidth=1.8)
    ax.axvline(external.label.mean(), color=ORANGE, linewidth=1.8)
    ax.set_title("A  The external labels are shifted", loc="left")
    ax.set_xlabel("Lipophilicity (logD)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)

    shifts = {"logD": (external.label.mean() - train.label.mean()) / train.label.std(ddof=0)}
    for column in train_desc.columns:
        shifts[column] = ((external_desc[column].mean() - train_desc[column].mean())
                          / train_desc[column].std(ddof=0))
    shifts = pd.Series(shifts).sort_values()

    ax = axes[1]
    colors = [RED if abs(value) >= 0.5 else BLUE for value in shifts]
    bars = ax.barh(shifts.index, shifts.values, color=colors)
    ax.axvline(0, color=SLATE, linewidth=1)
    ax.axvline(-0.5, color=GRID, linestyle="--", linewidth=1)
    ax.axvline(0.5, color=GRID, linestyle="--", linewidth=1)
    ax.set_xlabel("External − training mean (training SD units)")
    ax.set_title("B  Molecules are smaller and less lipophilic", loc="left")
    ax.set_xlim(-0.95, 0.55)
    ax.grid(axis="x", color=GRID, alpha=0.55, linewidth=0.7)
    for bar, value in zip(bars, shifts.values):
        ax.text(value - 0.015 if value < 0 else value + 0.015,
                bar.get_y() + bar.get_height() / 2, f"{value:+.2f}",
                va="center", ha="right" if value < 0 else "left", fontsize=8)

    fig.suptitle("The external set occupies a different chemical regime", fontsize=16,
                 fontweight="bold", x=0.04, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=2.4)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="experiments/results/*.csv")
    parser.add_argument("--external", default="External-Dataset_for_Task2.csv")
    parser.add_argument("--output-dir", default="figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data = load_results(args.results)
    task_results(data, os.path.join(args.output_dir, "task-results.png"))
    distribution_shift(args.external, os.path.join(args.output_dir, "distribution-shift.png"))
    print(f"wrote figures to {args.output_dir}/")


if __name__ == "__main__":
    main()
