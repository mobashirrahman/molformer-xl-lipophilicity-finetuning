"""Stage E: quantify the distribution shift the original report asserted.

The report explained its negative results by claiming the external dataset was
distributionally mismatched with the Lipophilicity data, but never measured it.
This script tests that claim four ways, each with a null model so "different"
means something:

  1. Label distribution        - two-sample KS test
  2. Embedding-space geometry  - kernel MMD with a permutation test
  3. Chemical scaffolds        - Bemis-Murcko overlap vs a within-dataset baseline
  4. Physicochemical space     - RDKit descriptors, standardized differences

The comparison that matters is external vs *training* data, since that is what
the selection step actually mixes together.
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from nnti.data import load_external, stratified_three_way_split  # noqa: E402
from nnti.models import build_regressor  # noqa: E402
from nnti.selection import embed_smiles  # noqa: E402
from scripts.experiment import _lipo_dataframe  # noqa: E402


def rbf_mmd2(X, Y, gamma=None):
    """Biased MMD^2 with an RBF kernel."""
    Z = np.vstack([X, Y])
    if gamma is None:
        # Median heuristic on a subsample keeps this O(1) in dataset size.
        idx = np.random.RandomState(0).choice(len(Z), size=min(500, len(Z)), replace=False)
        d = np.linalg.norm(Z[idx][:, None] - Z[idx][None, :], axis=-1)
        med = np.median(d[d > 0])
        gamma = 1.0 / (2 * med ** 2)

    def k(A, B):
        d2 = ((A[:, None] - B[None, :]) ** 2).sum(-1)
        return np.exp(-gamma * d2)

    n, m = len(X), len(Y)
    return k(X, X).sum() / (n * n) + k(Y, Y).sum() / (m * m) - 2 * k(X, Y).sum() / (n * m)


def mmd_permutation_test(X, Y, n_perm=500, seed=0):
    obs = rbf_mmd2(X, Y)
    Z = np.vstack([X, Y])
    n = len(X)
    rng = np.random.RandomState(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        p = rng.permutation(len(Z))
        null[i] = rbf_mmd2(Z[p[:n]], Z[p[n:]])
    # +1 corrections keep the p-value valid at finite permutation counts.
    return obs, (np.sum(null >= obs) + 1) / (n_perm + 1), null


def murcko_scaffolds(smiles):
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        out.append(MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else None)
    return [s for s in out if s]


def descriptors(smiles):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
    RDLogger.DisableLog("rdApp.*")
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        rows.append({
            "MolWt": Descriptors.MolWt(m),
            "cLogP": Crippen.MolLogP(m),
            "TPSA": rdMolDescriptors.CalcTPSA(m),
            "HBD": rdMolDescriptors.CalcNumHBD(m),
            "HBA": rdMolDescriptors.CalcNumHBA(m),
            "RotB": rdMolDescriptors.CalcNumRotatableBonds(m),
            "Rings": rdMolDescriptors.CalcNumRings(m),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--n-perm", type=int, default=500)
    args = ap.parse_args()

    df = _lipo_dataframe()
    train_df, _, _ = stratified_three_way_split(df, args.split_seed)
    ext = load_external("External-Dataset_for_Task2.csv")
    print(f"training molecules: {len(train_df)}   external molecules: {len(ext)}\n")

    # -- 1. labels ----------------------------------------------------------
    print("=" * 70)
    print("1. LABEL DISTRIBUTION (logD)")
    print("=" * 70)
    a, b = train_df.label.values, ext.label.values
    ks = stats.ks_2samp(a, b)
    print(f"  training  mean={a.mean():.3f} sd={a.std():.3f} range=[{a.min():.2f},{a.max():.2f}]")
    print(f"  external  mean={b.mean():.3f} sd={b.std():.3f} range=[{b.min():.2f},{b.max():.2f}]")
    print(f"  KS statistic={ks.statistic:.4f}  p={ks.pvalue:.3g}")
    print(f"  standardized mean difference = {(b.mean()-a.mean())/a.std():+.3f} sd units")

    # -- 2. embeddings ------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. EMBEDDING SPACE (MoLFormer), MMD with permutation test")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok, model = build_regressor()
    model.to(device)
    sub = train_df.sample(n=min(300, len(train_df)), random_state=args.split_seed)
    Etr = embed_smiles(sub.SMILES.tolist(), model.base_model, tok, device, amp=False)
    Eex = embed_smiles(ext.SMILES.tolist(), model.base_model, tok, device, amp=False)
    obs, p, null = mmd_permutation_test(Etr, Eex, n_perm=args.n_perm)
    print(f"  MMD^2 observed = {obs:.5f}")
    print(f"  null mean      = {null.mean():.5f}  (sd {null.std():.5f})")
    print(f"  permutation p  = {p:.4f}")
    # A within-dataset split gives the scale of "no shift at all".
    h = len(sub) // 2
    ref, _, _ = mmd_permutation_test(Etr[:h], Etr[h:], n_perm=100)
    print(f"  reference MMD^2 (train vs train) = {ref:.5f}")
    print(f"  -> external shift is {obs/ref:.1f}x the within-dataset baseline"
          if ref > 0 else "")

    # -- 3. scaffolds -------------------------------------------------------
    print("\n" + "=" * 70)
    print("3. BEMIS-MURCKO SCAFFOLDS")
    print("=" * 70)
    s_tr = murcko_scaffolds(train_df.SMILES.tolist())
    s_ex = murcko_scaffolds(ext.SMILES.tolist())
    set_tr, set_ex = set(s_tr), set(s_ex)
    shared = set_tr & set_ex
    print(f"  unique scaffolds: training={len(set_tr)}  external={len(set_ex)}")
    print(f"  shared scaffolds: {len(shared)}")
    print(f"  external scaffolds also seen in training: "
          f"{len(shared)/len(set_ex)*100:.1f}%")
    covered = sum(1 for s in s_ex if s in set_tr)
    print(f"  external MOLECULES whose scaffold appears in training: "
          f"{covered}/{len(s_ex)} ({covered/len(s_ex)*100:.1f}%)")
    # Baseline: how much does training overlap with itself when split in half?
    half = len(s_tr) // 2
    base = len(set(s_tr[:half]) & set(s_tr[half:])) / max(1, len(set(s_tr[half:])))
    print(f"  within-training baseline overlap: {base*100:.1f}%")

    # -- 4. physicochemical -------------------------------------------------
    print("\n" + "=" * 70)
    print("4. PHYSICOCHEMICAL DESCRIPTORS (RDKit)")
    print("=" * 70)
    Dtr, Dex = descriptors(train_df.SMILES.tolist()), descriptors(ext.SMILES.tolist())
    print(f"  {'descriptor':<10} {'train':>16} {'external':>16} {'std diff':>9} {'KS p':>10}")
    for c in Dtr.columns:
        x, y = Dtr[c].values, Dex[c].values
        d = (y.mean() - x.mean()) / x.std() if x.std() else 0
        kp = stats.ks_2samp(x, y).pvalue
        flag = "  <-- shifted" if abs(d) > 0.5 else ""
        print(f"  {c:<10} {x.mean():>8.2f}+/-{x.std():<6.2f} {y.mean():>8.2f}+/-{y.std():<6.2f} "
              f"{d:>+9.2f} {kp:>10.3g}{flag}")

    print("\n" + "=" * 70)
    print("A large, significant shift would support the original report's")
    print("explanation. A small one means the negative results need another.")
    print("=" * 70)


if __name__ == "__main__":
    main()
