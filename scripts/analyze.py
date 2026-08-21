"""Aggregate the sweep into the claims it can actually support.

The organising principle is that no comparison is reported without the noise
band it has to clear. The original project compared single runs against single
runs; stage B measures how far apart two identical configurations land, and
every later effect is judged against that.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

RESULTS_GLOB = "experiments/results/*.csv"
INFLUENCE_DIR = "experiments/influence"


def load_results(pattern=RESULTS_GLOB):
    frames = [pd.read_csv(f) for f in sorted(glob.glob(pattern))]
    if not frames:
        raise SystemExit(f"no result files matching {pattern}")
    d = pd.concat(frames, ignore_index=True)
    return d.drop_duplicates("id", keep="first")


def _fmt_ci(values, label=""):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return f"{label}n={len(v)}"
    m, sd = v.mean(), v.std(ddof=1)
    sem = sd / np.sqrt(len(v))
    return f"{label}{m:.4f} ± {1.96*sem:.4f} (sd={sd:.4f}, n={len(v)})"


# ---------------------------------------------------------------------------
def report_baseline(d):
    b = d[(d.stage == "B") & (d.status == "ok")]
    if b.empty:
        return None
    print("=" * 74)
    print("STAGE B - how far apart do identical configurations land?")
    print("=" * 74)
    print(f"  test MSE   {_fmt_ci(b.test_mse)}")
    print(f"  test R2    {_fmt_ci(b.test_r2)}")

    within = b.groupby("split_seed").test_mse.std(ddof=1).mean()
    between = b.groupby("split_seed").test_mse.mean().std(ddof=1)
    print(f"\n  within-split sd (seed only)   = {within:.4f}")
    print(f"  between-split sd (split only) = {between:.4f}")
    print(f"  -> {'seed' if within > between else 'split'} variation dominates")

    sd, m = b.test_mse.std(ddof=1), b.test_mse.mean()
    print(f"\n  A single run carries 95% uncertainty of +/-{1.96*sd/m*100:.1f}% of the mean.")
    print(f"  Any single-run comparison smaller than that is not evidence.")
    return {"mean": m, "sd": sd, "within": within, "between": between, "n": len(b)}


# ---------------------------------------------------------------------------
def load_influence_frames(directory=INFLUENCE_DIR):
    """Map run id -> per-external-sample influence scores."""
    out = {}
    for path in glob.glob(os.path.join(directory, "*.csv")):
        rid = os.path.splitext(os.path.basename(path))[0]
        try:
            df = pd.read_csv(path)
            if "influence" in df.columns:
                out[rid] = df
        except Exception:
            continue
    return out


def report_lissa_stability(d, influence):
    """A ranking that changes with the solver's own hyperparameters cannot be
    a property of the data, so this gates whether C3's selection means anything."""
    c1 = d[(d.stage == "C1") & (d.status == "ok")]
    if c1.empty or not influence:
        return None
    print("\n" + "=" * 74)
    print("STAGE C1 - is the influence RANKING stable, or an artifact of LiSSA?")
    print("=" * 74)
    conv = c1.get("converged")
    if conv is not None:
        print(f"  configurations converged: {int(conv.sum())}/{len(c1)}")

    rows = []
    for split, grp in c1.groupby("split_seed"):
        ids = [i for i in grp.id if i in influence]
        if len(ids) < 2:
            continue
        # Rank-correlate every pair of configurations on the same split.
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                x = influence[ids[a]]["influence"].values
                y = influence[ids[b]]["influence"].values
                if len(x) != len(y):
                    continue
                rho = stats.spearmanr(x, y).statistic
                ra = grp[grp.id == ids[a]].iloc[0]
                rb = grp[grp.id == ids[b]].iloc[0]
                rows.append({
                    "split_seed": split, "rho": rho,
                    "same_scope": ra.influence_scope == rb.influence_scope,
                    "same_scale": ra.lissa_scale == rb.lissa_scale,
                })
    if not rows:
        return None
    r = pd.DataFrame(rows)
    print(f"\n  pairwise Spearman rho between configurations (n={len(r)} pairs):")
    print(f"    all pairs                {_fmt_ci(r.rho)}")
    same = r[r.same_scope & r.same_scale].rho
    diffscale = r[r.same_scope & ~r.same_scale].rho
    diffscope = r[~r.same_scope].rho
    if len(same):      print(f"    same scope & scale       {_fmt_ci(same)}")
    if len(diffscale): print(f"    same scope, diff scale   {_fmt_ci(diffscale)}")
    if len(diffscope): print(f"    different scope          {_fmt_ci(diffscope)}")
    print("\n  Interpretation: rho near 1 means the ranking is a property of the data;")
    print("  rho near 0 means it is a property of the solver settings.")
    return r


# ---------------------------------------------------------------------------
def report_validity(d):
    """Checks that could invalidate everything else, run before any result.

    The sweep ran unattended across 12 machines for days. If the machine a run
    landed on shifted its result, or if strategies were unevenly distributed
    across machines, then every comparison downstream is confounded by
    hardware rather than by the thing being compared. Neither is visible in
    the per-stage summaries, so it is checked explicitly.
    """
    print("\n" + "=" * 74)
    print("VALIDITY - is anything confounded by which machine ran it?")
    print("=" * 74)
    b = d[(d.stage == "B") & (d.status == "ok")]
    if b.empty:
        return
    gpus = sorted({str(x) for x in b.gpu.dropna().unique()})
    print(f"  GPU models in fleet: {gpus}")
    if len(gpus) > 1:
        print("  WARNING: heterogeneous fleet; a host effect is confounded with GPU model")

    g = b.groupby("host").test_mse.agg(["mean", "std", "count"])
    groups = [v.test_mse.values for _, v in b.groupby("host") if len(v) > 1]
    if len(groups) > 1:
        f, p = stats.f_oneway(*groups)
        verdict = "HOST EFFECT PRESENT" if p < 0.05 else "no detectable host effect"
        print(f"  stage B one-way ANOVA across {len(groups)} hosts: "
              f"F={f:.3f} p={p:.4f} -> {verdict}")
    print(f"  between-host sd of means {g['mean'].std():.4f}"
          f"  vs overall run-to-run sd {b.test_mse.std():.4f}")

    c3 = d[(d.stage == "C3") & (d.status == "ok")]
    if not c3.empty and c3.host.nunique() > 1:
        ct = pd.crosstab(c3.strategy, c3.host)
        chi = stats.chi2_contingency(ct.values)
        verdict = ("CONFOUNDED: strategies are not evenly spread across hosts"
                   if chi.pvalue < 0.05 else
                   "strategies spread evenly across hosts")
        print(f"  C3 strategy x host chi-square: p={chi.pvalue:.4f} -> {verdict}")


# ---------------------------------------------------------------------------
def report_loo_validation(d, influence):
    """The decisive test: does predicted influence match the measured effect of
    actually adding that sample and retraining?"""
    c2 = d[(d.stage == "C2") & (d.status == "ok")]
    if c2.empty:
        return None
    print("\n" + "=" * 74)
    print("STAGE C2 - do influence scores predict the MEASURED effect? (ground truth)")
    print("=" * 74)
    print(f"  LOO runs: {len(c2)}  over {c2.external_index.nunique()} external samples,"
          f" {c2.split_seed.nunique()} splits, {c2.init_seed.nunique()} seeds")
    print(f"  loo_delta {_fmt_ci(c2.loo_delta)}")

    # Average the measured effect over seeds/splits to beat down the noise floor.
    measured = c2.groupby("external_index").agg(
        loo_mean=("loo_delta", "mean"),
        loo_sem=("loo_delta", lambda v: v.std(ddof=1) / np.sqrt(len(v))),
        n=("loo_delta", "size"),
    ).reset_index()
    print(f"\n  per-sample measured effect, averaged over {measured.n.median():.0f} runs:")
    print(f"    mean |effect| = {measured.loo_mean.abs().mean():.5f}, "
          f"typical SEM = {measured.loo_sem.median():.5f}")
    detectable = (measured.loo_mean.abs() > 2 * measured.loo_sem).sum()
    print(f"    samples whose effect exceeds 2*SEM: {detectable}/{len(measured)}")

    c1 = d[(d.stage == "C1") & (d.status == "ok")]
    if not influence or c1.empty:
        return measured

    print("\n  correlation of predicted influence vs measured LOO effect:")
    print("  (influence is defined so POSITIVE = helpful; loo_delta NEGATIVE = helpful,")
    print("   so a working estimator gives a NEGATIVE correlation)")
    best = []
    for _, cfg in c1.iterrows():
        fr = influence.get(cfg.id)
        if fr is None:
            continue
        m = measured.merge(fr.reset_index().rename(columns={"index": "external_index"}),
                           on="external_index", how="inner")
        if len(m) < 10:
            continue
        pr = stats.pearsonr(m.influence, m.loo_mean)
        sr = stats.spearmanr(m.influence, m.loo_mean)
        best.append((cfg.influence_scope, cfg.lissa_scale, cfg.lissa_depth,
                     pr.statistic, sr.statistic, sr.pvalue))
    if not best:
        return measured

    # Report the DISTRIBUTION over all configurations, never the best one.
    # Sorting 160 correlations and quoting the most negative is a guaranteed
    # false positive: the minimum of 160 draws from a null is always
    # "significant" at p<0.05. This project already produced one such artifact
    # (rho=-0.46, p=0.026) that dissolved under a sign test, so the summary is
    # built to make that mistake impossible rather than merely discouraged.
    rho = np.array([t[4] for t in best])
    pv = np.array([t[5] for t in best])
    n = len(rho)
    neg = int((rho < 0).sum())
    sign_p = stats.binomtest(neg, n, 0.5).pvalue
    mean_p = stats.ttest_1samp(rho, 0).pvalue
    bonf = 0.05 / n

    print(f"\n  across all {n} LiSSA configurations:")
    print(f"    mean rho   {rho.mean():+.4f}   median {np.median(rho):+.4f}"
          f"   sd {rho.std(ddof=1):.4f}")
    print(f"    range      {rho.min():+.3f} to {rho.max():+.3f}")
    print(f"    negative   {neg}/{n}   (sign test vs 50/50: p = {sign_p:.3f})")
    print(f"    H0 mean rho = 0:  t-test p = {mean_p:.4f}")
    print(f"    nominally p<0.05: {int((pv < 0.05).sum())}/{n}"
          f"   (expected by chance: {0.05 * n:.1f})")
    print(f"    surviving Bonferroni p<{bonf:.5f}: {int((pv < bonf).sum())}/{n}")

    resolved = (pv < bonf).any() and rho[pv < bonf].mean() < 0
    verdict = ("the estimator carries usable signal" if resolved else
               "NO evidence the estimator carries signal: the correlations are "
               "centred on zero\n             and no configuration survives "
               "correction for multiple comparisons")
    print(f"\n  Verdict: {verdict}.")
    print("  The single most negative configuration is deliberately not quoted"
          " as a result;\n  it is the minimum of "
          f"{n} draws and is significant by construction.")
    return measured


# ---------------------------------------------------------------------------
def report_selection(d, baseline):
    c3 = d[(d.stage == "C3") & (d.status == "ok")]
    if c3.empty:
        return
    print("\n" + "=" * 74)
    print("STAGE C3 - does the selection strategy beat random at matched budget?")
    print("=" * 74)
    for frac, grp in c3.groupby("fraction"):
        print(f"\n  fraction={frac}")
        for strat, g in grp.groupby("strategy"):
            print(f"    {strat:<18} {_fmt_ci(g.test_mse, 'test MSE ')}")
        # Paired against random on identical (split, seed) pairs.
        rnd = grp[grp.strategy == "random"].set_index(["split_seed", "init_seed"]).test_mse
        for strat, g in grp.groupby("strategy"):
            if strat == "random":
                continue
            gg = g.set_index(["split_seed", "init_seed"]).test_mse
            common = gg.index.intersection(rnd.index)
            if len(common) < 3:
                continue
            diff = gg.loc[common] - rnd.loc[common]
            t = stats.ttest_rel(gg.loc[common], rnd.loc[common])
            print(f"      {strat} - random: {diff.mean():+.4f} "
                  f"(paired t p={t.pvalue:.3g}, n={len(common)})")


def report_peft(d):
    dd = d[(d.stage == "D") & (d.status == "ok")]
    if dd.empty:
        return
    print("\n" + "=" * 74)
    print("STAGE D - PEFT methods, with a real no-external-data control")
    print("=" * 74)
    for method, g in dd.groupby("method"):
        print(f"\n  {method}  (trainable={g.trainable_params.iloc[0]:,.0f})")
        for strat, gg in g.groupby("strategy"):
            print(f"    {strat:<18} {_fmt_ci(gg.test_mse, 'test MSE ')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=RESULTS_GLOB)
    ap.add_argument("--influence", default=INFLUENCE_DIR)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    d = load_results(args.results)
    print(f"loaded {len(d)} unique runs across {d.host.nunique()} hosts\n")
    print(d.groupby(["stage", "status"]).size().to_string())
    print()

    influence = load_influence_frames(args.influence)
    report_validity(d)
    base = report_baseline(d)
    report_lissa_stability(d, influence)
    measured = report_loo_validation(d, influence)
    report_selection(d, base)
    report_peft(d)

    if args.json_out and base:
        with open(args.json_out, "w") as fh:
            json.dump({"baseline": base, "n_runs": int(len(d))}, fh, indent=2)


if __name__ == "__main__":
    main()
