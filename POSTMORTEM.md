# Post-mortem: auditing and rerunning this project

This document records what happened when the original submission was audited
and rerun with a corrected implementation and a measured noise floor.

The short version: **the report's headline finding was not a finding.** The
influence-function pipeline at the centre of Task 2 never produced a single
valid number, and the effect it reported was smaller than the run-to-run noise
of the pipeline that produced it. The report's *explanation* for its negative
results — that the external dataset was distributionally mismatched — turns out
to be broadly correct, but it was never actually tested.

Every claim below is backed by a reproducible check in this repository.

---

## 1. The defects

### 1.1 LiSSA diverged, so every influence score was NaN

`task2.py` implemented the LiSSA recursion as

```python
h = v + (1 - damping) * h - scale * hv      # multiplies
```

where Koh & Liang (2017) / Agarwal et al. (2016) require

```python
h = v + (1 - damping) * h - hv / scale      # divides
ihvp = h / scale                            # and a final division
```

With the default `scale=25`, the coefficient on the Hessian term is 625x too
large, so the Neumann series diverges instead of contracting. Tested against a
synthetic quadratic with a known `H^-1 v` (`tests/test_fixes.py`):

| step | 1 | 2 | 3 | 5 | 10 |
|---|---|---|---|---|---|
| as written | 1.98e2 | 1.25e4 | 9.71e5 | 7.74e9 | **inf** |
| corrected | 1.23e1 | 1.80e1 | 2.35e1 | 3.37e1 | 5.59e1 |

The default `recursion_depth` is 100, so the estimate was `inf` long before
finishing, and every influence score came out `NaN`.

### 1.2 The NaNs were written out unchecked, and selection silently degenerated

All 300 rows of the committed `results/task2/external_influence_scores.csv` have
an empty `influence` column. The selection step was

```python
top = external_df.sort_values(by="influence", ascending=False).head(k)
```

Sorting an all-NaN column places every row last and preserves the original
order, so this returns **exactly the first k rows of the file**. Verified: the
selected indices are `[0, 1, 2, ..., 29]`.

Task 2's "top 10% most influential external samples" were the first 30 lines of
the CSV. The influence machinery contributed nothing, and failed silently.

### 1.3 The reported effect was smaller than the noise

Task 2 reported a +7.57% test-MSE change and read it as evidence that external
data hurts. Running the *identical* baseline configuration 100 times across 10
data splits x 10 seeds:

```
test MSE  mean 0.4639   sd 0.0386   range 0.3797 - 0.5977
single-run 95% uncertainty: +/-16.3% of the mean
```

A +7.57% difference sits well inside that. The report's own reproducibility
table hinted at this — re-running one config gave 0.38175, 0.45489, 0.40246 —
but the implication was not carried through to the comparisons.

The variance decomposition is itself informative:

```
within-split sd (seed only)   = 0.0306
between-split sd (split only) = 0.0251
```

Re-initializing the model moves the result more than re-partitioning the data.

### 1.4 Task 3's models were predicting the mean

The report's Table 11 shows all three PEFT methods collapsing to R^2 ~ 0 once
external data is added. The test set's population variance is **1.45322**, and
the three reported MSEs are 1.4533, 1.4544, 1.4574 — equal to the variance to
four decimal places. An MSE equal to the target variance is the signature of a
model emitting a constant equal to the mean, i.e. one that learned nothing.

Three architecturally different methods landing on exactly that is not gradual
degradation from noisy data; it is a training failure. A mechanism was
available: `task3.py` loaded the Task 1 checkpoint with `strict=False`, which
tolerates a total mismatch in silence. Pointing it at the MLM checkpoint that
sits in the *same output directory* matches **0 of 209 parameters and raises
nothing**, leaving a randomly initialized regression head. With the backbone
frozen by PEFT, such a head converges to the optimal constant: the mean.

### 1.5 LoRA was not a no-op at initialization

`LoRALinear` initialized both factors randomly. Hu et al. (2021) initialize the
up-projection `B` to zero precisely so the adapter contributes nothing before
training. Measured perturbation at initialization: **1.3% relative per wrapped
layer**, injected into every attention and intermediate block. Real, though too
small to be the main story on its own.

### 1.6 Test-set leakage in model selection

`finetune_lipophilicity.py` and `task3.py` used the test set as the validation
set for early stopping *and* best-checkpoint selection, then reported metrics on
that same set. Every "test" number in the report is therefore optimistically
biased, including Task 1's — the reported best configuration was chosen by the
metric it is quoted against.

### 1.7 No control arm

`task3.py` as committed always incorporates external data; there is no flag to
disable it. The "without external data" rows in the report cannot be reproduced
from the submitted code, so the with/without comparison was not controlled.

### 1.8 Half the sweep grid could not run

`config/config_task3_*.yaml` sweep `model_name` over
`[ibm/MoLFormer-XL-both-10pct, task1_best_model.pt]`. The second is a `.pt`
state-dict file passed to `AutoModel.from_pretrained()`, which raises `OSError`.
Roughly half the sweep trials failed at startup.

---

## 2. Two environment findings

Neither is a defect in the original code, but both affect reproducibility and
were discovered only by rerunning on real hardware.

**MoLFormer evaluation was stochastic by default.** The remote model config
defaults `deterministic_eval` to false, which makes its random-feature attention
redraw the projection matrix on every forward pass even under `model.eval()`.
This was initially misdiagnosed as a padding-invariance defect because the
wide-padded and tightly padded inputs were evaluated in separate calls. With
`deterministic_eval=True`, repeated calls and both padding layouts produce
identical pooled embeddings (maximum absolute difference 0). The loader now
enables this option for MoLFormer models, while training remains stochastic as
intended. The change and its verification are recorded in `EXPERIMENT_LOG.md`.

**Mixed precision is unusable here.** fp16 autocast overflows the attention
block completely — every element of the backbone's hidden states returns
non-finite. bf16 stays finite but deviates from fp32 by ~0.79 on a prediction
range of ~0.8, and runs ~25% *slower*, because Turing (SM 7.5) has no native
bf16 tensor cores. All runs are fp32.

---

## 3. What the rerun found

Corrected implementation, three-way splits, validation-based selection, and the
test set touched exactly once per run.

### 3.1 The influence ranking is partly a solver artifact

`scale` rescales the Neumann series; on a converged estimate it cannot change
the ordering. Pairwise Spearman correlation between influence rankings:

| comparison | rho |
|---|---|
| same scope and scale | 0.898 |
| same scope, different scale | **0.578** |
| different parameter scope | 0.618 |

The drop to 0.58 means the recursion has not converged at depth 50-200, so the
ranking is substantially a property of the solver settings rather than the data.

### 3.2 Leave-one-out validation is infeasible at this scale

Influence scores are predictions about what happens if a sample is added. The
direct test is to add it, retrain, and measure. Across 2,900 retraining runs:

| | train subset 500 | train subset 50 |
|---|---|---|
| mean absolute effect | 0.00179 | 0.00399 |
| typical standard error | 0.00382 | 0.00398 |
| **reliability** | **0.000** | **0.000** |
| samples resolved above 2 SE | 0/100 | 1/30 |

Shrinking the training set ten-fold amplified the per-sample effect 2.2x exactly
as intended, but the retraining noise floor did not move. Between-sample
variance in true effect is indistinguishable from zero in both regimes: at these
scales **it makes no measurable difference which external molecule is added.**

This means the estimator can be neither validated nor refuted here, and any
correlation computed against this ground truth is uninterpretable. A tempting
`spearman = -0.46, p = 0.026` appeared in one configuration; across all 160
configurations only 82 of 160 were negative (binomial p = 0.41), i.e. the tail
of a null distribution. It is recorded here as a caution, not a result.

### 3.3 The distribution shift is real

The claim the report used to explain its results, finally measured:

| test | result |
|---|---|
| embedding MMD vs within-dataset baseline | **12.7x**, permutation p = 0.002 |
| cLogP | -0.81 sd (2.22 vs 3.27) |
| MolWt | -0.65 sd (314 vs 385) |
| ring count | -0.65 sd (2.75 vs 3.50) |
| logD label | -0.33 sd, KS p = 9e-16 |

The external molecules are systematically smaller, less lipophilic and less
ring-rich, with a molecular-weight spread **4x narrower** than the training
data — a narrow chemical series rather than a diverse sample. The report's
explanation was broadly right; the evidence it offered for it was not.

---

## 4. Lessons

1. **A silent `NaN` is worse than a crash.** `sort_values` on an all-NaN column
   does not error; it quietly returns file order. The pipeline ran end to end,
   wrote plausible-looking artifacts, and produced a number that went into a
   report. Non-finite values are now hard errors.
2. **`strict=False` hides exactly the failure it is reached for.** Loading the
   wrong checkpoint matched 0 of 209 parameters without complaint.
3. **Measure the noise before interpreting a difference.** The single cheapest
   experiment here — running the same config 100 times — invalidated the
   headline result and cost a few GPU-hours.
4. **Selecting on the test set makes every reported number optimistic**,
   including the hyperparameter choice quoted alongside them.
5. **A comparison without a control is not a comparison.** "With external data
   vs. no external data" conflates *does this ranking work* with *does more data
   help*; the matched-budget control is against random selection at the same
   sample count.
6. **Verify an estimator before trusting it, and be willing to conclude that
   you cannot.** The leave-one-out check was designed to validate the influence
   scores and instead established that validating them is infeasible at this
   scale. That is a result, not a failure.
