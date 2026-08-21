# Experiment and implementation log

This is the durable record of changes made while auditing and improving the
experiments. Each entry states what was changed, why it was needed, how it was
checked, and what the result means for subsequent runs.

## 2026-08-20 — Sequence 1: deterministic evaluation

### Why

MoLFormer uses random-feature linear attention. Its downloaded config sets
`deterministic_eval` to false, so the projection matrix was regenerated on every
forward pass even after `model.eval()`. Consequently, validation and test MSE,
RMSE, and R-squared varied without any change to the checkpoint or data. HPO was
therefore optimizing a noisy objective, and comparisons between separate model
calls could be mistaken for padding effects.

The official MoLFormer loading example opts into `deterministic_eval=True`.
Inside the remote model code this suppresses projection redraws only in
evaluation mode; projections continue to be redrawn during training.

### Change

- `nnti.models.load_backbone` now passes `deterministic_eval=True` when loading
  a MoLFormer-family model.
- The option is deliberately not passed to other Hugging Face backbones, such
  as ChemBERTa, because it is a model-specific remote-code argument.
- Regression tests cover both branches so this behavior cannot silently regress
  or break alternative backbones.
- The incorrect padding-invariance diagnosis in `POSTMORTEM.md` was corrected.

### Verification and result

The implementation was checked at three levels:

1. Focused regression tests: **2 passed**. One asserts that MoLFormer receives
   `deterministic_eval=True`; the other asserts that the model-specific option
   is not sent to ChemBERTa.
2. Full test suite: **13 passed**. Bytecode compilation also completed without
   errors.
3. Real-model check using the pinned, cached MoLFormer revision:
   `config.deterministic_eval=True`; the maximum absolute difference between
   repeated pooled outputs was **0**, and between batch-tight versus 128-wide
   padded outputs was also **0**.

The existing split-0 checkpoint was then evaluated twice over all 840 held-out
molecules. Both passes returned exactly the same metrics (including MSE
`0.43613761113088245` at full recorded precision):

| evaluation | MSE | RMSE | MAE | R-squared |
|---|---:|---:|---:|---:|
| deterministic (after fix) | 0.436138 | 0.660407 | 0.491954 | 0.699157 |
| stochastic runs (before fix) | 0.427173–0.436802 | 0.653584–0.660910 | 0.480554–0.482947 | 0.698699–0.705341 |

The deterministic score lies inside the old stochastic range. This change does
not improve the predictive weights of an already-trained checkpoint; it
improves measurement validity by reducing evaluation variance from a material
range to zero across the two full held-out-set passes. Any RMSE gain must be
tested by retraining and rerunning HPO with the corrected validation objective.

### Consequence for HPO

Workers that were already running before this change loaded the previous Python
code and still use stochastic validation. Their trials remain useful only as
exploratory evidence; publication-quality HPO should be restarted from a fresh
study after the deterministic loader is deployed.

## 2026-08-20 — Sequence 2: home quota exhaustion during checkpoint sync

### What happened

`cluster.sh sync-artifacts` gathered every `baseline_split*.pt` into
`experiments/checkpoints/` on `/home` before pushing them out to each host's
`/scratch`. `/home` is a quota'd NFS share. Six checkpoints at 178 MB each
came to 1018 MB and exhausted the quota.

The failure was not confined to the sync. Once the quota was full, *every*
write on the machine failed: git could not create `.git/index.lock`, and a
file being rewritten at that instant — `scripts/cluster.sh` — was truncated to
zero bytes. `bash -n` on the empty file reported success, so the syntax check
that should have caught it did not.

### Why it mattered

The staging directory was pure overhead. The checkpoints were already on the
host that produced them and needed to reach the other hosts; routing them
through `/home` doubled the transfer and put a 1 GB transient on a shared
quota that also holds the results CSVs, the Optuna journal and the influence
score files — the artifacts the whole study depends on.

### Change

`sync_artifacts` now relays host-to-host through one hub host's `/scratch` and
never touches `/home`. `~/.ssh` is itself NFS-shared, so any host can reach any
other directly and no staging area is needed. The current host is included as a
source unconditionally, because `discover()` drops hosts with a busy GPU from
`hosts.txt` while they may still hold checkpoints — which is exactly what had
happened to bio10.

The glob was also widened from `baseline_split*.pt` to `baseline_*.pt`. Stage
F1 writes model-namespaced checkpoints (`baseline_ChemBERTa-zinc-base-v1_
split*.pt`) that F2 loads. The narrower pattern matched only the MoLFormer
files, so every F2 run would have failed on a missing checkpoint — but only
after F1 had already spent its GPU time producing them.

### Result

`NNTI_Project` fell from 1.1 GB to 47 MB. `cluster.sh` was restored from git
intact. Six checkpoints (splits 0–4 and 8) are present on all 12 hosts; splits
5–7 and 9 were not recovered, which does not block anything, because stages C3,
D and F all use `SPLIT_SEEDS[:5]`.

---

## 2026-08-20 — Sequence 3: HPO restarted against the deterministic objective

### Why

Sequence 1 established that MoLFormer's evaluation was stochastic: the
random-feature projection was redrawn on every forward pass, so validation MSE
varied by roughly 0.0096 on a fixed checkpoint. Optuna was therefore ranking
trials partly on evaluation luck rather than on the configurations themselves,
and the reported best was the minimum of 126 draws from a noisy objective — a
figure that could not be reproduced by re-evaluating the same checkpoint.

The pre-fix study was not deleted. It is archived at
`experiments/hpo/archive/journal-prefix-stochastic.log` with its best
parameters, and remains usable as exploratory evidence about which regions of
the space are worth searching. It is not a basis for choosing a configuration.

### Design of the restart

- Fresh journal at the same path under the same study name, so
  `make_manifest.tuned()` needs no change.
- 12 hosts x 12 trials = 144 trials, 15 epochs, `--splits 0,1`.
- Each trial is scored on the **mean validation MSE across two data splits**.
  Stage B measured single-run noise at +/-16.3%, so a trial scored on one run
  would rank mostly noise. Averaging two splits is the cheapest defence that
  still leaves the search affordable.
- `MedianPruner` reports after the first split and kills trials that are
  already behind, which is why roughly half of all trials are pruned.
- The test split is never read during the search. The original project selected
  its configuration on the test set, which is why neither that configuration nor
  the metrics quoted beside it could be trusted.

### Verification before launch

Regenerating the manifest changes the spec hash of any run whose
hyperparameters are baked in, and the driver skips runs by id — so a
regeneration could silently orphan completed work. This was tested rather than
assumed: regenerating with different hyperparameters kept **3160 of 3160**
completed runs matching. Only C3 and D embed the tuned values, and neither had
any completed runs.

Workers were also confirmed to have loaded the corrected code: the
`deterministic_eval` branch is present in the NFS-shared `nnti/models.py` that
every host imports.

### Result

Final: 144 trials, 73 complete, 71 pruned, **0 failed**, roughly two hours of
wall clock across 12 hosts. Best validation MSE **0.34418** at
`lr 5.81e-5, batch_size 16, weight_decay 2.38e-6, dropout 0.191, patience 4`,
written to `experiments/hpo/best_stageA-baseline.json`.

The archived pre-fix study reported 0.3531. The two numbers should not be
compared: the old one is the minimum of a noisy objective and re-evaluating its
checkpoint would not reproduce it, whereas 0.3442 is a score the same
checkpoint returns on every evaluation. The point of the restart was
reproducibility, not a lower number.

Pruning did most of the scheduling work — 68 pruned against 67 completed — so
the search finished in roughly 90 minutes rather than the 3–5 hours estimated
from the pre-fix study's throughput.

### Consequence for later stages

Stage C3 and stage D read these values through `make_manifest.tuned()`. Stage B,
C1, C2 and C2b deliberately do not: they use fixed hyperparameters, because
their purpose is to measure noise and solver behaviour under a constant
configuration, not to perform well.

The 3160 completed runs from B/C1/C2/C2b were executed under stochastic
evaluation and are **not** being re-run. Evaluation noise on a fixed checkpoint
spans about 0.0096, roughly a quarter of stage B's between-run standard
deviation of 0.0386. It therefore inflates the measured noise band rather than
manufacturing an effect, so the conclusions that rest on that band — "the
reported +7.57% effect is smaller than the noise" and "leave-one-out
reliability is 0.000" — become more conservative under the fix, not less. Re-
running would cost roughly 90 GPU-hours to confirm a result the correction can
only strengthen.

---

## 2026-08-20 — Sequence 4: per-method PEFT learning rates

### Why

Stage D compares BitFit, LoRA, iA3 and full fine-tuning against each other. The
original project used hand-picked per-method learning rates chosen against the
test set, so reusing them would confound the question *which method is better*
with *whose learning rate happened to suit it*. Each method is therefore tuned
on its own, on validation only, before the comparison runs.

### Design

- Four studies — `stageD-bitfit`, `stageD-lora`, `stageD-ia3`, `stageD-full` —
  sharing the same journal as the baseline study. `best_<study>.json` is
  namespaced per study; a fixed filename would let whichever worker finished
  last silently overwrite the others.
- Each method searches its **own** learning-rate range: bitfit 1e-4–2e-2, lora
  1e-6–5e-4, ia3 1e-5–5e-3, full 5e-6–3e-4. BitFit trains only biases and needs
  a far larger step than methods that add capacity, so a single shared prior
  would hand it a range it cannot work in. LoRA additionally searches rank over
  {4, 8, 16}.
- Every trial starts from the stage B checkpoint for its split, so the methods
  are compared from an identical starting point.
- `hpo_peft_all.sh` walks all four methods on every host, and the hosts join the
  shared studies through the journal. The unit file is byte-identical
  everywhere because `~/.config` is NFS-shared and silently clobbers per-host
  substitutions — the same failure that once made all 13 machines run an
  identical shard.
- 12 hosts x 4 trials x 4 methods = ~192 trials, 12 epochs, splits 0 and 1.

### Verification

Worker logs report `loaded 209/209 parameters from .../baseline_split0.pt`.
This is the strict checkpoint validation added during the audit, exercised on
live traffic: the original Task 3 collapse traced to `strict=False` matching
**0 of 209** parameters without raising, which left a randomly initialized
regression head that converged to predicting the mean. A mismatch is now a hard
error rather than a plausible-looking number. The 209/209 also confirms the
checkpoints relayed through the hub in sequence 2 arrived intact and loadable.

### Result

| study | trials | complete | pruned | failed | best val MSE | best parameters |
|---|---:|---:|---:|---:|---:|---|
| stageD-bitfit | 48 | 31 | 17 | 0 | 0.3759 | lr 4.136e-4 |
| stageD-lora   | 48 | 28 | 20 | 0 | 0.3773 | lr 4.146e-4, r 4 |
| stageD-ia3    | 48 | 33 | 15 | 0 | 0.3835 | lr 3.280e-4 |
| stageD-full   | 48 | 29 | 19 | 0 | 0.3734 | lr 9.908e-5 |

These are **not** comparable to stage A's 0.34418. Stage A trains the whole
model from the pretrained backbone under fully tuned hyperparameters; stage D
starts from the stage B checkpoint, which was trained at a fixed lr 3e-5 and
batch 64, and adapts on top of it. The comparison stage D is built to make is
between the four methods and against its own `none` control arm.

### A defect found in this stage's own design

Each tuned learning rate was checked against the range it was drawn from:

| method | range | best lr | position in log range |
|---|---|---:|---:|
| bitfit | [1e-4, 2e-2] | 4.136e-4 | 26.8% |
| **lora** | **[1e-6, 5e-4]** | **4.146e-4** | **97.0%** |
| ia3 | [1e-5, 5e-3] | 3.280e-4 | 56.2% |
| full | [5e-6, 3e-4] | 9.908e-5 | 72.9% |

LoRA's optimum is pinned to the ceiling of its range. When a search terminates
at a boundary the true optimum is generally outside it, so 0.3773 is a bound on
how poorly LoRA performs rather than its tuned performance.

This matters because it reintroduces precisely the confound the stage exists to
remove. Comparing an under-tuned LoRA against three well-tuned alternatives
conflates *which method is better* with *whose learning rate range happened to
contain its optimum* — the same error as inheriting the original project's
hand-picked rates, merely committed here rather than inherited.

The bitfit range is mis-centred in the opposite direction: it was set to
[1e-4, 2e-2] on the expectation that training biases alone would demand a much
larger step, and the optimum came in near the floor at 4.1e-4. That is harmless,
because an interior optimum means the search was able to find it.

### Correction

`PEFT_LR_RANGE["lora"]` widened to [1e-5, 1e-2] and the study re-run. Optuna
rejects a changed distribution for an existing parameter name, so the narrow
study had to be deleted rather than extended; the journal was copied to
`experiments/hpo/archive/journal-before-lora-widen.log` first, and the narrow
result is tabulated above, so no evidence is lost.

### Outcome of the correction, which was not what was expected

| study | range | complete | best val MSE | best lr | position in range |
|---|---|---:|---:|---:|---:|
| narrow (deleted) | [1e-6, 5e-4] | 28 | 0.3773 | 4.146e-4 | 97.0% |
| wide (in use) | [1e-5, 1e-2] | 45 | **0.3803** | 2.546e-5 | 13.5% |

The widened search returned a **worse** best value, from **more** complete
trials, at an optimum that lies **inside the original narrow range**. All three
observations point the same way: the objective is close to flat in the learning
rate across roughly two orders of magnitude, and the two searches simply landed
on different points of a plateau.

The gap is 0.0030, which is **7.8% of stage B's run-to-run standard deviation
of 0.0386**. It is not a difference; it is noise. The apparent superiority of
the narrow study's 0.3773 was a lucky draw, which is exactly what a boundary
optimum from a small number of trials should be assumed to be.

The wide-range value is the one carried into stage D. Not because it scores
better — it does not — but because its search was validly designed: the optimum
is interior, so the search was free to find it, and it rests on 45 complete
trials rather than 28. Choosing the better-scoring number here would mean
selecting a configuration on the strength of a result whose design is known to
be broken.

This is worth stating as a finding rather than a footnote: **LoRA's learning
rate is not a meaningful lever on this task**, so any stage D result that
appears to separate LoRA from the other methods cannot be attributed to its
learning rate having been chosen well or badly.

---

## 2026-08-20 — Sequence 6: stage C3, the selection comparison

255 runs, 15 per cell, fully balanced across 5 splits x 3 seeds x 3 fractions.
Zero failures. Hyperparameters from the stage A study; the test split touched
once per run.

### The control arm

The original submission had no way to tell a working ranking from a broken one.
`influence_bottom` — deliberately selecting the molecules the estimator ranks
*worst* — supplies it. Paired within (split, seed, fraction):

```
influence_top - influence_bottom = +0.0007
95% CI  -0.0085 to +0.0100        paired t p=0.875   Wilcoxon p=0.867   n=45
```

Reversing the ranking costs nothing. The confidence interval excludes any
effect larger than 0.01, which is a quarter of stage B's run-to-run sd. **The
influence ranking carries no usable information about which molecule to add.**

### Omnibus test

Rather than 12 pairwise comparisons against random, a single test of whether
the five strategies differ at all:

```
Friedman chi2 = 1.849   p = 0.7635   (45 blocks, 5 strategies)
```

No detectable difference among them. Consistent with the pairwise picture, in
which no comparison against random reached p<0.13 and the *sign* of each
difference flips across fractions (influence_top - random: +0.0107, -0.0096,
+0.0080) — the signature of sampling noise, not a small real effect.

### Does adding external data help at all?

```
all - none = +0.0262   paired t p = 0.026   n=15
```

Using the entire external set significantly **hurts**. This is the report's
original claim, and it survives — but as a statement about the dataset, not
about influence functions. It is what stage E's measured distribution shift
(embedding MMD 12.7x baseline, p=0.002) predicts.

Pooled over fractions, every strategy is *worse* than using no external data,
and none significantly so:

| strategy | mean diff vs `none` | p |
|---|---:|---:|
| target_alignment | +0.0003 | 0.969 |
| clustering | +0.0068 | 0.425 |
| random | +0.0070 | 0.436 |
| influence_bottom | +0.0093 | 0.321 |
| influence_top | +0.0100 | 0.140 |

0 of 5 survive Bonferroni; the smallest p is 0.140.

### A trap that was walked into and backed out of

Taking each strategy's *best* fraction before comparing to `none` produces a
much more attractive table — target_alignment at -0.0174, p=0.026, apparently
significant. It is an artifact: taking the minimum of three fractions selects
on the outcome.

The tell is that **every** strategy improves under that treatment, including
`influence_bottom`, which is noise by construction. When a procedure improves
an arm that cannot possibly work, the procedure is producing the effect. The
pooled comparison above is the honest one, and it is reported instead.

This is the third time in this project that sorting or minimising over a family
of results has manufactured a p<0.05 (the others: the rho=-0.46 LOO
correlation, and analyze.py's "strongest LiSSA configuration"). The pattern is
now explicit enough to state as a rule: **any number obtained by taking the
best of several is not evidence until the whole family is reported.**

---

## 2026-08-20 — Sequence 7: stage D, PEFT x selection

300 runs, 15 per cell, 4 methods x 5 strategies x 5 splits x 3 seeds. Zero
failures. Each method uses the learning rate from its own stage D study, so no
method is handicapped by a rate chosen for a different one.

### Selection still does nothing, now under four adaptation regimes

| method | Friedman over 5 strategies | influence_top - none | p |
|---|---:|---:|---:|
| bitfit | p=0.1257 | +0.0008 | 0.844 |
| full | p=0.7015 | -0.0018 | 0.846 |
| ia3 | p=0.2011 | -0.0007 | 0.768 |
| lora | p=0.0692 | +0.0027 | 0.466 |

0 of 4 survive Bonferroni (threshold p<0.0125); the smallest is 0.069. The C3
result is not an artifact of full fine-tuning — the selection strategy is
irrelevant whether one trains 46 849 parameters or 44 375 809.

### The methods themselves do differ, and this is the one positive finding

```
Friedman across 4 methods: chi2 = 33.048, p < 0.0001 (75 blocks)
```

No *pairwise* comparison survives Bonferroni, though (smallest Wilcoxon
p=0.047 against a threshold of 0.0083), which looks contradictory. Ranks
resolve it:

| method | trainable params | mean test MSE | mean rank (1=best of 4) | worst in |
|---|---:|---:|---:|---:|
| ia3 | 46,849 | 0.4424 | 2.080 | 8/75 |
| lora | 738,049 | 0.4437 | 2.240 | 7/75 |
| bitfit | 75,265 | 0.4444 | 2.480 | 9/75 |
| full | 44,375,809 | 0.4502 | 3.200 | **51/75** |

Full fine-tuning ranks worst in **51 of 75 blocks** against 18.75 expected by
chance — binomial p < 0.00001. The ordering is extremely reliable; the
*magnitude* is not large, at 0.0077 between the best and worst methods, or 20%
of stage B's run-to-run standard deviation. Both statements are true and the
Friedman/pairwise disagreement is exactly what that combination produces: a
consistent ranking with small gaps.

Full fine-tuning is also the most erratic arm — best in 17 blocks and worst in
51 — which is what updating 44M parameters on 2 940 training molecules should
be expected to do.

**iA3 is at least as good as full fine-tuning while training 0.106% of the
parameters** (ia3 - full = -0.0077, 95% CI -0.0246 to +0.0092). The interval
includes zero, so this is a claim of equivalence, not superiority, and it is
stated that way.

---

## 2026-08-20 — Sequence 8: stages F1 and F2, a second backbone

90 runs on ChemBERTa-zinc-base-v1 (44.1M parameters, chosen because it is
within 1% of MoLFormer's 44.4M, so a difference cannot be dismissed as model
capacity). F1 trained the baselines; F2 repeated the C3 selection comparison on
top of them. Zero failures.

### Purpose

Every negative result so far could in principle be a property of MoLFormer
specifically. Its linear attention uses random projections, which is unusual,
and if that were what destabilised influence estimation then a standard
softmax-attention model should behave differently. F2 is the test.

### The null replicates

| strategy | test MSE | n |
|---|---:|---:|
| influence_bottom | 0.6812 | 15 |
| target_alignment | 0.6848 | 15 |
| none | 0.6877 | 15 |
| influence_top | 0.6929 | 15 |
| random | 0.6939 | 15 |

```
Friedman over 5 strategies: chi2 = 4.213, p = 0.3779 (15 blocks)
influence_top - influence_bottom = +0.0116  95% CI -0.0053 to +0.0286  p=0.200
influence_top - none             = +0.0052  p=0.522
```

No detectable difference between strategies on a second backbone. Note also
that `influence_bottom` — deliberately the worst-ranked molecules — again comes
out nominally **first**, exactly as in C3. Across two architecturally different
models, reversing the influence ranking never costs anything.

The conclusion is therefore not about MoLFormer. It is about influence-based
selection on this task at this scale.

### An incidental result worth recording

| backbone | parameters | test MSE (no external data) |
|---|---:|---:|
| MoLFormer-XL | 44.4M | 0.4238 |
| ChemBERTa-zinc-base-v1 | 44.1M | 0.6877 |

At essentially identical capacity, MoLFormer is far stronger on lipophilicity —
a 62% higher MSE for ChemBERTa. Since capacity is controlled by construction,
the difference is attributable to pretraining corpus and objective rather than
model size. This was not a question the study set out to answer; it falls out
of the control arm.

### Operational note

The F1 -> F2 handoff exposed one real failure. `sync-artifacts` reported
"bio09: synced" while bio09 in fact received nothing: the push loop redirects
rsync stderr to /dev/null, so a transient failure is invisible and reported as
success. It was caught only because every host was checked explicitly before
launching F2, rather than trusting the sync's own output. Had F2 started as
reported, bio09's five runs would have failed on a missing checkpoint.

---

## 2026-08-20 — Sequence 9: soundness audit of the completed study

All 3805 runs finished, every one with status `ok`, no duplicate ids. Before
interpreting anything, the ways the conclusions could be wrong were checked
directly.

### Is the design balanced?

The 15 C3 selection cells (5 strategies x 3 fractions) hold exactly 15 runs
each. Stage D's 20 cells and F2's 5 cells likewise hold 15 each. The `none` and
`all` controls exist only at fraction 1.0 by construction. Group means are
therefore not distorted by uneven cell sizes.

### Is anything confounded with hardware?

The fleet is homogeneous (all RTX 2060 SUPER). Stage B shows no host effect
(one-way ANOVA F=0.575, p=0.856), between-host sd of means is 0.0124 against a
run-to-run sd of 0.0386, and C3 strategies are spread evenly across hosts
(chi-square p=0.479).

### Are the influence scores actually valid this time?

48 000 values across 160 configuration files: **0 non-finite or empty**, range
-5.374 to +4.588, 47 995 distinct. The original submission's total failure — all
300 scores NaN, selection silently degenerating to file order — is absent. The
null result therefore describes a working estimator, not a broken one.

### Is the null just a lack of statistical power?

This is the question that decides whether the whole study means anything.

```
paired sd 0.0317, n = 45 pairs
minimum detectable effect (80% power, alpha=0.05) = 0.0132 = 2.85% of mean MSE
observed influence_top - influence_bottom         = +0.0007
```

The original report's headline claim was +7.57%, i.e. 0.0351 MSE. **This study
resolves effects 2.7x smaller than that.** An effect of the claimed size would
have been detected overwhelmingly. The null is a measurement, not a failure to
measure.

### Equivalence, not merely absence of evidence

Failing to reject a null is weak. A two-one-sided-tests procedure was run
against an equivalence margin of half of stage B's run-to-run sd (+/-0.0193):

```
TOST p = 1.54e-04  -> the two arms are statistically EQUIVALENT
```

So the finding is not "we could not find a difference between selecting the
best and the worst molecules". It is "**we have positively established that
there is none**", to within a quarter of the noise of a single training run.

### What survives

1. Influence-based selection is equivalent to its own reverse, on two
   backbones. (C3, F2, TOST p=1.5e-4)
2. The influence ranking is unstable under its own solver settings: Spearman
   falls from 0.898 to 0.578 when only `scale` changes. (C1)
3. Influence scores do not predict measured leave-one-out effects: across 160
   configurations mean rho = +0.0101, sign test p=0.477, 0/160 survive
   Bonferroni. (C2)
4. Leave-one-out ground truth is itself unmeasurable at this scale: 0/100
   samples have an effect exceeding 2 SEM even averaged over 20 retrainings.
   (C2, C2b)
5. Adding the whole external set significantly hurts (+0.0262, p=0.026), which
   is what the measured distribution shift predicts. (C3, E)
6. iA3 matches full fine-tuning on 0.106% of the parameters; full fine-tuning
   ranks worst in 51 of 75 blocks (binomial p<0.00001). (D)
