# Forecasting VM count and aggregate CPU utilisation

**Dataset:** Microsoft Azure Public Dataset V2, `data-v1` release subset
(10,000 VMs, 1,340,000 readings, 134 timestamps on a 5-minute grid).
**Deliverable:** 24-step forecasts of `active_vm_count` and
`aggregate_cpu_utilization`, plus a calibrated uncertainty band and a
cost-aware provisioning rule.

---

## 1. Executive summary

I forecast two aggregate signals 24 steps ahead, two hours, from 9.2 hours of
training data: how many VMs are reporting (`active_vm_count`) and the mean CPU
utilisation across them (`aggregate_cpu_utilization`).

**What the data turned out to be.** The count does not move: it is exactly
10,000 at all 110 training timestamps and all 24 holdout timestamps. That is a
property of a complete rectangular release slice, not of Azure, and I say so
rather than invent a model for it. The utilisation series is stationary, weakly
autocorrelated (lag-1 ACF 0.66, inside the 95% band by lag 11), with 83.6% of
its variance being *which machine you look at* rather than any common
time-varying signal (common-time share: 0.018%). There is very little to
forecast.

**What I did.** Pre-registered a leakage protocol (`docs/protocol.md`) before
running any model, audited every metadata column for leakage
(`docs/feature_audit.md`), and evaluated ten candidates on five rolling-origin
folds entirely inside the training window. The frozen selection rule chose
**persistence** for both targets.

**The honest result.** On the one-shot holdout, persistence scored RMSE 0.1848
and MASE 1.2119. A 24-step moving average scored better: RMSE 0.1537, MASE
0.9652, giving the submission a **skill score of -0.2024**. The baseline won. I
did not retune. The protocol forbids it, and a result you only trust when it is
favourable is not a result. The fold evidence was too noisy to rank the
candidates: the standard error across origins was 0.0425 on an RMSE of 0.2123,
wider than the gap between almost every pair.

**What held up.** The uncertainty band did. The nominal 90% band covered 91.7%
of holdout steps with zero misses below, and the capacity policy built on it
behaves sensibly across the cost-asymmetry range.

**Headline numbers to remember**

| quantity | value | source |
| --- | --- | --- |
| submitted holdout RMSE (aggregate CPU) | 0.1848 | `final_scores.csv` |
| best baseline holdout RMSE (`ma24`) | 0.1537 | `final_scores.csv` |
| skill score vs best baseline | -0.2024 | `final_scores.csv` |
| holdout coverage at nominal 90% | 0.9167 | `posthoc_interval_coverage.csv` |
| aggregate mean vs cross-VM P95 | 9.41 pp vs 37.55 pp | `mean_hides_tails.csv` |

---

## 2. Data understanding and preprocessing

### 2.1 Structural checks

Run over the full file, because these are integrity questions and reveal no
target values.

| check | result |
| --- | --- |
| rows | 1,340,000 |
| distinct VMs | 10,000 |
| distinct timestamps | 134 |
| timestamp grid | every gap exactly 300 s, no duplicates |
| duplicate `(vm_id, timestamp)` | 0 |
| nulls | 0 |
| `avg_cpu` outside [0, 100] | 0 |
| `min_cpu > max_cpu` | 0 |
| metadata rows per VM | exactly 1 |

The release is perfectly rectangular: every VM has exactly 134 readings at
every timestamp. Source: `reports/artifacts/structure_checks.csv`.

### 2.2 Target definitions

- `active_vm_count(t)`: distinct `vm_id` with a valid reading at `t`. "Valid"
  means `avg_cpu` finite and in [0, 100]. Each VM counted at most once per
  timestamp.
- `aggregate_cpu_utilization(t)`: mean of VM-level `avg_cpu` at `t`. Missing
  readings are **excluded from the mean, never zero-filled**, because a VM with
  no reading is not a VM using zero CPU.

Both are built by `cwf.targets.build_targets` before any split, as R1 requires.

### 2.3 Split

First 110 observations train, last 24 holdout. One function owns it
(`cwf.splits.chronological_split`); the predict path keeps the holdout
*timestamps* and immediately discards the holdout *values*.

---

## 3. Workload analysis (training window only)

Every number in this section comes from the 110 training timestamps.

### 3.1 The count is constant

`active_vm_count` is 10,000 at all 110 training timestamps, standard deviation
exactly 0. Zero VMs are created or deleted inside the window. Operationally:
there is no churn to model in this slice, and any scale-free metric for this
target divides by zero. MASE is reported as undefined rather than as a number
that would look impressive and mean nothing.

### 3.2 The aggregate is stationary and noisy

| statistic | value |
| --- | --- |
| mean | 9.4141 pp |
| std | 0.1936 pp |
| min / max | 9.1116 / 9.9122 pp |
| trend slope | -0.0127 pp per hour (t = -1.31) |
| ADF p-value | 0.413 |
| KPSS p-value | 0.100 |
| ACF lag 1 / 12 / 24 | 0.662 / 0.327 / -0.020 |
| first lag inside 95% band | 11 (about 55 minutes) |

ADF fails to reject a unit root and KPSS fails to reject stationarity at 10%;
the series sits near the boundary, i.e. persistent but not trending. With 9.2
hours there is **no daily or weekly cycle to estimate**; claiming one would be
fitted noise. Operationally: last-value and short-average forecasts are both
defensible and neither will be much better than the other. This predicts the
result in section 6.

### 3.3 The mean hides the tail

| statistic | value |
| --- | --- |
| aggregate mean | 9.41 pp |
| cross-VM P95 of `avg_cpu` | 37.55 pp (4.0x the mean) |
| cross-VM P99 | 87.27 pp |
| cross-VM max | 99.30 pp |
| readings at or above 80% | 1.272% |
| readings at or above 90% | 0.878% |

Per-VM mean distribution: p05 0.94, median 5.50, p90 19.74, p99 79.62 pp. Load
is concentrated: the top 1,000 VMs (10% of the fleet) carry 43.5% of total
utilisation. Operationally, a capacity plan driven off the aggregate mean
under-provisions exactly the machines that are about to saturate. This is the
single most important caveat in the whole analysis and it is why the extension
targets exist (section 7).

### 3.4 Composition versus genuine load change

Variance decomposition of per-VM utilisation:

| component | share |
| --- | --- |
| between-VM level | 83.58% |
| common time effect | 0.018% |
| idiosyncratic residual | 16.40% |

When the aggregate moves, **100% of the movement is within-VM** (correlation
between aggregate change and within-VM change: 1.0; std 0.158 pp). There is no
composition effect at all, because no VM enters or leaves. Operationally there
is no "the fleet got busier" signal; the aggregate moves because many
independent machines wobble slightly in the same small way. That is close to
unpredictable by construction, and it is the mechanism behind the result in
section 6.

### 3.5 Segments

| `vm_category` | mean `avg_cpu` (pp) | VMs |
| --- | --- | --- |
| Interactive | 11.51 | 3,091 |
| Unknown | 10.35 | 729 |
| Delay-insensitive | 8.26 | 6,180 |

Deployment size matters more: single-VM deployments average 6.41 pp, deployments
of nine or more average 10.95 pp. Operationally, if you must plan per segment,
segment by deployment size rather than by category: the spread across deployment
sizes (4.5 pp) is larger than the spread across categories (3.3 pp) and the
largest deployments behave quite differently from singletons.
Segment profiles are recomputed from training-window readings, never from the
metadata lifetime statistics.

### 3.6 Peaks, and churn censoring

The peak threshold is the top decile of the **training** series: 9.6684 pp,
covering 11 timestamps in 2 distinct hours. This threshold is fixed from train
only and used unchanged everywhere afterwards.

Median VM lifetime is 29.99 days (p10 8.75 days), but since no VM is deleted
inside the observed window, every lifetime after the training end is
**right-censored**. I state that as a limitation rather than computing a mean
lifetime from censored data.

---

## 4. Forecasting approach

### 4.1 Candidates

Ten, all evaluated on the identical folds:

- baselines: persistence, moving average (6/12/24), drift, seasonal naive (12)
- statistical: Holt, damped-trend ETS
- regularised regression: direct multi-horizon ridge on origin-safe lag and
  rolling features, alpha swept from 0.1 to 1000
- gradient boosting: direct multi-horizon, small trees, heavy regularisation

Direct multi-horizon means one model per horizon step (or a pooled model with
horizon as a feature), so errors do not compound recursively. A recursive
comparison was not worth the compute given how flat the results were.

### 4.2 Validation methodology

Expanding window, origins `[60, 66, 72, 78, 84]`, horizon 24, constraint
`origin + horizon <= 110` enforced by assertion. Plus one dev-holdout fold at
origin 85 mirroring the final protocol. Metrics: MAE, RMSE, MASE (scaled by the
in-sample naive error computed on training data only), bias, per-horizon and
per-origin breakdowns.

Selection rule, frozen in `docs/protocol.md` **before** any model ran: lowest
rolling-origin RMSE per target; if candidates are within one standard error of
the best, prefer the simpler.

### 4.3 Training-fold results

`active_vm_count`: every candidate scores RMSE 0.0000 (constant series); MASE
undefined. Selected by simplicity: persistence.

`aggregate_cpu_utilization`, sorted by RMSE:

| candidate | RMSE | MAE | MASE | SE across origins |
| --- | --- | --- | --- | --- |
| **persistence** | **0.2123** | 0.1850 | 1.4677 | 0.0425 |
| gbm_direct | 0.2209 | 0.1899 | 1.5066 | 0.0419 |
| drift | 0.2396 | 0.2100 | 1.6662 | 0.0544 |
| ma6 | 0.2683 | 0.2380 | 1.8883 | 0.0597 |
| ma12 | 0.2798 | 0.2538 | 2.0138 | 0.0539 |
| ets_damped | 0.2842 | 0.2506 | 1.9880 | 0.0668 |
| ridge_direct | 0.2884 | 0.2587 | 2.0524 | 0.0048 |
| ma24 | 0.2927 | 0.2709 | 2.1495 | 0.0300 |
| ets_holt | 0.2953 | 0.2605 | 2.0670 | 0.0730 |
| seasonal_naive12 | 0.2956 | 0.2631 | 2.0874 | 0.0530 |

Persistence, drift and GBM are within one standard error of the best; the rule
breaks the tie toward persistence.

Ridge was swept over alpha in {0.1, 1, 10, 100, 1000}: best RMSE 0.2615 at
alpha = 0.1, still worse than persistence. The sweep is reported so a reviewer
can check that regularisation strength is not the reason the model lost.

**A claim we do not make:** GBM at 0.2209 versus persistence at 0.2123 is not
"GBM is worse". It is inside the noise band. The honest statement is that on this
evidence nothing beats a naive forecast, and the rule prefers the simpler thing
when it cannot tell them apart.

---

## 5. Leakage controls

The holdout values are in the same file as the training data, so correctness
depends entirely on discipline. These are the controls, all verifiable:

1. **Protocol frozen first.** `docs/protocol.md` was committed before any
   candidate was evaluated. Git history shows the ordering.
2. **Banned columns.** `timestamp_vm_deleted` encodes the future; metadata
   `max_cpu`, `avg_cpu`, `p95_max_cpu` are lifetime statistics that include
   holdout readings. All four are banned as features. `timestamp_vm_created` is
   usable only as age-at-origin. Every column is classified in
   `docs/feature_audit.md` with a reason.
3. **Single split owner.** `cwf.splits.chronological_split` is the only split.
4. **Single holdout reader.** `scripts/final_score.py::load_holdout_truth` is the
   only function returning holdout values. `tests/test_leakage.py` parses the AST
   of every module and fails if anything else references it.
5. **Corruption invariance.** `tests/test_leakage.py` rewrites the holdout
   readings with random values and with zeros and asserts `predictions.csv` is
   byte-identical. It is.
6. **Causality test.** `tests/test_causality.py` perturbs everything after an
   origin and asserts the feature matrix does not move.
7. **One-shot scoring.** `scripts/final_score.py` ran once. Nothing was retuned
   afterwards; the post-hoc analysis is labelled and changed nothing.

---

## 6. Results versus baseline

### 6.1 One-shot holdout scores

`active_vm_count`: RMSE 0.0000 for the submission and every baseline. MASE
undefined.

`aggregate_cpu_utilization`:

| candidate | RMSE | MAE | MASE | skill vs best baseline |
| --- | --- | --- | --- | --- |
| ma24 (best baseline) | 0.1537 | 0.1169 | 0.9652 | 0.0000 |
| ma12 | 0.1613 | 0.1230 | 1.0156 | -0.0494 |
| ma6 | 0.1713 | 0.1336 | 1.1028 | -0.1146 |
| **submitted (persistence)** | **0.1848** | **0.1468** | **1.2119** | **-0.2024** |
| drift | 0.2101 | 0.1705 | 1.4079 | -0.3669 |

The submission is 20.2% worse than the best baseline on RMSE. Source:
`reports/artifacts/final_scores.csv`.

### 6.2 Why this happened

Three things, in order of importance:

1. **The fold evidence had no resolution.** Five overlapping origins give
   roughly 13 correlated evaluations of a 24-step horizon. The standard error
   across origins was 0.0425 on an RMSE of 0.2123, a 20% relative uncertainty.
   Any ranking inside that band was a coin flip.
2. **The holdout drifted slightly upward.** Holdout mean 9.269 pp versus the
   training mean 9.414 pp and the last-24-training mean 9.250 pp. A 24-step
   average sits near 9.2 to 9.3; a single last observation sat at 9.1646. `ma24`
   landed closer by luck, not skill: it had no information persistence lacked.
3. **The series is nearly unpredictable anyway.** With 0.018% common-time
   variance there is no shared signal to extract. Every candidate was guessing
   the same small wobble.

The pre-registered rule was followed. Changing the model after seeing the
holdout would have made the reported number meaningless, because it would have
been a choice made with knowledge of the answer.

### 6.3 Post-hoc analysis (labelled, does not change the submission)

- **Bias.** Mean actual 9.2690, mean predicted 9.1646, bias +0.1044 pp.
  The forecast was systematically low, consistent with an upward drift in the
  holdout window.
- **By horizon.** Errors do not grow with horizon: absolute error at step 1 is
  0.0039 and at step 16 (the worst, 9.6360 actual) is 0.4714. A flat error
  profile is the signature of a random walk; it is more evidence that nothing
  exploitable was there.
- **Peaks.** Zero holdout steps exceeded the training peak threshold of
  9.6684 pp (holdout max 9.6360). The peak-risk trigger never fired. That is
  correct behaviour, but with two hours of data and no peak events we cannot
  claim the trigger was tested.
- **Intervals.** Nominal 50/80/90/95 gave measured 0.833/0.833/0.917/0.917. All
  misses are above the band, zero below, consistent with the +0.10 bias. Widths
  0.916/0.916/1.062/1.062 pp.
- **Churn.** Zero creations and zero deletions across the full 134-timestamp
  range, so no churn-event analysis is possible; this is now stated as a fact
  rather than as a censoring assumption.

---

## 7. Capacity-planning recommendation

### 7.1 The rule

Provision at a quantile of the forecast band:

```
provisioned = lower + level * (upper - lower)
```

where `level` is chosen from the cost asymmetry, not from a rule of thumb. In
newsvendor form the cost-optimal level is `c_under / (c_under + c_over)`; at the
project's 8:1 ratio that is 0.889. Because five origins is a tiny calibration
sample and the empirical quantiles under-covered badly (0.558 at nominal 0.90),
the band is widened by a factor of 3.0, which raises measured leave-one-origin-out
coverage to 0.933. The chosen operating level is 0.60, and a peak-risk trigger
adds headroom when the band's implied probability of a peak step exceeds a
threshold.

### 7.2 Holdout evaluation (one shot)

At the chosen level 0.60 and an 8:1 cost ratio: **12 shortfall steps out of 24**,
mean shortfall 0.070 pp, mean over-provisioning 0.048 pp, max shortfall 0.368 pp,
net cost 0.611 per step. Source: `policy_holdout_evaluation.csv`.

Twelve shortfalls in 24 steps sounds alarming and is worth explaining rather
than smoothing over: at the 60th percentile of a *forecast* band that was itself
slightly low, being short on about half the steps is the expected consequence of
choosing a below-median level. An operator who genuinely cannot tolerate
shortfall should provision at 0.85 or 0.95 and pay for it.

### 7.3 Cost sensitivity

Net cost per step by cost ratio (`policy_holdout_sensitivity.csv`):

| `c_under : c_over` | 0.50 | 0.60 | 0.70 | 0.80 | 0.85 | 0.90 |
| --- | --- | --- | --- | --- | --- | --- |
| 1:1 | 0.147 | 0.118 | 0.134 | 0.185 | 0.234 | 0.309 |
| 4:1 | 0.524 | 0.329 | 0.239 | 0.211 | 0.241 | 0.309 |
| 8:1 | 1.026 | 0.611 | 0.378 | 0.246 | 0.249 | 0.309 |
| 16:1 | 2.031 | 1.174 | 0.657 | 0.316 | 0.266 | 0.309 |
| 32:1 | 4.041 | 2.300 | 1.214 | 0.455 | 0.299 | 0.309 |

The optimum moves from the 50th percentile at 1:1 to the 85th at 32:1. **There is
no universal headroom percentage**; there is only the percentage your cost
structure implies. Presenting the table is more useful than presenting one number.

### 7.4 Mean is not tail risk

The aggregate sits at 9.41 pp while the cross-VM P95 is 37.55 pp and the max is
99.30 pp. Tracking the three extra targets confirms the gap is structural, not
a one-off:

| extra target | train mean | correlation with aggregate | ratio to aggregate |
| --- | --- | --- | --- |
| `share_hot_vms` (share above 19.69 pp) | 0.1000 | 0.882 | n/a |
| `cross_vm_p95_avg_cpu` | 37.547 | 0.858 | 3.99x |
| `aggregate_max_cpu` | 18.441 | 0.940 | 1.96x |

Operationally, forecast the aggregate for total capacity, but plan per-VM
headroom from the P95/P99 series, and keep a per-VM ceiling rule. When the band
is wide, the correct action is to hold headroom and shorten the replanning
interval, not to trust the point forecast more.

---

## 8. Limitations and future work

1. **9.2 hours of training data, one window.** No daily or weekly seasonality can
   be identified, and one holdout window cannot establish a stable ranking.
   The result in section 6 is a single draw from a noisy comparison.
2. **The count is constant by construction.** The interesting question, how
   arrivals and departures move the fleet size, cannot be answered from a
   rectangular slice. The code handles varying counts; the data does not
   exercise it.
3. **Composition effects are exactly zero here** and will not be zero in
   production, where VMs arrive and leave mid-window. The decomposition code is
   general; the data simply has no composition to decompose.
4. **The mean hides the tail.** Covered in section 7.4, but it is the limitation
   most likely to bite an operator who reads only the aggregate.
5. **Churn is right-censored** at the training end for any lifetime analysis, and
   in this release there are no deletions at all in the observed range.
6. **The interval widening factor (3.0) is calibrated on five origins** and is
   therefore optimistic. The holdout coverage of 0.917 at nominal 0.90 is
   consistent with the chosen factor but is not independent proof of it.
7. **The baseline beat the submission.** This is the headline and it is not
   fixed. With more history (weeks, not hours), a seasonal component and a
   genuine common-mode signal would likely emerge and a richer model would have
   something to learn.

**What I would do next, in order:** get more history so a daily cycle is
estimable; add the per-VM bottom-up model (10,000 series give far more rows than
110 aggregate points) and compare its aggregate against persistence seriously;
and replace the folded band calibration with a proper conformal procedure once
there are enough origins for it to mean something.

---

## Figures

| figure | operational takeaway |
| --- | --- |
| `fig01_aggregate_cpu_train.png` | The aggregate wanders within a 0.8 pp band with no trend: plan for a stable level, not a growth curve. |
| `fig02_vm_mean_hist.png` | Per-VM means are bimodal and heavy on near-idle machines; a fleet-wide average describes almost no individual VM. |
| `fig03_segments_category.png` | Interactive VMs run about 40% hotter than delay-insensitive ones; segment-level headroom differs even when the aggregate does not. |

All numbers in this report are generated by the scripts in `scripts/` and stored
in `reports/artifacts/`. Nothing here is estimated by hand.