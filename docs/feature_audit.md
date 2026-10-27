# Feature and column audit

Every column in `vm_cpu_readings.csv.gz` and `vm_metadata.csv.gz` is classified
below. The rule applied: **a feature is valid only if it can be computed at the
forecast origin from information available at or before that origin.**

## `vm_cpu_readings.csv.gz`

| Column | Meaning | Usage | Verdict |
|---|---|---|---|
| `timestamp` | seconds relative to the source trace, 300 s grid | index of the aggregate series; holdout timestamps come from here | **Safe** (timestamps only) |
| `vm_id` | anonymised VM id | groups rows into VMs; count of distinct ids = `active_vm_count` | **Safe** as a grouping key |
| `min_cpu` | minimum CPU in the interval | descriptive only (min/avg relationship) | **Safe** (row is contemporaneous) |
| `max_cpu` | maximum CPU in the interval | optional extra target `aggregate_max_cpu`; descriptive | **Safe** for timestamps <= origin |
| `avg_cpu` | mean CPU in the interval | builds both required targets; lag/rolling features | **Safe** for timestamps <= origin; **banned** for timestamps > origin |

Note: `avg_cpu` is the *target's* source. It is usable as a *lag* feature
(one-step-ahead production setting) but never for the timestamp being predicted.

## `vm_metadata.csv.gz`

| Column | Meaning | Verdict | Reasoning |
|---|---|---|---|
| `vm_id` | join key | Safe | - |
| `subscription_id` | anonymised subscription | Segmentation key only, with care (high cardinality) | Not used as a model feature |
| `deployment_id` | anonymised deployment | Segmentation key only, with care (high cardinality) | Not used as a model feature |
| `timestamp_vm_created` | VM creation time | **Conditional**: allowed only as "age at origin" for VMs created at or before the origin | A VM created after the origin did not exist yet, so counting it is leakage |
| `timestamp_vm_deleted` | VM deletion time | **BANNED** | Encodes the future: knowing a VM dies in the holdout is future information |
| `max_cpu` | lifetime maximum CPU statistic | **BANNED** | Lifetime statistic computed over a window that includes holdout readings |
| `avg_cpu` (metadata) | lifetime average CPU statistic | **BANNED** | Same; and it would trivially explain the target mean |
| `p95_max_cpu` | lifetime P95 max CPU statistic | **BANNED** | Same |
| `vm_category` | Delay-insensitive / Interactive / Unknown | **Safe** (static workload class) | Assignment is fixed at VM creation |
| `virtual_core_bucket` | bucketed vCPU count | **Safe** (static) | - |
| `memory_gb_bucket` | bucketed memory | **Safe** (static) | - |

### Consequences

1. The core `predict` path depends on **`vm_cpu_readings.csv.gz` only**. Metadata
   is optional and skipped if absent; when present it is used for descriptive
   segment analysis only, never as a model feature. This also makes the smoke
   dataset (readings only) valid input.
2. Segment statistics quoted in the report (per-category mean CPU, churn, etc.)
   are **recomputed from training-window readings**, never taken from the
   metadata lifetime statistics.
3. Churn/lifetime analysis on deletion timestamps is descriptive and explicitly
   **censored at the end of the training window**: a VM deleted after the
   training window ends is treated as still alive. Deletion timestamps are used
   to characterise the population, never as a feature.

## Derived features (all inside the training window, at origin `o`)

| Feature family | Definition | Availability at origin |
|---|---|---|
| lags | `y[o]`, `y[o-1]`, `y[o-2]`, `y[o-3]`, `y[o-6]`, `y[o-12]`, `y[o-24]` | known at `o` |
| rolling mean | mean of `y[o-w+1 .. o]`, `w` in {3, 6, 12, 24} | known at `o` |
| rolling std | std of the same windows | known at `o` |
| growth | `y[o] - y[o-3]`, `y[o] - y[o-12]` | known at `o` |
| range | `max(y[o-11..o]) - min(y[o-11..o])` | known at `o` |
| horizon index | `h` in 1..24 | known by construction |
| calendar | hour-of-day / time-of-day sine and cosine | known, but **tested and rejected** (11 h of data) |

Banned by construction: any statistic computed over the *full* series
(e.g. global min/max scaling fitted on all 134 points). All scaling is fitted on
training data only.