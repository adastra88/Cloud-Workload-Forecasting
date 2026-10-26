# Phase 1 journal — data understanding (training window only)

Everything below was computed on the **110 training timestamps only** (0 … 32700 s, ≈9.2 h).
No holdout target value was printed, plotted or summarised. Artifacts in `reports/artifacts/`.

1. **Structure is clean and rectangular.** 10,000 VMs × 134 readings = 1,340,000 rows, every gap a
   perfect 300 s, no nulls, no duplicate `(vm_id, timestamp)`, no `avg_cpu` outside [0,100], and
   `min_cpu <= max_cpu` everywhere (`structure_checks.csv`). Operationally this means the release is a
   curated slice, not a raw trace: I cannot study churn, gaps, or late-registering VMs from it.
2. **`active_vm_count` is exactly constant at 10,000** across the training window
   (`target_summary_train.csv`). Every VM is present at every timestamp. I forecast it with persistence
   and spend no modelling effort there — the pipeline still handles varying counts (smoke dataset).
3. **No churn in-window.** 100% of VMs were created before the training end and 0% deleted within it
   (`lifetime_summary.csv`, `churn_by_category.csv`). Deletions are all **censored** past the window, so
   any lifetime statement is descriptive only.
4. **Aggregate CPU barely moves.** Mean 9.41 pp, std 0.19 pp, range 9.11–9.91 pp
   (`target_summary_train.csv`). A ~8% coefficient of variation over 9 hours — the "forecasting problem"
   is tighter than the headline suggests.
5. **No trend, no stationarity violation.** Slope −0.013 pp/h (t = −1.31), ADF p = 0.41, KPSS p = 0.10
   (`stationarity_trend.csv`). Nothing to extrapolate.
6. **Short memory.** ACF lag-1 = 0.66, lag-12 = 0.33, lag-24 = −0.02; it first falls inside the 95%
   band at lag 11 (≈55 min, `acf_train.csv`, `acf_decay.csv`). With 9 hours of data I explicitly do not
   claim daily or weekly seasonality — a 24-step (2 h) persistence baseline will be hard to beat.
7. **Levels are between-VM; movements are within-VM.** 83.6% of variance is between-VM level
   differences, only 0.018% is a common time component (`composition_vs_load.csv`). Because all VMs are
   present at every timestamp, composition effects are exactly zero and aggregate movement is 100%
   genuine within-VM load change (corr 1.00, `movement_decomposition.csv`).
8. **The mean hides the tail.** Aggregate mean 9.4 pp vs cross-VM P95 37.5 pp (4.0×) vs P99 87.3 pp;
   1.27% of readings are ≥80% and 0.88% ≥90% (`mean_hides_tails.csv`). Per-VM means: p05 0.94, median
   5.50, p90 19.74, p99 79.62 — 46% of VMs average under 5% while 3.1% ever reach 90%. Provisioning on
   the mean would starve the hot VMs, so the capacity story must use the tail/band, not the point forecast.
9. **Load is concentrated.** Top 10 VMs carry 1.0% of CPU, top 100 → 9.6%, top 500 → 30%, top 1000 →
   43.5% (`load_concentration.csv`). A small number of VMs dominate, so host-level decisions on those
   VMs matter far more than the fleet average.
10. **Segments differ but modestly.** Interactive 11.5 pp mean vs Delay-insensitive 8.3 pp
    (`segment_profiles_train.csv`); deployment size correlates with utilisation (9+ VM deployments
    average 10.9 pp vs 6.4 pp for singletons, `deployment_size_segments.csv`). Recomputing from readings,
    never from the banned metadata lifetime statistics.
11. **Peak definition fixed from train only:** top-decile threshold 9.67 pp, 11 timestamps ≈ 2 distinct
    hours touched (`peak_definitions_train.csv`). "Peak" is a within-window construct, not a daily cycle.
12. **Modelling consequence.** Expect a persistence-level baseline to be competitive; the honest win, if
    any, is small and must be shown against fold spread. The value I can add is the uncertainty band and
    the provisioning policy, not a large accuracy gain.
