# Phase 9 — Report, handoff summary and interview guide

The last phase produces no new modelling. It produces the things that decide whether
a reviewer believes the modelling.

The report had to lead with the loss. Persistence won my five folds at RMSE 0.2123;
on the one-shot holdout a 24-step moving average scored 0.1537 against persistence's
0.1848, a skill score of −0.20. Writing that up first, rather than burying it, turned
out to be the most useful test of the whole protocol: because the selection rule and
the scoring step were frozen in advance, there was no temptation and no ambiguity about
what to report. The number was the number.

The second lesson is the standard error. Across five overlapping origins the spread in
persistence's RMSE is 0.0425 — larger than the gap between almost every pair of
candidates. A fold ranking with that much noise cannot resolve a 0.08 difference, and
pretending otherwise would have been the real methodological failure. The honest
conclusion is not "persistence is best"; it is "this slice cannot distinguish these
forecasters, and the simplest one is the defensible default".

Operationally, the more interesting finding is independent of forecast quality. Mean
utilisation is 9.4% but the 95th-percentile machine sits at 37.5% and 1.3% of readings
exceed 80%. A team provisioning on the mean would be provisioning for a workload that
does not exist. That conclusion survives regardless of which forecaster wins, which is
why the capacity policy is framed as an explicit cost ratio with a sensitivity table
rather than a single recommended number.

The remaining work of this phase is personal preparation, not a deliverable: the
reasoning behind each number has to be explainable from the artifacts on demand, which
is a different exercise from writing them down.
