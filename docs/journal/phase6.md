# Phase 6 - the honest result

The one-shot holdout score came out worse than a moving average. Persistence
scored RMSE 0.1848 against ma24's 0.1537, a skill score of -0.20.

Why this is worth reporting rather than hiding: the training folds said the
opposite (persistence 0.2123, ma24 0.2927). With five overlapping origins the
standard error is 0.0425, wider than the gap between almost every candidate, so
the fold ranking carried much less information than the digits suggested. A
24-step average also happened to land closer to where the holdout drifted
(9.269 versus the 9.250 of the last training day), which is luck, not skill.

What it means operationally: on a stationary, low-variance series with 110
points, the honest deliverable is a well-documented simple baseline with
calibrated uncertainty, not a complex model. The number worth defending is not
0.1848. It is that the protocol predicted this fragility in advance and was
followed anyway.

The post-hoc checks also matter. The calibrated 90% band covered 91.7% of the
holdout with zero misses below, so the uncertainty statement held up even though
the point forecast lost. And no holdout step breached the training peak
threshold of 9.67 pp, so the peak-risk trigger never fired. That is correct
behaviour, but with two hours of data the trigger cannot be claimed as tested.