# Phase 0 journal - setup and protocol freeze

What I did: read every upstream document, downloaded the fixed `data-v1`
release, checked the asset hash and the three file checksums, and confirmed the
data's structure (schema, grid, duplicates, ranges) without looking at any
target values.

What matters operationally: the release is a *curated* population. 10,000 VMs,
every one of them present at all 134 timestamps, five-minute spacing, no gaps, no
duplicates. That makes the "churn" analysis descriptive rather than predictive:
the release deliberately removed VMs that did not have a complete trace. In a
real fleet the active-VM count would move every 5 minutes; here it is expected to
be flat, and that needs saying plainly instead of inventing a model for it.

The other operational point: because the holdout values live in the same file, the
integrity of the whole exercise rests on process. Writing `protocol.md` first
(selection rule, metrics, folds, candidate list) before running a single model is
what makes the later numbers trustworthy. It is committed ahead of any result, so
git history shows the order.

Next: training-window-only EDA.