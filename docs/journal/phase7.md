# Phase 7 - provisioning on a band, not on a point

The forecast alone is not a capacity decision. What an operator buys is a number
of cores, and the cost of being one core short is not the cost of being one core
long. Making that asymmetry a parameter is the whole point.

At the 8:1 ratio the chosen 60th percentile of the band gives 12 shortfall steps
out of 24 with mean over-provisioning of 0.048 pp. That sounds bad until you
read the sensitivity table: at a 1:1 ratio the optimum moves to the 50th
percentile and net cost halves, while at 32:1 the optimum climbs to the 85th
percentile. There is no universally right margin - there is only the margin your
cost structure implies. Saying that explicitly is more useful than quoting a
single "safe" headroom.

Two things to be honest about. The band was widened 3x because raw residual
quantiles under-covered badly; that factor is calibrated on five origins, so it
is optimistic. And aggregate CPU of 9.4 pp says nothing about the hot VMs: the
cross-VM P95 is 37.5 pp, four times the mean. A capacity plan built only on the
mean would under-provision the tail even when the mean forecast is perfect.
