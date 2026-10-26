# Phase 8 - packaging, or: can a stranger reproduce this?

Three things had to become true before this counted as a submission.

First, one command. python -m cwf.predict takes a data directory and an output
path, infers the cadence and the holdout timestamps from whatever it is given,
and writes three columns. Nothing about 10,000, 134 or 110 is hard-coded.

Second, the tests have to be the argument, not a formality. The causality test
perturbs everything after an origin and asserts the features do not move. The
leakage test parses the AST of every module and fails if anything but
inal_score.py references the holdout loader. The corruption test rewrites the
holdout readings with random numbers and with zeros and asserts the output CSV is
byte-identical - if any holdout value were sneaking into the model, that test
would catch it. Sixty-one tests, all green.

Third, a smoke dataset. A reviewer swapping in three VMs and forty timestamps
must get a valid file, not a traceback. Four synthetic datasets cover varying
counts, missing readings, a short history, a constant series and a gap in the
timestamp grid. Two of my own guesses about the API turned out to be wrong while
writing these tests, which is the tests doing their job.

Packaging also forced a correction: the Makefile I first wrote referenced flags
that do not exist. Reading the actual argparse definitions took two minutes and
saved a reviewer a confusing failure.
