"""Cloud workload forecasting package.

Modules
-------
data      : loading + structural validation of the challenge release
targets   : building ``active_vm_count`` / ``aggregate_cpu_utilization`` series
splits    : the single chronological 110/24 split function
features  : origin-safe feature construction
baselines : naive / moving-average / drift baselines
models    : candidate forecasting models
evaluate  : metrics + rolling-origin evaluator
intervals : empirical residual quantile bands
policy    : capacity provisioning rule
predict   : the CLI that writes predictions.csv
"""

__version__ = "1.0.0"
