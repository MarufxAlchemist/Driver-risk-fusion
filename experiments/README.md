# Experiments

Every training or evaluation run is logged with `driverisk.utils.experiment.start_run`:

```python
from driverisk.utils.experiment import start_run

with start_run("toy-fusion", cfg, seed=0, setting="stub", split_file="splits.json") as run:
    ...
    run.log_metrics({"macro_f1": 0.71, "fa_per_hour": 4.2})
```

| Path | In git? | Content |
|---|---|---|
| `results.csv` | yes | one row per run: id, time, name, seed, commit, setting, status, headline metrics, split hash, duration |
| `runs/<run_id>/config.yaml` | no | the resolved config of the run |
| `runs/<run_id>/meta.json` | no | commit (`+dirty` if uncommitted changes), seed, library versions, device, split-file hash, status |
| `runs/<run_id>/metrics.json` | no | every metric the run logged |
| `ablations/A<n>.md` | yes | one file per ablation: table, plot, two-line interpretation (`docs/04_experiment_protocol.md` section 5) |

`<run_id>` = `<yyyymmdd-HHMMSS>_<name>_s<seed>_<commit>`. A run that raises is recorded with status `failed`.

Rules (`docs/04_experiment_protocol.md` section 8): config committed, commit hash recorded, seeds fixed with `driverisk.utils.seed.set_seed`, splits from `manifest.csv` with the split-file hash, three seeds for any number that goes in a table. Never commit checkpoints.
