# Data

Nothing under `data/` except this file is committed (see `.gitignore`). Raw recordings of people stay on the private team drive.

## Layout (`docs/03_data_contract.md` section 6)

```
data/
├── raw/{uah_driveset,dmd,own}/     # exactly as downloaded / recorded
├── interim/                        # scratch outputs of loaders
└── processed/{uah,dmd,own}/        # one parquet per trip + manifest.csv + events/meta json
```

The root can be moved with the environment variable `DRIVERISK_DATA_DIR` (see `src/driverisk/utils/paths.py`).

## How to obtain each dataset

| Dataset | Access | Steps | Status |
|---|---|---|---|
| UAH-DriveSet | Public download from the authors' page (see R11 in `docs/01_literature_notes.md`) | Download, unzip into `data/raw/uah_driveset/`, read the licence and readme, then run the inspection script (P0-B1) | Not downloaded |
| DMD | Registration and acceptance of terms at https://dmd.vicomtech.org/ | Register, accept terms, download the subsets listed in `docs/human_tasks/01_datasets.md`, place in `data/raw/dmd/` | Not registered |
| Own recordings | Recorded under `docs/06_data_collection_protocol.md` | Copy from the team drive into `data/raw/own/`; never commit | None yet |

Downloads are never scripted from guessed URLs. A human downloads, then the loaders read local files.

## Dataset card

Filled in P3 (v0.5) and P6 (v1.0): drivers, trips, hours, classes per driver, posed vs real, phone models, known issues.
