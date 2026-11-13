# 01. Datasets: registration, download, verification (H1, H10)

Claude does not download datasets by guessing URLs. You download; then the scripts read the local files.

## A. UAH-DriveSet (about 40 trips, 6 drivers, per the paper)

- [ ] Open the dataset page given in the paper: `http://www.robesafe.com/personal/eduardo.romera/uah-driveset` (printed in R11; if the link is dead, search for "UAH-DriveSet").
- [ ] **Read the licence / terms and the dataset readme.** Write the citation requirement into `data/README.md`.
- [ ] Download and unzip into `data/raw/uah_driveset/` (git-ignored). Keep the driver folders as they are.
- [ ] Run: `uv run python scripts/inspect_uah.py data/raw/uah_driveset`
- [ ] Compare the output with the table below. Anything that differs is a finding: tell Claude, do not "fix" the data.

| Check | Paper says (R11) | Your output |
|---|---|---|
| Drivers | D1-D6 | |
| Trips | 3 motorway + 4 secondary per driver (D6: 3 + 2), about 40 | |
| Behaviours | normal, drowsy, aggressive | |
| Accelerometer rate | 10 Hz, 12 columns incl. timestamp | |
| GPS rate / columns | 1 Hz, 9 columns | |
| Folder name format parsed | all folders | |
| Video | one file per trip, road view | |
| Events file | only above 50 km/h | |
| Distraction label | none | |

- [ ] Tick the "Verified" boxes in `docs/03_data_contract.md` section 3.1 **only** for rows you confirmed, in a PR or by telling Claude.
- [ ] Also open one video and confirm it shows the road, not the driver's face.

## B. DMD (Driver Monitoring Dataset)

- [ ] Go to https://dmd.vicomtech.org/ , register, accept the terms. Approval may take days: do this first.
- [ ] Read the licence. Note what is allowed (academic use, redistribution of features, publication of example frames).
- [ ] Find which subsets exist (face camera view, body, hands; RGB / depth / IR; real driving vs simulator).
- [ ] Download the **face-camera RGB** videos and their annotation files first, for a few subjects. Do not download everything before the label vocabulary is checked.
- [ ] Put them in `data/raw/dmd/`. Write down: number of subjects, label names, annotation file format, video fps. Send this to Claude; the DMD inspection script is written after seeing the real annotation files.
- [ ] Confirm there are **no vehicle motion signals** (contract section 3.2).

## C. Missing papers (H10)

Use the college IEEE access to download, into `docs/Research-papers/` (git-ignored): R6 Tan 2024, R7 Kuang 2026, **R9 Chen 2023 (closest prior work)**, R10 DMD (Ortega 2020), and Streiffer et al. (camera + phone IMU, cited in R2). Then read R9 and fill the difference table in `docs/01_literature_notes.md`.

## D. Storage

Large downloads stay outside git. If the disk is small, set `DRIVERISK_DATA_DIR` to another drive.
