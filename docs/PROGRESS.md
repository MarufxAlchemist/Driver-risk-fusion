# Progress log

Running log for the solo build (master prompt section 3.1). Newest entries on top within each section. Numbers are from commands run in this repo; nothing is estimated unless it says so.

---

## Decisions taken after the Phase 0 gate review (2026-11-13)

| # | Decision | What changed |
|---|---|---|
| 1 | **A10 and A11 approved** (WebSocket `?token=` auth, `recovered` message) | `docs/05` sections 1, 3.6 and 3.8 now describe them; the "pending approval" comments are gone from the code |
| 2 | **ADR-0007 part 2 approved: event/state alert taxonomy** | Alert `type` = 6 event names + `aggressive`, `drowsy`, `distracted`. `aggressive_accel` is now `harsh_accel`. Predictive alerts only for event types (`Alert` validator). The scripted alert is a predictive `micro_sleep`. OpenAPI snapshot regenerated; 3 new tests pin the `Literal` to `schema.EVENT_TYPES + schema.CLASSES[1:]`. ADR-0007 is **Accepted** |
| 3 | **Contract freeze (P0-C2)** | Still blocked on the UAH and DMD inspection. `contract_version` stays `"0.1"` |
| 4 | **History rewrite: leave for now** | Recorded as risk R21 with the contingency. No rewrite, no force-push |
| 5 | **Pending doc edits: approved, all, on branch `docs/c-p0-edits`** | Done in six commits, one per document group: `05`; `03`; `02` + ADR-0002; ADR-0003/0007 + index; `01`; CONTRIBUTING (new section 0, solo mode) + risk register R21/R22 |

Verification after these changes: `mingw32-make lint`-equivalent (`ruff check` and `ruff format --check` on `src tests scripts`) clean; `pytest -m "not slow"` **102 passed** (99 before, plus 3 taxonomy tests); `scripts/demo_stub.py` still works and now raises a state alert (`distracted` in the last run).

Things I want you to know about these edits:
- **`03` documents the rules the code follows, which differ slightly from the wording of fix F10.** For `y_risk` the code (assumption A7, tested) sets -1 when an uncertain event *starts* inside the horizon; F10 said "overlaps". I documented the code's rule in `03` section 7.1.
- **`03` marks as "planned, not implemented"** the manifest column `quality` and the meta keys `posed_segments`, `parked_segments`, `ear_threshold`, `ear_threshold_source` (F20, F9). They need the own-data pipeline (P1-D). The validator does not check them yet.
- **`03` section 4.4 (UAH conversion rules) comes from the UAH paper, not from the files.** It stays unverified until `inspect_uah.py` runs on the real download.
- **F21 added R12 (Ultra-Fusion) with an incomplete citation:** the repo's README only records "Tian et al." and the arXiv number. The PDFs are not in the working folder any more, so I could not copy the title. Please fill it in.
- **CONTRIBUTING solo mode says "merge with `--no-ff`", not "squash"** (F23 proposed squash), because that is how every branch so far was merged and it keeps small commits visible. Tell me if you want squash.
- Not done from the approved list: F13-F17 and F24 (protocol and personalization docs `02`/`04`), F18 (task list `07`), F27 (README licence line), the own-file naming rule in `06` (part of F11). They were not in the approved set.

---

## Phase 0 gate report (2026-11-13)

**Status: code complete, gate check passed locally; open items below need you.**

### Gate check (`00_PROJECT_PLAN.md` section 3 / `07` P0 gate)
Gate text: "make demo-stub runs stub data -> stub features -> toy fusion -> mock API -> dashboard shell."

| Check | Command | Result |
|---|---|---|
| Lint + format | `mingw32-make lint` | passed, 47 files |
| Tests | `uv run pytest -m "not slow"` | **99 passed**, 1 warning (see Surprises), about 107 s |
| Stub pipeline | `mingw32-make demo-stub` | OK: loss 1.454 -> 0.307, alert round-trips through the API |
| Windows shim | `.\make.ps1 lint` / unknown target | passes / rejects with a clear message |
| Real server | `scripts/mock_server.py` on a socket | health, 401 without key, dashboard HTML (200), WebSocket delivered the 10 scripted messages in order over 10.4 s |
| Contract vs code | `tests/contract/test_schema.py` | feature tables in `03` parsed and equal to `schema.py` (16 vision, 14 motion) |
| API vs snapshot | `tests/contract/test_api.py` | OpenAPI equals `docs/openapi_snapshot.json` |
| Clean clone of `dev` | `git clone` -> `uv sync --locked` -> lint -> pytest -> demo | sync 45 s (warm uv cache; a cold download of PyTorch will be slower), lint clean, **99 passed** (168 s), demo identical to the in-place run |
| Stubs vs validator | `tests/contract/test_validate.py` + both stub scripts | all stub trips pass; every rule in contract section 10 has a test that it catches its violation (columns, dtype, time step, range, `inf`, 0/1 flags, undeclared all-NaN, labels, version, mask rule, meta, split leakage, missing driver, events, orphan files) |

**Not verified, and why:**
- **The dashboard page was never opened in a browser** (no browser available to me). It is served and its WebSocket target works (checked with a socket client), but rendering and the gauge/alert display are untested. Please open `make serve` -> http://127.0.0.1:8000/dashboard/ and press "Start demo trip".
- **Mac was not tested.** CI (lint, tests, stub demo) passed on GitHub for `dev` at the `v0.0` commit on ubuntu-latest and windows-latest (run 43440961188).
- **The Colab notebook has not been run** on Colab/Kaggle.
- `make setup` itself (which also installs pre-commit hooks) was not run; the clean clone used `uv sync --locked` directly. Local runs were Windows only; CI added Linux.

### P0 tasks

| Task | State | Notes |
|---|---|---|
| P0-D2 scaffold | done | `feat/d-scaffold`; uv + Python 3.11.18, `uv.lock`, Makefile + `make.ps1`, pre-commit config, `.gitattributes`, `data/README.md` |
| P0-D3 CI | done | `.github/workflows/ci.yml`; green on ubuntu + windows after the first push of `dev` |
| P0-D4 docs into `docs/` | done | `docs/d-relayout`; ADRs in `docs/decisions/`; PDFs untracked |
| P0-C3 validator + WindowDataset | done | `schema.py`, `validate.py`, `splits.py`, `normalize.py`, `datasets.py` |
| P0-A2 / P0-B2 stubs | done | `tests/fixtures/make_stub_vision.py`, `make_stub_motion.py`; generator in `driverisk/data/stubs.py` |
| P0-C4 toy fusion | done | `models/fusion.py`, `training/loop.py`; masked-value invariance and loss-drop tests |
| P0-D7 mock API | done | `serving/schemas.py`, `mock_app.py`, `scripted.py`, snapshot test, WS sequence |
| Dashboard shell (moved from P1-D3, F18) | done, **browser-untested** | `dashboard/index.html` |
| `make demo-stub` | done | `scripts/demo_stub.py` + integration test |
| Human-task packs | done | `docs/human_tasks/` (5 packs, Colab notebook, 2 templates) and tools: `check_phone_log.py`, `annotation_agreement.py`, `inspect_uah.py` (all tested on synthetic inputs only) |
| P0-C1 decisions D1-D8 | partly | defaults taken, recorded below; ADR-0006/0007 written as **Proposed** |
| P0-C2 freeze contract v1.0 | **not done** | blocked on dataset verification; `contract_version` stays `"0.1"` |
| P0-A1 DMD / P0-B1 UAH verification | **blocked (human)** | needs downloads (H1); the inspector is ready |
| P0-D1 GitHub repo settings | **human** | branch protection, labels, project board (GitHub settings, not doable from here) |
| P0-D5 phone app test | **blocked (human)** | needs a phone (H4); checker is ready |
| P0-D6 guide meeting | **human** | one-pager ready (`03_guide_meeting.md`) |
| P0-ALL R8/R9 notes | partly | R8 read in full (notes in `11_implementation_analysis.md`); R9 PDF missing |

### Branches
Work was done on `docs/d-relayout`, `feat/d-scaffold`, `feat/c-utils`, `feat/c-contract`, `feat/c-window-dataset`, `feat/c-toy-fusion`, `feat/d-mock-api`, `feat/c-demo-stub`, `docs/c-p0-docs`, merged with `--no-ff` into `dev`. `dev` and tag `v0.0` are pushed to GitHub (the repo is public); `main` is unchanged on both sides. Commits carry a Claude co-author trailer (the earlier ones name Opus 5.5, the later ones Sonnet 5.5, because the model was switched mid-session).

---

## Assumptions (please confirm or overrule; each is cheap to change now, costly after P1)

| ID | Assumption | Why / source |
|---|---|---|
| A1 | Docs moved to `docs/`; PDFs untracked and git-ignored; **git history not rewritten** | fix F1, F2. Old commits still contain the PDFs. If the repo is public or will be, tell me and we decide on history rewrite |
| A2 | Python 3.11 via uv, CPU PyTorch index, exact `uv.lock` | ADR-0006 |
| A3 | No face detected: `face_visible=0`, `face_conf=0`, other 14 vision features NaN | `03` does not say; F9. Implemented in stubs and the validator |
| A4 | `perclos_*` are NaN until 10 s of history exist | `03` does not say; F9. Not enforced by the validator (feature extractor does not exist yet) |
| A5 | Window label = majority over labelled steps; -1 if under 50% labelled or tied | `03` section 7 is silent; F10. `datasets.window_label` |
| A6 | Driver key = `<dataset>:<driver_id>` for splits and personalization | ADR-0007 part 1 |
| A7 | `y_risk[h]` = 1 if a non-uncertain event starts in (window end, end + H]; -1 if the trip has no annotation, the horizon passes the trip end, or an uncertain event starts in the horizon | `03` is silent; F10. Meta key `events_annotated` (default true) and event key `uncertain` are my additions |
| A8 | Mock `windows_required = 600`; real value set in P4 | `05` example only |
| A9 | ~~Alert types stay as in `05`~~ **Superseded 2026-11-13:** event/state taxonomy of ADR-0007 part 2 (approved) | `05` section 3.4, `schemas.py` |
| A10 | WebSocket auth by `?token=` (REST keeps `X-API-Key`) | **Approved 2026-11-13**; in `05` section 1 and 3.6 |
| A11 | Added WS message `recovered` | **Approved 2026-11-13**; in `05` section 3.6 |
| A12 | OpenAPI snapshot at `docs/openapi_snapshot.json` (as `05` says), mock app in `src/driverisk/serving/mock_app.py` with `scripts/mock_server.py` as a wrapper | F19; `03` section 9 and `05` section 5 disagree on the location |
| A13 | Stub pairing: UAH-like = motion_only, DMD-like = vision_only, own-like = paired; stub class mixes follow the real datasets (no distraction in UAH, no aggressive in DMD) | stubs only |
| A14 | Unparseable timestamps/rows in phone logs are reported, never repaired | `phone_log.py` |
| A15 | Inter-annotator agreement uses 10 Hz raster kappa + one-to-one IoU matching at 0.5 | `06` names kappa or IoU without details |

A10 and A11 were approved and `05` was edited to match; `contract_version` stays `"0.1"` because the contract is still a draft until the P0-C2 freeze.

---

## Status of the proposed fixes (F1-F27 in `11_implementation_analysis.md`)

| Fix | State |
|---|---|
| F1 re-layout, F2 PDFs untracked | done |
| F3 ADR-0006, F4 reports README + consent template | done (ADR Proposed; consent template unapproved) |
| F5 schema.py as code source of truth | done (+ contract test against `03`) |
| F6 `mask_v`/`mask_m` naming | done in code and in `02`, `03` |
| F7 separate `e_clip` tensor | documented in `02`, `03`, ADR-0002; no code yet (the video branch comes in P3) |
| F8 all-NaN declaration | documented in `03` section 10 (the validator already enforces it via meta `all_nan_columns`) |
| F9, F10, F11 | implemented as assumptions A3-A7 and now documented in `03`; F9's `ear_threshold*` meta keys are planned |
| F12 taxonomy | done (code, snapshot, `05`, ADR-0007 Accepted) |
| F19 | done: `05` section 3.8 lists every request/response; open: severity mapping and the meaning of `confidence` (P4) |
| F20 `quality`, posed/parked segments | documented in `03` as **planned**; not implemented |
| F21 literature corrections | done in `01`, ADR-0003 (R12 citation incomplete, see above) |
| F22 Windows shim + Windows CI | done (CI green) |
| F23 CONTRIBUTING solo mode | done (`--no-ff` instead of squash) |
| F25 UAH unit/axis rules | documented in `03` section 4.4 from the paper; implementation needs P1-B1 and a real file |
| F26 risk R21, R22 | done |
| F13-F18, F24, F27 | **not started**; they belong to doc edits and phases named in the analysis |

**Pending doc edits:** none of the approved ones. Not yet approved or done: F13-F18, F24 (`02`, `04`, `07`), F27 (README licence line), and the own-file naming rule in `06`.

---

## Surprises and findings

- **Live-server check found a gap the tests could not:** uvicorn had no WebSocket library, so the dashboard's `/stream` would have failed in a browser while all the API tests passed (the in-process test client does not use uvicorn). Added `websockets` (reason in the commit and ADR-0006).
- **Library versions are very recent:** pandas 3.1, NumPy 2.4, PyTorch 2.15, Pydantic 2.14, FastAPI 0.162, Starlette warns that `httpx` in its test client is deprecated and suggests `httpx2` (exists on PyPI, 2.16.0). Not changed: it is a warning only. Decision for you: switch the dev dependency to `httpx2` or ignore.
- `FastAPI RequestValidationError.errors()` has no `include_input` argument; the error envelope strips `input`/`ctx` itself so request data is never echoed.
- Test time grew from about 40 s (77 tests) to about 107 s (99 tests) between runs, mostly from regenerating stub data per session and machine load; not investigated.
- Line endings: `.gitattributes` forces LF (and CRLF for `.ps1`); git still warns on Windows checkouts about CRLF conversion.
- This machine has only Python 3.12 and no `make` (only `mingw32-make`); uv's installation was done by downloading its release binary after `pip install uv` hung. uv lives in `C:\Users\Maruf\.local\uv-bin` (not on PATH).

---

## What I need from you (short)

1. ~~Approve or change A10, A11 and ADR-0007 part 2~~ **Done 2026-11-13.**
2. **Open the dashboard** once and tell me whether it looks right (`make serve`, then http://127.0.0.1:8000/dashboard/, "Start demo trip"). The alert type shown is now `micro_sleep` (predictive).
3. ~~History rewrite~~ **Decided: leave it for now** (risk R21).
4. **Start the long-lead human tasks** (`docs/human_tasks/README.md`): DMD registration and UAH download (H1), missing papers (H10), guide meeting (H5), phone and passenger for the app test (H4, H6).
5. ~~Say go for P1~~ **Given 2026-11-13 ("go ahead for the next hour, do what is best").**
6. **Agree the event thresholds** in `configs/events.yaml` with the guide (they are my proposals, see below).

## Next (P1, remaining)
Not started: MediaPipe vision features and causal PERCLOS (P1-A1/A2; needs `mediapipe` added to the lock file, ADR-0006, and your webcam), then the loaders UAH (P1-B1), DMD (P1-A3), splits script (P1-C2) and label maps (P1-C1), which wait for the downloads (H1). The own-data pipeline (P1-D1) waits for the phone-app test (H4).

---

## P1 work log (2026-11-13, autonomous hour after "go")

Branches `fix/d-make-ps1-uv-path`, `feat/c-metrics`, `feat/d-experiment-log`, `feat/b-motion-features`, `feat/b-events`, `feat/c-augment`, all merged into `dev` with `--no-ff`. Tests: **186 passed** (102 before this hour), lint clean. Every module was written test-first; the expected numbers in the metric tests are worked out by hand in comments.

| Task | State | What it is |
|---|---|---|
| `make.ps1` finds uv | done | `make serve` now works in a fresh PowerShell; tested with uv off PATH |
| P2-C1 metrics library (moved up) | done, synthetic cases only | `evaluation/metrics.py`: macro-F1, per-class P/R/F1, confusion matrix, balanced accuracy, ECE (classes and binary), Brier, PR-AUC per horizon, alerting metrics (FA/h, recall, lead time, early-warning rate), pooling over trips, `alerts_from_scores`, `select_threshold` |
| Robustness degradations (04 §3.4, F24) | done | `evaluation/robustness.py`: modality masking (random or contiguous), noise at a given SNR, time-offset shift, `robustness_curve`. "Phone moved" is not here: it needs the raw IMU, so it goes with the motion code |
| P1-D4 experiment logging | done | `utils/experiment.py` + `experiments/README.md`: run folder, config/meta/metrics, `results.csv` |
| P1-B2 motion features | done on **synthetic data only** | `features/motion.py`: all 14 features from raw phone IMU + GPS, causal. On a synthetic drive with a rotated phone it recovers `acc_long`/`acc_lat` with RMSE 0.024 m/s^2 and correlation above 0.999 |
| P1-B3 event detector | done on synthetic frames only | `features/events.py` + `configs/events.yaml`: harsh_accel, harsh_brake, sharp_turn, lane_weave |
| P1-C3 normalization + augmentation | normalization was done in P0; augmentation done | `data/augment.py`: motion jitter, scale and axis rotation (in physical units), vision jitter, random face-not-found steps; `WindowDataset(..., transform=Augmenter(...))`, off by default. Magnitudes in `AugmentConfig` are proposals; tune on validation only |

### New assumptions (please confirm or overrule)

| ID | Assumption | Where |
|---|---|---|
| A16 | Macro-F1 averages over the classes present in the truth (UAH has no `distracted`, DMD no `aggressive`); pass `classes=` to force a set. A class that is predicted but absent from the truth counts as errors for the true class, not as a zero-F1 term | `metrics.macro_f1` (F15) |
| A17 | Alerting rules (fix F15): an event's window is `[onset - 5 s, offset + 2 s]`; an alert inside any event window, certain or uncertain, is never a false alarm; an event is detected by an alert in `[onset - 5 s, offset]`, lead time = onset minus the earliest such alert (negative = late); FA/h = false alarms / hours outside all event windows, excluded (posed) time and, if given, outside the evaluated range (UAH: speed above 50 km/h); trips pool by summing counts. `doc 04` still says something looser; **not edited** | `metrics.alerting_metrics` |
| A18 | Forward axis from regression of horizontal acceleration on the GPS speed change over the whole trip, up axis from stationary moments (fallback: trip mean). Replaces the contract's "first stationary-to-driving transition". Alignment fails cleanly (axis features NaN) when the speed profile has no acceleration | `motion.estimate_alignment` |
| A19 | `speed` and `heading_rate` are **held** from the latest GPS fix (NaN and `gps_valid = 0` after 2 s), not interpolated, so the features are causal and identical on the phone. The contract word "interpolated" is not followed | `motion.py` |
| A20 | Sign convention: left turn is positive for `acc_lat`, `yaw_rate` and `heading_rate` (`heading_rate = -d(compass)/dt`). The UAH yaw sign still has to be checked on the real files (F25) | `motion.py` |
| A21 | `lateral_drift_5s` without lane data is a proxy (std over 5 s of the detrended double integral of `acc_lat`); an indicator of weaving, not a lane position | `motion.py` |
| A22 | Run folders live in `experiments/runs/<run_id>/` (already git-ignored), not `experiments/<run_id>/` as `04` and CONTRIBUTING say; `results.csv` stays in `experiments/` | `utils/experiment.py` |
| A23 | Event thresholds in `configs/events.yaml` (harsh accel +2.5, harsh brake -3.0, sharp turn 2.5 m/s^2, lane weave 0.5 m with 3 excursions) are **my placeholders**, not from a paper. Per the contract they must be agreed with the guide and then frozen before any model evaluation | `configs/events.yaml` |

### Not verified (honest list)
- **Nothing here has touched real data.** Motion features and events are checked on synthetic signals I generated; they may still break on real phone logs (clock jitter, irregular sampling, spikes, phones that report linear acceleration instead of raw). Run `scripts/check_phone_log.py` and then the extractor on the first real pilot trip.
- The synthetic phone has no gyro bias, no GPS outliers and perfectly regular 50 Hz IMU timing.
- UAH specifics (G units, Z longitudinal, km/h, yaw angle sign) are not applied anywhere yet; that is the UAH loader (P1-B1).
- The metrics library was not cross-checked against sklearn (not a dependency); expected values were computed by hand.
- Doc `04` was not updated for A16, A17 and A22.
