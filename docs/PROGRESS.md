# Progress log

Running log for the solo build (master prompt section 3.1). Newest entries on top within each section. Numbers are from commands run in this repo; nothing is estimated unless it says so.

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
| Stubs vs validator | `tests/contract/test_validate.py` + both stub scripts | all stub trips pass; every rule in contract section 10 has a test that it catches its violation (columns, dtype, time step, range, `inf`, 0/1 flags, undeclared all-NaN, labels, version, mask rule, meta, split leakage, missing driver, events, orphan files) |

**Not verified, and why:**
- **The dashboard page was never opened in a browser** (no browser available to me). It is served and its WebSocket target works (checked with a socket client), but rendering and the gauge/alert display are untested. Please open `make serve` -> http://127.0.0.1:8000/dashboard/ and press "Start demo trip".
- **CI has never run** (nothing is pushed). The workflow exists but is untested on GitHub; Mac was not tested at all.
- **The Colab notebook has not been run** on Colab/Kaggle.
- `make setup` was not run from a clean clone (I used `uv sync` in place). Fresh-checkout install time is unmeasured.

### P0 tasks

| Task | State | Notes |
|---|---|---|
| P0-D2 scaffold | done | `feat/d-scaffold`; uv + Python 3.11.18, `uv.lock`, Makefile + `make.ps1`, pre-commit config, `.gitattributes`, `data/README.md` |
| P0-D3 CI | written, **not run** | `.github/workflows/ci.yml`, ubuntu + windows |
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
| P0-D1 GitHub repo settings | **human** | branch protection, labels, project board; nothing pushed |
| P0-D5 phone app test | **blocked (human)** | needs a phone (H4); checker is ready |
| P0-D6 guide meeting | **human** | one-pager ready (`03_guide_meeting.md`) |
| P0-ALL R8/R9 notes | partly | R8 read in full (notes in `11_implementation_analysis.md`); R9 PDF missing |

### Branches
Work was done on `docs/d-relayout`, `feat/d-scaffold`, `feat/c-utils`, `feat/c-contract`, `feat/c-window-dataset`, `feat/c-toy-fusion`, `feat/d-mock-api`, `feat/c-demo-stub`, `docs/c-p0-docs`, merged with `--no-ff` into `dev`. `main` is unchanged and nothing is pushed. Commits carry a Claude co-author trailer (the earlier ones name Opus 5.5, the later ones Sonnet 5.5, because the model was switched mid-session).

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
| A9 | Alert types stay exactly as in `05` (`aggressive_accel`, `drowsy` ...); the demo's class-to-alert table is demo-only | ADR-0007 part 2 is a proposal that changes `05`; needs your approval |
| A10 | WebSocket auth by `?token=` (REST keeps `X-API-Key`) | browsers cannot set WS headers; **changes `05` section 1/3.6** |
| A11 | Added WS message `recovered` | the sequence in `05` section 5 ends with "recovered" but no such message type exists; **changes `05` 3.6** |
| A12 | OpenAPI snapshot at `docs/openapi_snapshot.json` (as `05` says), mock app in `src/driverisk/serving/mock_app.py` with `scripts/mock_server.py` as a wrapper | F19; `03` section 9 and `05` section 5 disagree on the location |
| A13 | Stub pairing: UAH-like = motion_only, DMD-like = vision_only, own-like = paired; stub class mixes follow the real datasets (no distraction in UAH, no aggressive in DMD) | stubs only |
| A14 | Unparseable timestamps/rows in phone logs are reported, never repaired | `phone_log.py` |
| A15 | Inter-annotator agreement uses 10 Hz raster kappa + one-to-one IoU matching at 0.5 | `06` names kappa or IoU without details |

A10 and A11 touch the API contract, so by the master prompt they need your explicit approval and a version bump; they are implemented because the scripted sequence and a working browser dashboard cannot exist without them. The text of `05` is **not yet edited**.

---

## Status of the proposed fixes (F1-F27 in `11_implementation_analysis.md`)

| Fix | State |
|---|---|
| F1 re-layout, F2 PDFs untracked | done |
| F3 ADR-0006, F4 reports README + consent template | done (ADR Proposed; consent template unapproved) |
| F5 schema.py as code source of truth | done (+ contract test against `03`) |
| F6 `mask_v`/`mask_m` naming | code follows it; docs `02`/`03` not yet edited |
| F9, F10, F11 | implemented as assumptions A3-A7; docs not yet edited |
| F19 | partly (A10-A12); remaining request/response details in `05` not yet written into the doc |
| F22 Windows shim + Windows CI | done (CI unrun) |
| F25 UAH unit/axis rules | not implemented (needs P1-B1 and a real file); `inspect_uah.py` will show units |
| F7, F8, F12, F13-F18, F20, F21, F23, F24, F26, F27 | **not started**; they belong to the doc edits and phases named in the analysis. None blocks P1 except F12 (taxonomy) and F25 |

**Pending doc edits that need your approval:** `03` (F6-F10, F20, F25, version stays 0.1 until the freeze), `05` (A10, A11, F12, F19), `01` (F21 literature corrections), CONTRIBUTING solo mode (F23), risk register R21/R22 (F26). I did not edit these documents.

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

1. **Approve or change** A10, A11 (they alter `05`) and ADR-0007 part 2 (alert names).
2. **Open the dashboard** once and tell me whether it looks right.
3. **Push `dev`** to GitHub when you are happy, so CI runs; tell me if the repo is public (then decide on rewriting history for the PDFs).
4. **Start the long-lead human tasks** (`docs/human_tasks/README.md`): DMD registration and UAH download (H1), missing papers (H10), guide meeting (H5), phone and passenger for the app test (H4, H6).
5. Say **go for P1**. Until the datasets exist, P1 can start with motion features (P1-B2), the metrics library (P2-C1, moved up), event detector, and vision features on your webcam.

## Next (P1, after your go)
Order from the analysis: normalization/augmentation, experiment logging, metrics library, motion features and events, MediaPipe vision features, then loaders once downloads exist.
