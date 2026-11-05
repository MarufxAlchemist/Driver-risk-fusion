# Contributing

Rules for four people working in parallel. Short on purpose; if something is unclear, ask in the team chat and then fix this file.

## 1. Setup

```bash
git clone <repo-url> && cd driver-risk-fusion
make setup        # creates the environment, installs dependencies, installs pre-commit hooks
make test
```

- Python 3.11 (pinned in `pyproject.toml`). Use the same dependency lock for everyone.
- Never `pip install` something without adding it to `pyproject.toml` in the same PR.
- GPU training happens on Colab, Kaggle or the college lab; keep notebooks and scripts runnable from the repo.

## 2. Branches

| Branch | Purpose |
|---|---|
| `main` | Always demo-able. Updated only at phase ends via PR from `dev`. Tagged `v0.<phase>` |
| `dev` | Integration branch. All feature PRs target `dev` |
| `feat/<ws>-<name>` | New work, for example `feat/a-perclos`, `feat/c-cross-attention` |
| `fix/<name>` | Bug fixes |
| `docs/<name>` | Documentation only |
| `exp/<ws>-<name>` | Experimental branches that may never merge |

`<ws>` is the workstream letter a, b, c or d.

## 3. Commits

Conventional style, imperative mood:

```
feat(vision): add causal PERCLOS over 30 s and 60 s windows
fix(sync): correct offset sign when end tap precedes start tap
docs(contract): bump to v0.2, add lane_offset nan rule
test(fusion): masked stream does not change output
chore(ci): cache pip
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`, `exp`.
Keep commits small. Reference issues with `#<number>`.

## 4. Pull requests

1. Open the PR early as a draft if you want feedback.
2. Fill in the PR template.
3. CI must pass (lint, tests, contract validator).
4. At least one approval from a **different workstream**.
5. Reviewer replies within 48 hours; author resolves comments, then merges with **squash**.
6. Delete the branch after merging.

PR size: aim for under 400 changed lines. Large moves (renames, reformatting) go in their own PR.

## 5. Code standards

- Format and lint with `ruff` (`make lint`, `make format`). Pre-commit runs it on commit.
- Type hints on public functions. Docstrings that state **tensor shapes** and units, for example `x_m: [B, T, 14] float32`.
- Config in YAML under `configs/`. No magic numbers or paths in code.
- Randomness: always take a `seed` and call `utils.seed.set_seed(seed)`.
- Paths via `pathlib` and the repo's `utils.paths` helper, never absolute paths or user names.
- No data, checkpoints, videos or secrets in git. `.gitignore` covers `data/`, `checkpoints/`, `*.pt`, `*.onnx`, `.env`. Report an accidental commit immediately.
- Notebooks are for exploration only. Move reusable logic into `src/driverisk/`. Clear outputs before committing.
- Logging via the `logging` module, not `print`, inside the library.

## 6. Tests

| Kind | Where | Rule |
|---|---|---|
| Unit | `tests/unit/` | One test file per module; use stub fixtures from `tests/fixtures/` |
| Contract | `tests/contract/` | Validates parquet schema, shapes, label values, split disjointness |
| Integration | `tests/integration/` | Stub pipeline end to end (`make demo-stub`) |
| Slow / GPU | marked `@pytest.mark.slow` | Not run in CI by default |

New behavior needs a test. A bug fix needs a test that failed before the fix.

## 7. Experiments

Follow `docs/04_experiment_protocol.md`. In short:

- One config file per run, committed.
- Run id includes the commit hash; outputs go to `experiments/<run_id>/`.
- Three seeds minimum for any number that goes in a table.
- Tune on validation only.
- Add the result row to `experiments/results.csv` in the same PR as the code change.

## 8. Contracts and docs

- Changing `docs/03_data_contract.md` or `docs/05_api_contract.md` requires approval from **all four** members and a version bump.
- Update docs in the same PR as the behavior change.
- Decisions that are expensive to reverse get an ADR in `docs/decisions/`.

## 9. Data and privacy

- Recordings of people stay on the private team drive, never in git or public chats.
- Use anonymous driver IDs everywhere in code, filenames and logs.
- Follow `docs/06_data_collection_protocol.md` for every recording, including safety rules.

## 10. Issues and labels

| Label | Meaning |
|---|---|
| `ws:a`, `ws:b`, `ws:c`, `ws:d` | Workstream |
| `phase:p0` ... `phase:p7` | Phase |
| `interface` | Needs coordination between workstreams |
| `blocked` | Waiting on something; say what |
| `bug`, `enhancement`, `docs`, `question` | Type |
| `risk` | Linked to an entry in the risk register |

Issue titles start with the task ID from `docs/07_phase_task_breakdown.md`, for example `P1-A1: MediaPipe vision features`.

## 11. Authorship

Everyone commits under their own name and email. Do not commit as someone else. Co-author trailers are fine for pair work.
