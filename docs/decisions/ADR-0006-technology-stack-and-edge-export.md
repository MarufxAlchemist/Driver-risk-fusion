# ADR-0006: Technology stack and edge export path

Status: Proposed
Date: 2026-11-13
Deciders: Maruf (Claude drafted)

## Context
`02_architecture.md` section 8 lists technology choices without versions, and the plan (D5) requires pinned versions. The machine used for development is Windows with Python 3.12 and no `make`; CI and the guide's laptop may be Mac. GPU training happens on Colab/Kaggle. The title names a 3D-CNN, but ADR-0004 keeps the feature path as the default.

## Options considered
1. **pip + venv + requirements.txt.** Familiar; no lock file, and the Python version is the user's problem (3.12 here, 3.11 required).
2. **conda.** Handles Python versions; heavy, slow in CI, not what Colab uses.
3. **uv (Python manager + lock file).** One tool installs Python 3.11 and an exact `uv.lock`; works on Windows, Mac and Linux; adds a tool the team has not used.

## Decision
- **Python 3.11** (`requires-python = ">=3.11,<3.12"`), managed with **uv**; the exact environment is `uv.lock`, committed.
- **PyTorch from the official CPU index** for local and CI runs. GPU runs use the Colab notebook in `docs/human_tasks/`, which installs a CUDA build.
- **Data:** pandas + pyarrow (parquet), PyYAML for configs.
- **API:** FastAPI + Pydantic v2, uvicorn with `websockets` (needed to serve WebSockets; found by a live-socket check, the in-process test client does not need it).
- **Quality:** ruff (lint + format), pytest, pre-commit; CI on ubuntu and windows.
- **Windows:** `make.ps1` mirrors the Makefile targets.
- **Vision features (P1):** MediaPipe Face Landmarker; the version is pinned when P1-A1 starts and recorded here.
- **Edge export (P5):** ONNX + ONNX Runtime for the fusion and heads; the vision path stays feature-based unless ADR-0004 says go.
- **Tracking:** CSV plus one folder per run under `experiments/`; MLflow or W&B only if you ask for it (not needed to meet the protocol).

## Consequences
- `make setup` works from a clean checkout on Windows, Mac and Linux, given uv.
- Everyone must install uv (one command). The lock file changes with every dependency change; reviewers see it.
- Library versions are very recent; if one breaks (for example the Starlette test-client warning about `httpx`), pin it here with a reason.
- Wrong if: uv cannot be installed on the college machines. Then fall back to pip with a `requirements.txt` exported from the lock.
