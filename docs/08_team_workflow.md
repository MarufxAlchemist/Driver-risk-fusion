# 08. Team Workflow

How four people build one system in parallel without blocking each other.

Status: DRAFT v0.1 (confirm at the first team call).

---

## 1. Roles

| Workstream | Owner | Backup | One-line mission |
|---|---|---|---|
| A. Vision pipeline | Anuja Naik | Gautam Iyengar | Turn driver video into reliable vision features and a video backbone |
| B. Motion pipeline | Tanishq Basagare | Maruf Nadaf | Turn phone IMU/GPS into clean motion features and Stage 1 |
| C. Fusion, personalization, forecasting | Maruf Nadaf | Anuja Naik | Own the models that make the project novel, and the evaluation protocol |
| D. Platform and delivery | Gautam Iyengar | Tanishq Basagare | Own the repo, CI, API, dashboard, edge pipeline, data logistics and report assembly |

Rotation: at each phase boundary, swap backups so everyone has seen two workstreams by the end of P4. The owner stays responsible for the deliverable.

Cross-cutting ownership:

| Topic | Owner |
|---|---|
| Data contract changes | C (with a PR that all four approve) |
| Experiment protocol | C |
| API contract | D |
| Data collection protocol and consent | D |
| Literature notes and guide communication | Rotating weekly |
| Meeting notes | Rotating weekly |

---

## 2. Interfaces between workstreams

```
        A (vision features) ──┐
                              ├──► C (fusion, personalization, forecaster) ──► alerts ──► D (inference, API, dashboard)
        B (motion features) ──┘                     ▲
                                                    │
                        D (own data pipeline) ──────┘ (paired trips)
```

| Producer | Consumer | Interface | Stub available by |
|---|---|---|---|
| A | C | `x_v`, `mask_v` per `03_data_contract.md` | End of week 2 |
| B | C | `x_m`, `mask_m` | End of week 2 |
| C | D | Alert JSON (`05_api_contract.md`) and a Python `Predictor` class with `predict(window) -> Alert or None` | End of week 3 (stub), real in P4 |
| D | A, B, C | Processed own-data parquet | Pilot in P1 |
| B | A, C | Event labels for harsh events | P1 |

Rule: **if you need something from another workstream, open an issue labelled `interface` and tag the owner.** Do not edit another person's module without a PR they review.

---

## 3. Git workflow (short)

Full details in `CONTRIBUTING.md`.

- Branches: `main` (always demo-able), `dev` (integration), `feat/<ws>-<short-name>`, `fix/...`, `docs/...`.
- Every change via PR into `dev`, at least one reviewer from a different workstream.
- CI must pass. No force-push to shared branches.
- `dev` merges to `main` at phase ends, tagged `v0.<phase>`.
- Large files (data, checkpoints, video) never go in git.

---

## 4. Rituals

| Ritual | When | Length | Content |
|---|---|---|---|
| Stand-up (async message) | Mon, Wed, Fri | 2 minutes | Done, doing, blocked (plain text in the team chat) |
| Weekly sync | Once a week, same slot | 20 minutes | Board walk, blockers, risk register, decisions needed |
| Phase demo | Phase end | 30 minutes | Live demo + gate check (section 5) |
| Guide meeting | Every two weeks | 20 minutes | Progress, decisions, approvals |
| Paper reading | Weekly | 15 minutes inside the sync | Each person says one thing they learned |
| Retro | Phase end | 15 minutes | What to keep, change, stop |

Meeting notes go into `docs/decisions/` if they contain a decision, otherwise into a shared document.

---

## 5. Definitions

### Definition of Ready (before starting a task)
- Issue exists with ID, owner, effort and "done when" criteria.
- Inputs and outputs specified in the contract or the issue.
- Dependencies done or stubbed.

### Definition of Done (for a PR)
- Code runs from a clean checkout (`make setup`).
- Unit tests for new logic, CI green.
- Docstrings with input/output shapes; config in YAML, not hard-coded.
- Contract validator passes for any new data output.
- Docs updated if behaviour or interface changed.
- Experiment results logged in `experiments/` if the PR changes results.
- Reviewed and approved by one other member.

### Phase gate
Each phase has an exit criterion (`00_PROJECT_PLAN.md` section 3). The gate is checked live in the phase demo. A failed gate means a re-plan, not a silent slip.

---

## 6. Communication

| Channel | Use |
|---|---|
| Team chat | Daily coordination, quick questions |
| GitHub issues and PRs | Task state, review discussion, decisions that affect code |
| GitHub Projects board | Source of truth for task status (Backlog, Ready, In progress, In review, Done) |
| Shared drive | Raw data, recordings, consent forms (private, outside git) |
| Weekly sync | Decisions needing all four |

Response norm: acknowledge a review request within 24 hours; reviews finish within 48 hours.

---

## 7. Onboarding checklist (each member, week 1)

- [ ] Read `00_PROJECT_PLAN.md`, `02_architecture.md`, `03_data_contract.md`
- [ ] Skim `10_glossary.md`
- [ ] Clone repo, `make setup`, run `make test`
- [ ] Confirm GPU access route (college, Colab, Kaggle)
- [ ] Join the board; pick or accept your P0 tasks
- [ ] Add yourself to the consent-process roster (D keeps it)
- [ ] Read your assigned first paper and write the 5-line note

---

## 8. Conflict and slippage handling

| Situation | Action |
|---|---|
| A task is blocked for more than 3 days | Raise at the next sync; reassign or reduce scope |
| Someone is unavailable (exams, illness) | Backup takes over the in-flight tasks; update the board the same day |
| Two people disagree on a technical choice | Write a short ADR with options and a recommendation; decide in the sync; if still split, ask the guide |
| A phase gate is missed | 30-minute re-plan: cut a Could item first, then a Should item; never cut baselines or evaluation |
| Merge conflicts are frequent | Smaller PRs, rebase daily, split files by owner |

---

## 9. Academic calendar protection

- Mark exam and submission dates on the board in week 1; plan no gate or demo in exam weeks.
- Assume each member has about 8-10 project hours per week. Task effort sizes in `07_phase_task_breakdown.md` assume that and should be revisited after P1.
- No critical single point of failure: every module has a backup who has read the code.
