# 09. Risk Register

Review at every weekly sync (top 5 risks) and fully every two weeks. Likelihood and impact: L (low), M (medium), H (high). Score = rough product, used only for ordering.

Status: DRAFT v0.1, last reviewed 2026-10-18.

---

## 1. Register

| ID | Risk | L | I | Early-warning trigger | Mitigation | Contingency | Owner |
|---|---|---|---|---|---|---|---|
| R1 | **No public dataset pairs driver-face video with phone IMU for the same trip**, so fusion cannot be trained or validated honestly on public data alone | H | H | Dataset check (P0-A1, P0-B1) confirms the assumption | Own paired data is a **Must** (ADR-0003); start pilot trips in P1; use F2 pretrain-then-finetune; keep F3 synthetic pairing only for robustness and clearly labelled | If own data is too small, present fusion as a proof of concept with pilot data and report the limitation; focus claims on robustness and personalization | C, D |
| R2 | Own dataset too small for reliable conclusions (few drivers and trips) | H | H | Fewer than 8 paired trips by week 8 | Recruit more volunteers early; schedule fixed recording slots; record normal driving widely; report per-driver variance | Reduce claims, use bootstrap confidence intervals, say "pilot study" | D |
| R3 | UAH-DriveSet has few drivers, so subject-disjoint results are noisy | H | M | Per-fold variance larger than model differences | Leave-one-driver-out, report per-driver, three or more seeds | State clearly that differences below one std are not claimed | B |
| R4 | Simulated or posed drowsiness and distraction do not reflect real behaviour | H | M | Large gap between DMD and posed own data | Use DMD real annotations as the main source for these classes; label posed data as posed | Present drowsy/distracted as detection of behavioural indicators (eye closure, gaze), not fatigue state | A |
| R5 | Timestamp synchronization between camera and phone sensors is inaccurate | M | H | Residual offset above 100 ms in pilot trips | Start and end tap events (protocol section 5), offset and drift estimation, same-device recording where possible | Use coarser windows (e.g. 1 s steps) for the paired data | D |
| R6 | Compute is not enough for 3D-CNN training | M | M | Epoch time too long on Colab/Kaggle by week 8 | Subset experiments, smaller clips (8 frames, 96x96), mixed precision, pretrained init; go/no-go decision at P2 | 3D-CNN becomes an offline Could; feature-based path is the main path (ADR-0004) | A |
| R7 | Scope creep: too many models and features | H | H | More than 2 weeks of slip on any phase gate | MoSCoW priorities; scope guard at week 10; cut Could first | Drop 3D-CNN, then A8, then optional edge export | C |
| R8 | Edge latency misses the budget | M | M | A9 benchmark above 300 ms | Feature path instead of clips, quantization, pruning, lower frame rate | Report offline accuracy and measured latency honestly; demo on laptop | D |
| R9 | Unsafe or ethically problematic data collection | L | H | Anyone proposes testing real distraction or drowsiness while driving | Safety rules in `06_data_collection_protocol.md`, guide approval before recording, consent forms | Stop collection until reviewed | D |
| R10 | Privacy breach of recordings | L | H | Raw video found in git or shared chats | `.gitignore` for data, shared drive with restricted access, anonymous IDs, deletion policy | Remove data, rewrite history if needed, notify participants | D |
| R11 | Label noise from trip-level labels and thresholds | M | M | Events labeled aggressive in calm trips | Event-level labels for alerts, review a 20% sample, report agreement | Treat noisy classes with robust loss, report event-level metrics only | B |
| R12 | Fusion does not beat the best single modality | M | H | A1 fails in P3 | Check data quality first; tune modality dropout and gates; try late fusion with gating; analyze per-class | Report honestly; emphasize robustness (A3) and personalization (A4); discuss with the guide | C |
| R13 | Personalization shows no gain or too little calibration data | M | M | A4 per-driver gains inconsistent | Shrinkage to the global baseline; more trips per driver; test window sizes for calibration | Report negative or mixed result with analysis; keep module as a design contribution | C |
| R14 | Early prediction yields lead times below 1 s or too many false alarms | M | M | A5 median lead time below 1 s at the FA/h budget | Longer context, richer features, tune horizons, move threshold using validation | Report the trade-off curve; present detection plus prediction as incremental | C |
| R15 | Team availability (exams, illness, other commitments) | H | M | Stand-up responses drop; board stale | Backups per workstream, small PRs, 8-10 h/week assumption, exam calendar on the board | Rebalance tasks at the weekly sync; defer Could items | D |
| R16 | Integration pain: modules work alone but not together | M | H | Stub pipeline fails after a PR | Contract validator in CI, `make demo-stub` in CI, integration test every week | Freeze interfaces; schedule a dedicated integration week in P5 | D |
| R17 | Dataset license or access problems (DMD registration, terms) | L | M | Download blocked or terms restrict use | Register early (P0), read licenses, cite properly | Use alternative public sets or reduce to own data and UAH | A |
| R18 | Reference quality and numbering mistakes in presentations and reports | M | L | Citations that do not match the list | Canonical R1-R11 numbering in `01_literature_notes.md`; fix deck numbering; check references before each submission | Correct and re-submit | Rotating |
| R19 | Reproducibility failures (cannot regenerate a number) | M | M | A run without config or commit hash | Experiment protocol rules, `make reproduce`, one config per run | Re-run and replace the number | C |
| R20 | Guide expectations differ from the plan | L | M | Feedback at fortnightly meetings | Share this plan in week 1; adjust priorities; record approvals | Re-scope with a new ADR | Rotating |

---

## 2. Top five at start (expected to dominate)

1. R1 paired data problem
2. R2 small own dataset
3. R7 scope creep
4. R5 synchronization
5. R12 fusion not beating single modality

---

## 3. Risk review log

| Date | Reviewed by | Changes |
|---|---|---|
| 2026-10-18 | Maruf | Initial register |
