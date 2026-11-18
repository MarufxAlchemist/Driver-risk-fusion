# Reports: map from the repo to the report chapters

Written in P0 as a skeleton. Each row says where the material for a chapter lives. Nothing here contains results; results appear only after runs exist under `experiments/` with a config, seed and commit hash.

| Chapter | Sources in the repo | Status |
|---|---|---|
| Introduction and problem statement | `README.md`, `00_PROJECT_PLAN.md` section 1 | draft later (P7) |
| Literature survey | `01_literature_notes.md`, `docs/Research-papers/README.md` | needs R6, R7, R9, R10 read first |
| Methodology | `02_architecture.md`, `03_data_contract.md`, ADRs | stable after contract freeze |
| Data and preprocessing | `data/README.md` (dataset card), `03` sections 3 and 8, `06_data_collection_protocol.md`, `experiments/data_report.md` | P1, P6 |
| Experimental setup | `04_experiment_protocol.md`, `configs/` | P2 |
| Results and ablations | `experiments/ablations/A1.md` ... `A9.md`, `experiments/results.csv` | P3-P6 |
| Explainability and user study | `src/driverisk/explain/`, P6-D2 survey | P4, P6 |
| System and deployment | `05_api_contract.md`, `src/driverisk/serving/`, A9 benchmark | P5 |
| Threats to validity | `04` section 9, `09_risk_register.md`, posed vs real labelling | P6 |
| Ethics, consent, safety | `06` sections 1-2, `docs/reports/consent_form_template.md` | P0 |
| Conclusion and future scope | `00` section 4 (Won't list) | P7 |

Report rules (from the master prompt): no invented numbers or citations; negative results are reported as they are; synthetic pairing is always labelled "synthetic pairing"; posed behaviours are always labelled posed; the use of AI assistance in building the code is disclosed as agreed with the guide.
