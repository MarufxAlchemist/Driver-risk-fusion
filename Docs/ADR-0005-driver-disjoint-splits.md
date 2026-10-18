# ADR-0005: Driver-disjoint splits and within-driver personalization evaluation

Status: Proposed
Date: 2026-10-18
Deciders: team

## Context
Windows from the same trip are highly correlated, and windows from the same driver share style and appearance. Random window-level splits leak this information and produce inflated scores. Personalization, by definition, depends on having earlier data of the same driver, so it needs a different evaluation.

## Options considered
1. Random window splits. Simple; badly leaky.
2. Trip-level splits. Better; still leaks driver identity.
3. **Driver-disjoint splits for generalization, chronological within-driver splits for personalization.** Honest for both questions; needs enough drivers and trips.

## Decision
- **Generalization experiments (A1-A3, A5-A8):** split by driver. UAH uses leave-one-driver-out or grouped k-fold because of few drivers; DMD uses a subject-disjoint split; own data is driver-disjoint where possible.
- **Personalization (A4):** for each driver, the first K normal trips calibrate the baseline; later trips are the test set. The same model with global versus personalized thresholds is compared at equal recall.
- Split assignment is done once by script and stored (`splits.json`); a CI test checks that no driver appears in more than one split.

## Consequences
- Scores will be lower and more honest than window-level splits.
- Small numbers of drivers mean high variance; the report shows per-driver results and uncertainty.
- Drivers with fewer than 4 trips are reported as "not enough data" for personalization.
- Wrong if: so few drivers exist that no stable estimate is possible. Then use bootstrap over trips and state the limitation clearly.
