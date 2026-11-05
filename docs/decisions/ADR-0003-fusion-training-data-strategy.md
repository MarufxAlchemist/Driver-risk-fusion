# ADR-0003: Fusion training data strategy (paired data problem)

Status: Proposed (confirm after the dataset check in P0-A1 and P0-B1)
Date: 2026-10-18
Deciders: team

## Context
Our public datasets complement each other but are not paired:
- UAH-DriveSet: smartphone motion and GPS, driving styles (normal, drowsy, aggressive), road-facing video. No driver-face video and no distraction class (to be verified).
- DMD: driver-facing video with distraction and drowsiness related labels. No vehicle motion (to be verified).

The project's core claim is *cross-attention fusion of driver camera and phone sensors*. Training and testing fusion needs samples where both modalities describe the **same moment**. Fabricating pairs can produce results that look good for the wrong reasons.

## Options considered
1. **Own paired recordings only.** Honest and matches the deployment setting. Small data, high effort, safety constraints on what behaviours can be recorded.
2. **Public datasets with class-conditional synthetic pairing** (pair a UAH motion window and a DMD vision window of the same class). Large data. The pair is statistically meaningless: the two streams share only the label. Accuracy gains would come from label leakage, not real fusion.
3. **Pretrain each branch on public data, fine-tune the fusion on own paired data** (F2). Uses public data where it is valid, own data where pairing matters.
4. **Change the claim** to late decision-level fusion. Weakens the novelty claim.

## Decision
- **Primary:** own paired dataset (setting F1) is a Must-have. Recording starts in P1.
- **Pretraining:** vision branch on DMD, motion branch and Stage 1 on UAH, then fine-tune fusion on own paired data (F2).
- **Synthetic pairing (F3):** allowed only to study the *robustness behaviour* (graceful degradation under masking), never to report headline accuracy, and always labelled "synthetic pairing" in tables and text.
- Report all three settings separately.

## Consequences
- Data collection moves from "optional extra" to critical path, with safety, consent and guide approval requirements (`06_data_collection_protocol.md`).
- Headline results come from a small dataset, so the report must give per-driver variance, confidence intervals and threats to validity.
- Fallback if own data stays small: present fusion as a proof of concept on pilot data plus robustness and personalization analysis (risk R1, R2).
- Wrong if: a public paired dataset turns up (check once more in P0 and in literature search). Then switch to it for F1.
