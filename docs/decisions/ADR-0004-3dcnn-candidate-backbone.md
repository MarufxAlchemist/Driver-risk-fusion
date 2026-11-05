# ADR-0004: 3D-CNN as a candidate backbone, validated by ablation

Status: Proposed
Date: 2026-10-18
Deciders: team

## Context
The final project title names 3D-CNN. Literature supports that 3D CNNs beat 2D CNNs for driver behavior recognition (R6) and that pruned 3D CNNs can run on edge devices (R7). Practical constraints: GPU time, small own dataset, and the latency budget on a phone or laptop CPU.

## Options considered
1. **Commit to 3D-CNN everywhere.** Matches the title; risky on compute, data size and edge latency.
2. **Feature-only vision path (MediaPipe + LSTM).** Cheap and robust; does not match the title.
3. **Both paths, compared in ablation, with a go/no-go gate.** Honest evidence for the choice; costs more effort.

## Decision
Build the feature path first (B3). Build the 2D+LSTM (B5) and 3D-CNN (B4) clip paths in P2 and compare them (ablation A7). Make a **go/no-go decision at the end of P2 (week 8)**:
- **Go:** 3D-CNN is at least as accurate as the alternatives on DMD validation, and fits the latency budget (or can be pruned to fit). It is included in fusion (as embedding input) and in the demo.
- **No-go:** keep the feature path in the deployed system and report 3D-CNN as an offline comparison. The title still holds because the 3D-CNN is a studied backbone, but the report must say exactly what was deployed.

## Consequences
- The title's claim is protected either way, as long as the comparison is run and reported.
- Deployed system may be feature-based; discuss with the guide at the week 8 meeting.
- Effort in P2-A2 and P2-A3 is real; if compute is a blocker use subsets, smaller clips (8 frames, 96x96) and pretrained initialization.
- Wrong if: 3D-CNN is much better and fits the budget, in which case move it into the main path in P3.
