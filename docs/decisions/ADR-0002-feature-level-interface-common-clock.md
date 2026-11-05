# ADR-0002: Feature-level interface with a common 10 Hz clock

Status: Proposed
Date: 2026-10-18
Deciders: team

## Context
Vision runs at 15-30 fps, IMU at about 10 Hz (UAH) to 50-100 Hz (own phone), GPS at 1 Hz. Workstreams A and B must hand data to C without depending on each other's internals. Fusion needs aligned sequences and must handle one stream disappearing.

## Options considered
1. **Raw interface:** fusion model receives frames and raw IMU. Maximum flexibility; heavy, hard to parallelize, hard to run on the edge.
2. **Feature-level interface on a common clock:** each stream outputs `[T, D]` features at 10 Hz plus an availability mask. Light, testable with stubs, matches edge constraints. Loses information not captured by the features.
3. **Embedding-level interface only:** each branch outputs learned embeddings. Compact; couples training of branches to the fusion network.

## Decision
Use option 2 as the primary contract: 10 Hz common clock, `X_v [T,16]`, `X_m [T,14]`, masks, windows of 50 steps with stride 10. Keep an **optional clip tensor** for the 3D-CNN path (path + frame index, not stored pixels), whose embedding can be appended to `X_v`.

## Consequences
- Easy stubs and unit tests; A, B, C, D can work from week 2.
- Quality depends on the feature set. If error analysis shows a missing signal, bump the contract version and add the feature.
- 10 Hz may blur sub-100 ms events. Acceptable for the behaviours in scope (micro-sleep is 1-3 s, braking events are seconds).
- Wrong if: fusion results are poor and analysis shows high-rate vision or IMU detail matters. Then add a higher-rate branch for that stream.
