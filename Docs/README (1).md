# Architecture Decision Records (ADRs)

A short note for every decision that is expensive to reverse or that someone will ask "why?" about later (including the guide and external examiners).

## Rules

- One file per decision: `ADR-NNNN-short-title.md`. Numbers never change or get reused.
- Status: `Proposed` -> `Accepted` (after the weekly sync) -> optionally `Superseded by ADR-XXXX`.
- Keep it under one page. Options, decision, consequences.
- Do not edit an accepted ADR to change the decision. Write a new ADR that supersedes it.

## Index

| ADR | Title | Status |
|---|---|---|
| 0001 | Documentation-first monorepo with phase gates | Proposed |
| 0002 | Feature-level interface with a common 10 Hz clock | Proposed |
| 0003 | Fusion training data strategy (paired data problem) | Proposed |
| 0004 | 3D-CNN as a candidate backbone, validated by ablation | Proposed |
| 0005 | Driver-disjoint splits and within-driver personalization evaluation | Proposed |
| 0006 | Technology stack and edge export path | Proposed |

## Template

```markdown
# ADR-NNNN: Title

Status: Proposed | Accepted | Superseded by ADR-XXXX
Date: YYYY-MM-DD
Deciders: names

## Context
What forces the decision? Facts, constraints, deadlines.

## Options considered
1. Option A: pros, cons
2. Option B: pros, cons

## Decision
What we chose, in one or two sentences.

## Consequences
What gets easier, what gets harder, what we must do next, how we will know if it was wrong.
```
