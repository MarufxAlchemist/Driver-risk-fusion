# ADR-0001: Documentation-first monorepo with phase gates

Status: Proposed
Date: 2026-10-18
Deciders: Anuja Naik, Tanishq Basagare, Maruf Nadaf, Gautam Iyengar

## Context
Four people with different strengths will work in parallel for about 24 weeks. The project has many interacting parts (vision, motion, fusion, personalization, forecasting, API, dashboard). Without agreed interfaces, parallel work leads to incompatible code. The guide also needs to be able to see progress at fixed points.

## Options considered
1. **Code first, document at the end.** Fast start; high risk of integration failure and of a report written from memory.
2. **Multiple repositories per component.** Clean ownership; overhead for a small team and harder integration.
3. **Single monorepo, docs-first, with phase gates.** Slower first two weeks; shared structure, one CI, interfaces agreed up front.

## Decision
Use one monorepo (`driver-risk-fusion`). Write the plan, architecture, data contract, experiment protocol and API contract before model code. Work in phases P0-P7 with an exit criterion for each, checked in a short demo.

License: default MIT for our code unless the guide or college requires otherwise (decision D8). Dataset licenses are separate.

## Consequences
- Weeks 1-2 produce documents, stubs and CI, not models. This is intentional.
- Any interface change requires a PR to the contract docs.
- Phase demos create natural material for guide meetings and the mid-term report.
- If a phase gate is missed we re-plan instead of drifting.
- Wrong if: contracts change every week, which would mean they were frozen too early. In that case freeze later but keep stubs.
