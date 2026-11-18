# 03. Guide meeting: one-page summary (H5, H6, H11)

Bring this page. Goal: get the **approvals** at the bottom. Nothing is recorded in a car until they are given.

## The project in three lines
A driver-monitoring system that fuses a driver-facing camera with phone IMU/GPS, learns each driver's own "normal", warns a few seconds before risky behaviour, and says why. Features only leave the device, never raw video.

## The plan in one table
| Phase | Weeks | Result |
|---|---|---|
| P0 Foundations | 1-2 | Repo, contract, stubs, mock API: **done, see `docs/PROGRESS.md`** |
| P1 Data and features | 3-6 | UAH + DMD loaded; pilot own trips |
| P2 Baselines | 5-8 | Camera-only, sensor-only, autoencoder; 3D-CNN go/no-go |
| P3 Fusion | 8-12 | Cross-attention fusion, modality dropout, robustness |
| P4 Personalization | 11-16 | Personal baselines, forecaster, explanations |
| P5-P7 | 14-24 | Live demo, final evaluation, reports |

## The problem the guide should know about: paired data
UAH-DriveSet has phone motion but no driver-face video. DMD has face video but no vehicle motion. **No public dataset has both for the same trip** (to be confirmed once the downloads are inspected). So the fusion claim is tested on our **own paired recordings**; public data only pre-trains each branch. Results from "synthetic pairing" of unrelated trips are used only to study robustness and are always labelled that way. With few drivers, results are a pilot study with per-driver variance, not a population claim.

## Safety rules (from `06_data_collection_protocol.md`)
- Nobody tests unsafe driving on public roads: no deliberate drowsiness, texting, reaching or dangerous manoeuvres while moving.
- Drowsy, yawning, looking-away and reaching behaviours are **posed only with the car parked**, with a second person present; reports say which data is posed.
- "Assertive" driving means moderately firm acceleration and braking within the law, on an empty road, with a passenger.
- A passenger operates the recording; the driver never touches the phone or laptop while moving. Only licensed drivers drive; seatbelts on.

## Consent and privacy
Written consent (`docs/reports/consent_form_template.md`): adults, volunteers, random IDs, name-to-ID table outside the repo, raw video on a private drive and deleted after the project, withdrawal on request, GPS route stored only with consent.

## Open questions for the guide
1. **Approve** the data collection protocol and the consent form (adapt to college rules).
2. **AI-assisted development.** The code is written with an AI coding assistant under the student's direction, and each phase is reviewed by the student. How should this be disclosed in the report and viva, and is it acceptable? (Required before relying on this workflow.)
3. **Team arrangement.** The documents are written for four members; the build is currently done by one. Confirm the division and what is expected from the others (recording, annotation, report chapters).
4. **Compute:** college GPU lab access?
5. **Scope guard:** if behind, the cut order is 3D-CNN depth, context-feature ablation, dashboard polish, user study size. Agree this now.
6. **Alert names** (ADR-0007): classes and alert types do not match in the current documents; preferred wording?
7. **Licence:** MIT for our code (the repo has an MIT file); confirm the college allows it.
8. **Public repo?** Paper PDFs have been removed from tracking because several are licensed IEEE copies.
9. FA/h budget of 6 per hour (`04` 3.2) is a placeholder: confirm or change.

## Approvals to record (date, initials)
- [ ] Data collection protocol and consent form
- [ ] AI-assisted development disclosure
- [ ] Division of work
- [ ] Scope guard order
