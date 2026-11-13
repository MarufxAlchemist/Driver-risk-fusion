# 05. Annotation and agreement (H9)

Used for own trips (`06` section 7). No special tool is needed: annotators write interval CSVs while watching the video and the plotted signals.

## CSV format
One row per event, file per annotator, columns exactly:

`trip_id,type,onset_s,offset_s,uncertain` (see `templates/annotation_template.csv`)

- `trip_id`: as in the manifest, for example `own_d01_20261201_01`.
- `type`: one of `harsh_accel, harsh_brake, sharp_turn, lane_weave, micro_sleep, distraction_episode`.
- `onset_s`, `offset_s`: seconds since trip start on the common clock (after sync).
- `uncertain`: `true` when the boundaries are unclear. These rows are excluded from headline metrics but counted.

## Rules
- `micro_sleep`: eyes closed for 1.0 to 3.0 s with no blink recovery.
- `distraction_episode`: gaze off the forward zone for more than 2 s, or a hand-device interaction visible.
- Posed events (parked) get a note in the trip sheet; they are labelled posed downstream.
- Harsh accel/brake/turn candidates come from `scripts/auto_events.py` (P1-B3); the annotator confirms or rejects them and marks uncertain ones.
- Annotators work independently and do not compare before submitting.

## Who and how much
- Primary annotator: whoever knows the trips. Second annotator (H9): a different person, on a random **20 percent** of trips (at least a few trips with real events, not only "no event" ones).

## Compute agreement
```
uv run python scripts/annotation_agreement.py annotator_a.csv annotator_b.csv ^
    --manifest data\processed\own\manifest.csv
```
(On Mac/Linux use `\` instead of `^`; the manifest is optional but gives exact trip lengths.)

It prints, per event type: Cohen's kappa on 10 Hz step labels, F1 of matched intervals (IoU at least 0.5, one-to-one), mean IoU, counts, and how many uncertain rows were excluded. Kappa is `nan` when neither annotator marked a type.

Report all of these in `experiments/` and in the report; do not pick only the good ones. If agreement is low for a type, discuss the rule with the other annotator before labelling more.

## Files
Annotation CSVs contain no video or faces and may be committed under `experiments/annotations/`; the trips are still referred to by anonymous IDs only.
