# ADR-0007: Driver keys and the class / event / alert taxonomy

Status: Accepted 2026-11-13 (Maruf approved both parts; both implemented)
Date: 2026-11-13
Deciders: Maruf (Claude drafted)

## Context
1. Driver IDs like `d03` can repeat across datasets (UAH, DMD, own), but splits and personalization key on the driver. `03` says `driver_id` is "str / int".
2. Three name lists describe abnormal behaviour and they do not match (`11_implementation_analysis.md` G21):
   - window classes: `normal, aggressive, drowsy, distracted` (`03` 5.1)
   - events: `harsh_accel, harsh_brake, sharp_turn, lane_weave, micro_sleep, distraction_episode` (`03` 5.2)
   - API alert types: `aggressive_accel, harsh_brake, sharp_turn, lane_weave, drowsy, micro_sleep, distracted` (`05` 3.4)

## Decision
**Part 1 (implemented).** The global driver key is `<dataset>:<driver_id>` (`schema.driver_key`). Splits, the leakage check and `WindowDataset.driver_id` use it. Anonymous tokens stay `d03` inside each dataset; the API maps `drv_xxxx` to the key on the server.

**Part 2 (implemented in `serving/schemas.py`; documented in `05` section 3.4).**
- Keep classes and events as in `03`.
- Alerts about events use the event name (`harsh_accel`, not `aggressive_accel`; `distraction_episode`).
- Alerts about a state use the class name (`drowsy`, `distracted`, `aggressive`).
- `predictive` alerts are allowed only for event types, because the forecaster predicts event onsets (`02` 4.7). The predictive `drowsy` example in `05` 3.4 would become `micro_sleep`.
- Add a `sharp_turn` template.

## Consequences
- Part 2 changed `05` and the API snapshot. `contract_version` stays `"0.1"` until the freeze. A test pins `AlertType` to `schema.EVENT_TYPES + schema.CLASSES[1:]`, so the three lists cannot drift apart again. The stub demo maps a class decision to the class name (a state alert).
- Wrong if: the guide wants alert names that match the classes exactly. Then drop the event/state split and rename events instead.
