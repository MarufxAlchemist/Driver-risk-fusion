# 04. Own-data recording: tests and procedure (H4, H7, H8)

Source of truth for safety and consent: `docs/06_data_collection_protocol.md`. This page is the practical checklist.

## 0. Before anything
- [ ] Guide approved the protocol and consent form (`03_guide_meeting.md`).
- [ ] You have a **passenger** for every trip. Recording alone is not allowed.
- [ ] Consent signed by every driver; the name-to-ID list is kept **outside** the repo.

## 1. Choose the logging app (decision D6) and test it, parked, 5 minutes
Candidates: Sensor Logger, phyphox, or a custom app. It must export **timestamped CSV** with accelerometer (and gyroscope, GPS if possible) on **both Android and iOS** if both will be used.

- [ ] Phone clock set to network time. Note the app and its version.
- [ ] Start the log, then the camera recording (video from the same phone or a second one, and note which).
- [ ] **Sync event:** driver blinks slowly three times and says "start", then the passenger taps the phone firmly **twice** (about half a second apart). Wait 2 s.
- [ ] Sit for about 3 minutes (parked). Tilt the phone gently once.
- [ ] Repeat the double tap at the end. Stop both recordings.
- [ ] Export the CSV and copy it to the computer (not into the repo).
- [ ] Run:

```
uv run python scripts/check_phone_log.py PATH\TO\accelerometer.csv --acc-cols x y z
```

Replace `x y z` with the CSV's accelerometer column names. It prints: time unit, duration, median rate, jitter, gaps over 0.5 s, and the times of tap pairs.

| Result you want | Why |
|---|---|
| `RESULT: OK` | rate at least 10 Hz, no gaps over 0.5 s, jitter under 0.25 |
| two tap pairs found (start and end) | needed for offset and drift (`03` section 8) |
| video shows the taps/blinks | compare the video frame times of the blinks with the IMU tap times; the difference is your sync offset, target under 100 ms |

- [ ] Send Claude the CSV header line and the script output (not the video). This decides the CSV reader in P1-D1.
- [ ] Record in the dataset notes: phone model, app and version, timestamp format, rates.

## 2. Parked-car test (pipeline test, nobody drives)
- [ ] Mount the phone as in `06` section 3; photo of the mount for the log.
- [ ] Record 10 minutes: normal sitting, then **posed** segments (each 3 to 5 times): eyes closed 1 to 3 s, yawn, head nod, looking at the phone, looking at the side window, reaching toward the passenger seat. Passenger notes the approximate times.
- [ ] Degradation tests (parked): cover the camera for a few seconds, tilt the phone, sunglasses, cap, dim the cabin light.
- [ ] These posed segments are labelled **posed** and are never reported as real drowsiness.

## 3. Pilot trips (after approval)
2 drivers x 2 trips, 15 to 20 minutes, a familiar quiet route. Follow `06` section 5. Fill a row of `templates/trip_sheet_template.csv` for every trip the same day, and back up the files to the team drive the same day. File names: `<dataset>_<driverid>_<yyyymmdd>_<nn>` plus `_video`, `_imu`, `_gps` (the same ID as the trip ID).

## 4. Quality gate for a trip to count (`06` section 9)
Face visible in at least 85 percent of normal steps; no IMU gap over 0.5 s; GPS fix for 90 percent of the trip; both sync events detected with residual under 100 ms; at least one event or an explicit "no events". Failed trips are kept and flagged low quality.

## 5. Stop conditions
Stop immediately if anyone feels unsafe, tired or uncomfortable. If in doubt about an activity, do it parked or skip it.
