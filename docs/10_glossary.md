# 10. Glossary

Terms in the order a new teammate is likely to meet them.

## Domain terms

| Term | Meaning |
|---|---|
| Abnormal driving | Driving behavior that deviates from safe, normal driving: aggressive acceleration, sudden braking, lane weaving, micro-sleep, distraction |
| Driver monitoring system (DMS) | In-vehicle system that watches the driver (usually with a camera) for drowsiness and distraction |
| ADAS | Advanced driver assistance systems (lane keeping, collision warning, etc.). Future scope only |
| IMU | Inertial measurement unit: accelerometer (and gyroscope) in the phone |
| GPS | Position and speed from satellites; low rate (about 1 Hz) |
| Jerk | Rate of change of acceleration (m/s^3). High jerk means abrupt acceleration or braking |
| Yaw rate | Rotation rate around the vertical axis (deg/s). Shows steering or turning |
| Lateral drift | Sideways movement within a lane |
| Lane weaving | Repeated drifting across the lane position |
| Micro-sleep | Brief (about 1-3 s) loss of attention with eyes closed |
| EAR | Eye aspect ratio from eye landmarks. Low values mean closed eyes |
| MAR | Mouth aspect ratio. High values suggest yawning |
| PERCLOS | Percentage of eye closure over a time window. Standard drowsiness measure. We use 30 s and 60 s windows |
| Head pose | Head orientation (yaw, pitch, roll) estimated from landmarks |
| Gaze | Direction the eyes are looking. We use a proxy from head pose and iris landmarks |
| Naturalistic driving | Real driving data collected in normal traffic, not in a lab |

## Machine-learning terms

| Term | Meaning |
|---|---|
| Multi-sensor fusion | Combining information from several sensors for one decision |
| Early / late fusion | Combining raw or low-level features early, versus combining model outputs or embeddings late |
| Cross-attention | Attention where queries come from one stream and keys/values from another, so each stream can pick out relevant information from the other |
| Modality dropout | Randomly hiding a whole modality during training so the model learns to work when one sensor fails |
| Availability mask | Boolean per step saying whether a modality has valid data |
| LSTM | Long short-term memory network, a recurrent model for sequences |
| Autoencoder | Network trained to reconstruct its input. Trained on normal data only, high reconstruction error suggests abnormal data |
| 3D-CNN | Convolutional network with convolutions over space and time on video clips |
| 2D CNN + LSTM | Frame-wise 2D CNN features followed by an LSTM over time. Our baseline for the 3D-CNN |
| Embedding | Vector summarizing a window of input |
| Shrinkage | Blending a per-driver statistic with the global statistic when the driver has little data |
| Cold start | The period before a new driver has enough calibration data |
| Forecaster | Stage 3 model predicting event onset within the next H seconds |
| Calibration (model) | Whether predicted probabilities match observed frequencies (ECE measures this) |
| Calibration (driver) | Collecting the first trips to learn a driver's own normal |
| ONNX | Open model exchange format for deployment on edge devices |

## Evaluation terms

| Term | Meaning |
|---|---|
| Macro-F1 | F1 score averaged over classes with equal weight |
| Recall | Fraction of true events detected |
| FA/h | False alarms per hour of normal driving |
| Lead time | How many seconds before the event onset the alert fired |
| ECE | Expected calibration error |
| PR-AUC | Area under the precision-recall curve; better than ROC-AUC for rare events |
| Brier score | Mean squared error of predicted probabilities |
| Ablation | Experiment that removes or replaces one component to measure its contribution |
| Driver-disjoint split | Train, validation and test sets contain different drivers |
| Leakage | Information from the test set influencing training or tuning |
| Seed | Random initialization setting; three or more seeds give mean and std |

## Project terms

| Term | Meaning |
|---|---|
| Common clock | 10 Hz time base that all modalities are resampled to |
| Window | 50 steps (5 s) cut with a 1 s stride |
| Contract | `03_data_contract.md`, the frozen data interface |
| Pairing | Whether a trip has both camera and motion data (`paired`) or one (`vision_only`, `motion_only`) |
| F1 / F2 / F3 | Fusion evaluation settings in `04_experiment_protocol.md` (own paired / pretrain + finetune / synthetic pairing) |
| Edge | The device in the car (phone or laptop standing in for it) |
| ADR | Architecture decision record in `docs/decisions/` |
| Workstream A-D | Vision, motion, fusion/personalization/forecasting, platform/delivery |
| MoSCoW | Must, Should, Could, Won't priority labels |
| Phase gate | Exit check at the end of each phase |
