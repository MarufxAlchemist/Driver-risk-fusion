"""Motion features X_m (docs/03_data_contract.md section 4.2): raw phone IMU + GPS -> 14 features.

Everything here is **causal**: the value at step t uses only samples up to t, so the same code can
run live on the phone (P5) and offline on recorded data. The one thing that looks at a whole trip
is `estimate_alignment`, which is a calibration step; on the phone it runs once on the first
minutes and its result is passed back in.

Conventions (assumptions A18-A21 in docs/PROGRESS.md)
* Input units: acceleration in m/s^2 *including gravity* (a phone at rest reads +9.8 along the
  axis pointing up), gyro in deg/s, speed in m/s, compass heading in deg (clockwise, as GPS gives
  it). Loaders convert (UAH: G -> m/s^2, km/h -> m/s, contract section 4.4).
* Vehicle frame: x forward, y left, z up (right-handed). A left turn gives positive `acc_lat` and
  positive `yaw_rate`; `heading_rate` is `-d(compass)/dt` so it has the same sign as `yaw_rate`.
* Gravity is removed with the up axis measured while the vehicle is stationary (fallback: mean
  of the trip). A low-pass filter would also remove sustained accelerations such as a 3 s brake.
* The forward axis is found by regressing horizontal acceleration on the GPS speed change; the
  contract's "first stationary-to-driving transition" is replaced by all intervals of the trip
  (more robust, and needs no clean start). Without any acceleration in the speed profile the
  alignment fails: axis-dependent features are NaN and `Alignment.ok` is False.
* GPS speed and heading rate are held from the latest fix (no interpolation, which would need the
  next fix) and become NaN / `gps_valid = 0` once the fix is older than `gps_max_age_s`.
* The 14 features of the contract are all produced; `lane_offset` is NaN unless a lane series is
  passed in, and then `lateral_drift_5s` is its rolling std, else a proxy from `acc_lat`
  (std over 5 s of the detrended double integral, in metres, only an indicator of weaving).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from driverisk.data import schema as S

G_STD = 9.80665
_PREFIX = S.MOTION_PREFIX


@dataclass(frozen=True)
class MotionConfig:
    stationary_speed_mps: float = 0.3  # both fixes below this = the vehicle is standing
    stationary_min_s: float = 2.0  # least stationary time needed to trust the up axis
    min_accel_mps2: float = 0.3  # a GPS interval counts as accelerating above this
    min_accel_intervals: int = 3  # how many such intervals the forward-axis fit needs
    gps_max_gap_s: float = 2.5  # fixes further apart than this are not differenced
    gps_max_age_s: float = 2.0  # fix older than this: speed NaN, gps_valid 0
    imu_max_age_s: float = 0.2  # contract: mask_m needs an IMU sample this recent
    min_heading_speed_mps: float = 1.0  # compass heading is noise below walking pace
    attitude_tau_s: float = 2.0  # roll/pitch low-pass time constant
    jerk_smooth_steps: int = 3  # trailing mean applied before differencing
    window_s: float = 5.0  # speed variance and drift window
    min_window_steps: int = 10  # least valid steps for a rolling statistic


@dataclass(frozen=True)
class Alignment:
    ok: bool
    reason: str
    up_source: str  # "stationary" | "trip_mean"
    up_dev: np.ndarray  # [3] unit vector pointing up, in the phone frame
    g_dev: np.ndarray  # [3] gravity as the phone senses it at rest, in the phone frame
    R: np.ndarray | None  # [3, 3] rows x, y, z of the vehicle frame in the phone frame, or None

    def to_meta(self) -> dict:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "up_source": self.up_source,
            "up_dev": self.up_dev.tolist(),
            "R": None if self.R is None else self.R.tolist(),
        }


@dataclass
class MotionResult:
    frame: pd.DataFrame  # t, 14 m_* columns, mask_m (contract section 6.2 subset)
    alignment: Alignment
    all_nan_columns: list[str] = field(default_factory=list)  # declare these in trip meta


# --- helpers --------------------------------------------------------------------------------
def _bin_index(t: np.ndarray, dt: float) -> np.ndarray:
    """Step k covers (k*dt - dt, k*dt]; the epsilon keeps float noise out of the edges."""
    return np.ceil(np.asarray(t, dtype=np.float64) / dt - 1e-9).astype(np.int64)


def _bin_mean(t: np.ndarray, x: np.ndarray, n: int, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Mean of `x [m, d]` per step (NaN where empty) and the sample count per step."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    idx = _bin_index(t, dt)
    keep = (idx >= 0) & (idx < n) & np.isfinite(x).all(axis=1)
    idx, x = idx[keep], x[keep]
    count = np.bincount(idx, minlength=n).astype(np.int64)
    out = np.full((n, x.shape[1]), np.nan)
    for j in range(x.shape[1]):
        total = np.bincount(idx, weights=x[:, j], minlength=n)
        out[count > 0, j] = total[count > 0] / count[count > 0]
    return out, count


def _bin_last(t: np.ndarray, x: np.ndarray, n: int, dt: float) -> np.ndarray:
    idx = _bin_index(t, dt)
    x = np.asarray(x, dtype=np.float64)
    out = np.full(n, np.nan)
    keep = (idx >= 0) & (idx < n) & np.isfinite(x)
    out[idx[keep]] = x[keep]  # later samples overwrite earlier ones inside a step
    return out


def _ffill_one(x: np.ndarray) -> np.ndarray:
    """Fill a single empty step from its predecessor (an IMU sample 0.1-0.2 s old is current)."""
    out = x.copy()
    empty = np.isnan(x).any(axis=1)
    idx = np.flatnonzero(empty)
    idx = idx[idx > 0]
    idx = idx[~empty[idx - 1]]  # only when the previous step has data
    out[idx] = x[idx - 1]
    return out


def _hold_last_fix(t_grid: np.ndarray, t_fix: np.ndarray, values: np.ndarray, max_age: float):
    """Value of the latest fix at or before each grid time; NaN when older than `max_age`."""
    pos = np.searchsorted(t_fix, t_grid + 1e-9, side="right") - 1
    out = np.full(len(t_grid), np.nan)
    fresh = pos >= 0
    age = np.where(fresh, t_grid - t_fix[np.clip(pos, 0, None)], np.inf)
    fresh &= age <= max_age + 1e-9
    out[fresh] = values[pos[fresh]]
    return out, fresh


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


# --- alignment ------------------------------------------------------------------------------
def estimate_alignment(
    t_acc: np.ndarray,
    acc: np.ndarray,
    t_gps: np.ndarray,
    speed_mps: np.ndarray,
    config: MotionConfig | None = None,
) -> Alignment:
    """Find the up axis (gravity) and the forward axis of the vehicle in the phone frame."""
    cfg = config or MotionConfig()
    t_acc, acc = np.asarray(t_acc, dtype=np.float64), np.asarray(acc, dtype=np.float64)
    t_gps, speed = np.asarray(t_gps, dtype=np.float64), np.asarray(speed_mps, dtype=np.float64)

    still = np.zeros(len(t_acc), dtype=bool)
    still_s = 0.0
    for k in range(len(t_gps) - 1):
        dt = t_gps[k + 1] - t_gps[k]
        if dt <= cfg.gps_max_gap_s and max(speed[k], speed[k + 1]) < cfg.stationary_speed_mps:
            still |= (t_acc >= t_gps[k]) & (t_acc <= t_gps[k + 1])
            still_s += dt
    if still_s >= cfg.stationary_min_s and still.any():
        rest = acc[still].mean(axis=0)
        up_source, g_dev = "stationary", rest
    else:
        rest = acc.mean(axis=0)
        up_source, g_dev = "trip_mean", _unit(rest) * G_STD
    up = _unit(rest)

    lin = acc - g_dev
    horizontal = lin - np.outer(lin @ up, up)
    num, den, n_acc = np.zeros(3), 0.0, 0
    for k in range(len(t_gps) - 1):
        dt = t_gps[k + 1] - t_gps[k]
        if not 0 < dt <= cfg.gps_max_gap_s:
            continue
        a_gps = (speed[k + 1] - speed[k]) / dt
        sel = (t_acc > t_gps[k]) & (t_acc <= t_gps[k + 1])
        if not sel.any():
            continue
        num += a_gps * horizontal[sel].mean(axis=0)
        den += a_gps**2
        n_acc += abs(a_gps) > cfg.min_accel_mps2
    if n_acc < cfg.min_accel_intervals or den == 0:
        return Alignment(
            False, "the speed profile has no acceleration to find the forward axis", up_source,
            up, g_dev, None,
        )  # fmt: skip
    fwd = num / den
    if np.linalg.norm(fwd) < 0.2:  # horizontal acceleration unrelated to the speed change
        return Alignment(
            False, "horizontal acceleration does not follow the GPS speed change", up_source,
            up, g_dev, None,
        )  # fmt: skip
    x_axis = _unit(fwd)
    y_axis = np.cross(up, x_axis)  # z x x = y, so y points left
    return Alignment(True, "ok", up_source, up, g_dev, np.stack([x_axis, y_axis, up]))


# --- feature extraction ---------------------------------------------------------------------
def _rolling(series: np.ndarray, window: int, min_periods: int, how: str) -> np.ndarray:
    roll = pd.Series(series).rolling(window, min_periods=min_periods)
    return (roll.var(ddof=0) if how == "var" else roll.std(ddof=0)).to_numpy()


def _drift_proxy(acc_lat: np.ndarray, window: int, dt: float) -> np.ndarray:
    """Std over `window` steps of the detrended double integral of lateral acceleration."""
    out = np.full(len(acc_lat), np.nan)
    idx = np.arange(window)
    for k in range(window - 1, len(acc_lat)):
        w = acc_lat[k - window + 1 : k + 1]
        if np.isnan(w).any():
            continue
        v = np.cumsum(w - w.mean()) * dt
        d = np.cumsum(v - v.mean()) * dt
        d = d - np.polyval(np.polyfit(idx, d, 1), idx)
        out[k] = d.std()
    return out


def extract_motion_features(
    t_acc: np.ndarray,
    acc: np.ndarray,
    t_gps: np.ndarray,
    speed_mps: np.ndarray,
    *,
    gyro_dps: np.ndarray | None = None,
    heading_deg: np.ndarray | None = None,
    yaw_deg: tuple[np.ndarray, np.ndarray] | None = None,
    lane_offset: tuple[np.ndarray, np.ndarray] | None = None,
    alignment: Alignment | None = None,
    config: MotionConfig | None = None,
) -> MotionResult:
    """Compute the 14 motion features on the 10 Hz grid.

    `t_acc [m]`, `acc [m, 3]` (m/s^2 with gravity), optional `gyro_dps [m, 3]` at the same
    times, `t_gps [g]`, `speed_mps [g]`, optional `heading_deg [g]`. `yaw_deg=(t, deg)` is the
    fallback yaw-rate source when there is no gyro (angle must be left-positive).
    `lane_offset=(t, metres)` comes from UAH lane detection. Times are seconds since trip start.
    """
    cfg = config or MotionConfig()
    dt = S.STEP_S
    t_acc, acc = np.asarray(t_acc, dtype=np.float64), np.asarray(acc, dtype=np.float64)
    t_gps, speed = np.asarray(t_gps, dtype=np.float64), np.asarray(speed_mps, dtype=np.float64)
    n = int(np.floor(t_acc[-1] / dt + 1e-9)) + 1  # steps 0 .. last complete one
    grid = np.arange(n) * dt
    if alignment is None:
        alignment = estimate_alignment(t_acc, acc, t_gps, speed, cfg)
    up, ok = alignment.up_dev, alignment.ok

    acc10, count = _bin_mean(t_acc, acc, n, dt)
    acc10 = _ffill_one(acc10)
    steps_back = round(cfg.imu_max_age_s / dt)  # a sample in (t - 0.2 s, t] keeps the mask on
    mask_m = np.convolve(count > 0, np.ones(steps_back), mode="full")[:n] > 0
    lin = acc10 - alignment.g_dev
    cols: dict[str, np.ndarray] = {}

    # attitude of the phone from the low-passed accelerometer (a small share is real acceleration)
    alpha = 1.0 - np.exp(-dt / cfg.attitude_tau_s)
    g_att = np.stack(
        [
            pd.Series(acc10[:, j]).ewm(alpha=alpha, adjust=False, ignore_na=True).mean().to_numpy()
            for j in range(3)
        ],
        axis=1,
    )
    g_att[np.isnan(acc10).any(axis=1)] = np.nan
    cols["roll"] = np.degrees(np.arctan2(g_att[:, 1], g_att[:, 2]))
    cols["pitch"] = np.degrees(np.arctan2(-g_att[:, 0], np.hypot(g_att[:, 1], g_att[:, 2])))

    # vehicle-frame acceleration and jerk
    cols["acc_vert"] = lin @ up
    nan = np.full(n, np.nan)
    if ok and alignment.R is not None:
        veh = lin @ alignment.R.T
        cols["acc_long"], cols["acc_lat"] = veh[:, 0], veh[:, 1]
    else:
        cols["acc_long"], cols["acc_lat"] = nan.copy(), nan.copy()
    for src, dst in (("acc_long", "jerk_long"), ("acc_lat", "jerk_lat")):
        smooth = pd.Series(cols[src]).rolling(cfg.jerk_smooth_steps, min_periods=1).mean()
        cols[dst] = smooth.diff().to_numpy() / dt  # backward difference: causal

    # yaw rate: gyro about the up axis, else the unwrapped yaw angle
    if gyro_dps is not None:
        gyro10, _ = _bin_mean(t_acc, gyro_dps, n, dt)
        cols["yaw_rate"] = _ffill_one(gyro10) @ up
    elif yaw_deg is not None:
        yaw = _bin_last(yaw_deg[0], yaw_deg[1], n, dt)
        good = ~np.isnan(yaw)
        unwrapped = np.full(n, np.nan)
        unwrapped[good] = np.degrees(np.unwrap(np.radians(yaw[good])))
        cols["yaw_rate"] = np.diff(unwrapped, prepend=np.nan) / dt
    else:
        cols["yaw_rate"] = nan.copy()

    # GPS: speed, validity, variance, heading rate
    cols["speed"], fresh = _hold_last_fix(grid, t_gps, speed, cfg.gps_max_age_s)
    cols["gps_valid"] = fresh.astype(np.float64)
    window = round(cfg.window_s / dt)
    cols["speed_var_5s"] = _rolling(cols["speed"], window, cfg.min_window_steps, "var")
    if heading_deg is not None:
        h = np.degrees(np.unwrap(np.radians(np.asarray(heading_deg, dtype=np.float64))))
        d_t = np.diff(t_gps, prepend=np.nan)
        rate = -np.diff(h, prepend=np.nan) / d_t  # left turn positive
        rate[~(d_t <= cfg.gps_max_gap_s) | ~(speed >= cfg.min_heading_speed_mps)] = np.nan
        cols["heading_rate"], _ = _hold_last_fix(grid, t_gps, rate, cfg.gps_max_age_s)
    else:
        cols["heading_rate"] = nan.copy()

    # lane offset and lateral drift
    if lane_offset is not None:
        cols["lane_offset"] = _bin_last(lane_offset[0], lane_offset[1], n, dt)
        cols["lateral_drift_5s"] = _rolling(
            cols["lane_offset"], window, cfg.min_window_steps, "std"
        )
    else:
        cols["lane_offset"] = nan.copy()
        cols["lateral_drift_5s"] = _drift_proxy(cols["acc_lat"], window, dt)

    for name in ("acc_long", "acc_lat", "acc_vert", "jerk_long", "jerk_lat", "yaw_rate", "roll",
                 "pitch"):  # fmt: skip
        cols[name] = np.where(mask_m, cols[name], np.nan)  # no IMU, no IMU-derived value

    frame = pd.DataFrame({S.COL_T: grid.astype(np.float32)})
    for name in S.MOTION_NAMES:
        frame[_PREFIX + name] = cols[name].astype(np.float32)
    frame[S.COL_MASK_M] = mask_m.astype(bool)
    all_nan = [c for c in S.MOTION_COLUMNS if frame[c].isna().all()]
    return MotionResult(frame, alignment, all_nan)
