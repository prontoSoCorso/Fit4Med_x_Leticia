"""Signal-level helpers: quality numbers, cutting the still parts, choosing channels, cutting windows.

The pelvis is the body's reference point for walking: its horizontal path is the distance walked.
"""
import numpy as np
import pandas as pd

import config

PELVIS_POSITION_XY = ["position_Pelvis_x", "position_Pelvis_y"]  # metres, on the floor plane
PELVIS_VELOCITY_XY = ["velocity_Pelvis_x", "velocity_Pelvis_y"]  # m/s, on the floor plane


def walked_distance(table):
    """Length of the pelvis path on the floor, in metres."""
    steps = np.diff(table[PELVIS_POSITION_XY].to_numpy(), axis=0)
    return float(np.nansum(np.hypot(steps[:, 0], steps[:, 1])))


def recording_stats(table):
    """Duration, frames, missing values (NaN), repeated time stamps, largest gap, distance and mean speed."""
    time = table["time_s"].to_numpy()
    duration = time[-1] - time[0]
    distance = walked_distance(table)
    return {"duration_s": round(duration, 2), "n_frames": len(table),
            "n_nan": int(table.isna().sum().sum()),
            "n_repeated_times": int(pd.Series(time).duplicated().sum()),
            "max_gap_s": round(float(np.diff(time).max()), 3),  # much larger than 1/60 s = lost frames
            "distance_m": round(distance, 2), "mean_speed_m_s": round(distance / duration, 2)}


def pelvis_speed(table):
    """Horizontal speed of the pelvis (m/s) at each frame."""
    return pd.Series(np.hypot(*table[PELVIS_VELOCITY_XY].to_numpy().T), index=table.index)


def walking_part(table, fs):
    """Cut the standing-still parts at the start and at the end of a recording."""
    frames = max(1, int(config.SMOOTHING_SECONDS * fs))
    smooth = pelvis_speed(table).rolling(frames, center=True, min_periods=1).mean()
    moving = np.flatnonzero(smooth.to_numpy() > config.STILL_SPEED_THRESHOLD)
    return table.iloc[moving[0]:moving[-1] + 1] if len(moving) else table.iloc[0:0]


def channel_columns(table):
    """The columns named in config.CHANNELS (each name selects the columns starting with it)."""
    columns = []
    for prefix in config.CHANNELS:
        found = [c for c in table.columns if c.startswith(prefix)]
        if not found:
            raise ValueError(f"No column starts with '{prefix}': check CHANNELS in config.py.")
        columns += found
    return columns


def window_starts(n_frames, fs):
    """First frame of each window: WINDOW_SECONDS long, overlapping by WINDOW_OVERLAP."""
    length = int(round(config.WINDOW_SECONDS * fs))
    step = max(1, int(round(length * (1 - config.WINDOW_OVERLAP))))
    return list(range(0, n_frames - length + 1, step)), length
