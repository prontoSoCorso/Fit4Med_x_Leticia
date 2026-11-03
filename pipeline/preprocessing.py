"""Signal-level helpers: quality numbers of a recording.

The pelvis is the body's reference point for walking: its horizontal path is the distance walked.
"""
import numpy as np
import pandas as pd

PELVIS_POSITION_XY = ["position_Pelvis_x", "position_Pelvis_y"]  # metres, on the floor plane


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
