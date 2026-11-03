"""Cutting the still parts and the windows works on a simple synthetic recording."""
import numpy as np
import pandas as pd

import config
from pipeline.preprocessing import walking_part, window_starts


def test_walking_part_and_windows():
    fs = 60
    speed = np.r_[np.zeros(2 * fs), np.full(16 * fs, 0.5), np.zeros(2 * fs)]  # still 2 s, walk 16 s, still 2 s
    walk = walking_part(pd.DataFrame({"velocity_Pelvis_x": speed, "velocity_Pelvis_y": 0.0}), fs)
    assert fs <= walk.index[0] <= 3 * fs and 17 * fs <= walk.index[-1] <= 19 * fs

    starts, length = window_starts(len(walk), fs)
    assert length == round(config.WINDOW_SECONDS * fs) and starts[-1] + length <= len(walk)
    assert set(np.diff(starts)) == {round(length * (1 - config.WINDOW_OVERLAP))}
