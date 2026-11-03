"""Hand-made features of one window, for the classical models (SVR, kNN, Random Forest).

A feature is one number that summarises a window, e.g. the knee range of motion.
Four families (slide 6): time domain, frequency, movement quality, bilateral (left vs right).
"""
import numpy as np
from scipy.signal import find_peaks

from pipeline.preprocessing import pelvis_speed

GAIT_BAND_HZ = (0.5, 3.0)  # the walking rhythm (strides and steps) lies in this frequency band
JOINTS = ["Hip", "Knee", "Ankle"]
FLEXION = "z"  # in Xsens joint angles the z axis is flexion/extension (largest range when walking)


def time_features(signals):
    """Mean, spread (standard deviation), root mean square and range of each channel."""
    out = {}
    for name, x in signals.items():
        out |= {f"{name}_mean": x.mean(), f"{name}_std": x.std(),
                f"{name}_rms": np.sqrt(np.mean(x ** 2)), f"{name}_range": np.ptp(x)}
    return out


def frequency_features(signals, fs):
    """Dominant frequency, spectral entropy (how spread the rhythm is) and share of power in the gait band."""
    out = {}
    freqs = np.fft.rfftfreq(len(next(iter(signals.values()))), 1 / fs)
    band = (freqs >= GAIT_BAND_HZ[0]) & (freqs <= GAIT_BAND_HZ[1])
    for name, x in signals.items():
        power = np.abs(np.fft.rfft(x - x.mean())) ** 2
        share = power / (power.sum() + 1e-12)
        out |= {f"{name}_dominant_hz": freqs[np.argmax(power)],
                f"{name}_spectral_entropy": -np.sum(share * np.log(share + 1e-12)),
                f"{name}_gait_band_power": share[band].sum()}
    return out


def quality_features(window, fs):
    """Movement quality of the pelvis: mean and peak speed, number of speed peaks, jerk (low = smooth)."""
    speed = pelvis_speed(window).to_numpy()
    jerk = np.diff(window["acceleration_Pelvis_z"].to_numpy()) * fs  # how fast the acceleration changes
    return {"pelvis_mean_speed": speed.mean(), "pelvis_peak_speed": speed.max(),
            "pelvis_speed_peaks": len(find_peaks(speed, prominence=0.05)[0]),
            "pelvis_jerk_rms": np.sqrt(np.mean(jerk ** 2))}


def asymmetry(left, right):
    """0 = both sides equal, 1 = one side does everything. It does not need to know the affected side."""
    return abs(left - right) / (abs(left) + abs(right) + 1e-12)


def bilateral_features(window, fs):
    """Left vs right: joint range of motion, coordination, time with the foot on the ground, cadence."""
    out = {}
    for joint in JOINTS:
        left = window[f"jointAngle_jLeft{joint}_{FLEXION}"].to_numpy()
        right = window[f"jointAngle_jRight{joint}_{FLEXION}"].to_numpy()
        out[f"{joint.lower()}_range_asymmetry"] = asymmetry(np.ptp(left), np.ptp(right))
        out[f"{joint.lower()}_left_right_correlation"] = np.corrcoef(left, right)[0, 1]
    stance, heel_strikes = {}, 0
    for side in ("Left", "Right"):
        stance[side] = window[[f"footContacts_{side}Foot_Heel", f"footContacts_{side}Foot_Toe"]].max(axis=1).mean()
        heel_strikes += np.sum(np.diff(window[f"footContacts_{side}Foot_Heel"].to_numpy()) == 1)
    out["stance_asymmetry"] = asymmetry(stance["Left"], stance["Right"])
    out["cadence_steps_per_min"] = heel_strikes / (len(window) / fs) * 60
    return out


def window_features(window, channels, fs):
    """All features of one window, as {feature name: value}."""
    signals = {c: window[c].to_numpy() for c in channels}
    return (time_features(signals) | frequency_features(signals, fs)
            | quality_features(window, fs) | bilateral_features(window, fs))
