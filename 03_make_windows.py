"""Step 3: cut the walking part of each recording into windows and compute hand-made features.

Reads:  results/inventory.csv (recordings with status ok), data/extracted/*.csv, data/labels.csv
Writes: data/windows/windows.npy   (windows x channels x frames: input of the deep and foundation models)
        data/windows/windows.csv   (one row per window: subject, file, speed, start_s, label, metadata)
        data/windows/features.csv  (one row per window: hand-made features for the classical models)
"""
import sys

import numpy as np
import pandas as pd

import config
from pipeline.features import window_features
from pipeline.preprocessing import channel_columns, walking_part, window_starts


def load_labels():
    """One row per subject: the label at TARGET_TIMEPOINT and the metadata that exist."""
    if not config.LABELS_CSV.exists():
        sys.exit(f"{config.LABELS_CSV} not found: it needs the columns subject, timepoint, {config.TARGET}.")
    labels = pd.read_csv(config.LABELS_CSV)
    target = labels[labels["timepoint"] == config.TARGET_TIMEPOINT].set_index("subject")[config.TARGET]
    if target.index.duplicated().any():
        sys.exit(f"labels.csv has two rows for {sorted(set(target.index[target.index.duplicated()]))} "
                 f"at {config.TARGET_TIMEPOINT}: keep one.")
    metadata = [c for c in config.METADATA if c in labels]
    per_subject = labels.groupby("subject")[metadata].first()  # age, BMI, hemisphere do not change
    per_subject["label"] = target
    return per_subject.dropna(subset=["label"])


def usable_recordings():
    """Recordings with status ok, of the analysed test, timepoint and speeds."""
    path = config.RESULTS_DIR / "inventory.csv"
    if not path.exists():
        sys.exit(f"{path} not found: run 02_inventory.py first.")
    inventory = pd.read_csv(path)
    return inventory[(inventory["status"] == "ok") & (inventory["test"] == config.TEST)
                     & (inventory["timepoint"] == config.INPUT_TIMEPOINT) & inventory["speed"].isin(config.SPEEDS_USED)]


def main():
    labels, recordings = load_labels(), usable_recordings()
    if recordings["frame_rate"].nunique() > 1:
        sys.exit(f"The recordings have different frame rates {sorted(recordings['frame_rate'].unique())}: "
                 "windows would differ in length (resampling is not implemented).")
    print(f"No {config.TARGET} at {config.TARGET_TIMEPOINT}, skipped: {sorted(set(recordings['subject']) - set(labels.index))}")
    print(f"Labels without {config.INPUT_TIMEPOINT} recordings: {sorted(set(labels.index) - set(recordings['subject']))}")

    windows, rows, features, skipped = [], [], [], 0
    for rec in recordings[recordings["subject"].isin(labels.index)].itertuples():
        table = pd.read_csv(config.EXTRACTED_DIR / rec.file)
        walk, channels = walking_part(table, rec.frame_rate), channel_columns(table)
        starts, length = window_starts(len(walk), rec.frame_rate)
        if not starts:
            print(f"{rec.file}: walking part shorter than one window, not used")
        for start in starts:
            window = walk.iloc[start:start + length]
            values, feats = window[channels].to_numpy().T, window_features(window, channels, rec.frame_rate)
            if np.isnan(values).any() or np.isnan(list(feats.values())).any():
                skipped += 1  # missing frames: the window is not used
                continue
            windows.append(values)
            features.append(feats)
            rows.append({"subject": rec.subject, "file": rec.file, "speed": rec.speed,
                         "start_s": round(window["time_s"].iloc[0], 2), **labels.loc[rec.subject].to_dict()})
    if not windows:
        sys.exit("No windows: check the recordings in results/inventory.csv and WINDOW_SECONDS in config.py.")

    config.WINDOWS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.WINDOWS_DIR / "windows.npy", np.stack(windows).astype(np.float32))
    pd.DataFrame(rows).to_csv(config.WINDOWS_DIR / "windows.csv", index=False)
    pd.DataFrame(features).to_csv(config.WINDOWS_DIR / "features.csv", index=False)
    per_subject = pd.DataFrame(rows).groupby("subject").agg(recordings=("file", "nunique"), windows=("file", "size"),
                                                            label=("label", "first"))
    print(per_subject.to_string())
    print(f"{len(windows)} windows of {len(channels)} channels x {length} frames, {len(features[0])} features each "
          f"({skipped} skipped for missing values) -> {config.WINDOWS_DIR}")


if __name__ == "__main__":
    main()
