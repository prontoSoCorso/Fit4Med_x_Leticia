"""Step 0 (demo only): build a FAKE cohort of 20 patients from the one real example recording.

Reads:  examples/S2_10MWT_T0-004.mvnx (config.EXAMPLE_MVNX)
Writes: data/extracted/FAKE*.csv + recordings.csv (same format as 01_extract.py) and data/labels.csv
Each fake patient is the example walked slower or faster (time-stretch) plus noise, and its
mBI is made from that speed: mBI = 100 x speed (m/s) + noise. FAKE: no scientific value.
"""
import shutil
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import config
from pipeline.mvnx import mvnx_to_csv
from pipeline.preprocessing import walked_distance

N_PATIENTS = 20
NO_MAX_SPEED = {5, 12}        # these patients have no max-speed recording
TWO_REPETITIONS = {2, 7, 15}  # these patients repeated each test twice
HAS_T1 = {3, 9, 18}           # these patients also have recordings at t1 (end of treatment)
NO_BMI = {4, 16}              # missing BMI, to exercise imputation
DUPLICATED = 10               # this patient's first recording was exported twice
NO_LABEL = 20                 # this patient has recordings but no mBI
MAX_SPEED_FACTOR = 1.3        # max speed = 1.3 x normal speed
T1_IMPROVEMENT = 1.2          # at t1 patients walk 20% faster
START = datetime(2025, 1, 6, 9, 0)


def time_stretch(base, stretch, rng):
    """The same movement `stretch` times slower (> 1) or faster (< 1), plus a little noise."""
    time = base["time_s"].to_numpy()
    new_time = np.arange(0, time[-1] * stretch, 1 / 60)  # 60 frames per second, like the real data
    signals = base.drop(columns="time_s")
    values = np.column_stack([np.interp(new_time / stretch, time, signals[c]) for c in signals])
    # Noise: 10% of each signal's frame-to-frame change (small, so the walked distance stays right)
    values += rng.normal(0, 0.1, values.shape) * np.diff(signals.to_numpy(), axis=0).std(axis=0)
    table = pd.DataFrame(values, columns=signals.columns)
    contacts = [c for c in table if c.startswith("footContacts")]
    table[contacts] = (table[contacts] > 0.5).astype(int)  # foot contacts stay 0 or 1
    table.insert(0, "time_s", np.round(new_time, 3))
    return table


def walking_speeds(patient, normal_speed):
    """Normal walking speed (m/s) at each timepoint of one patient."""
    speeds = {"t0": normal_speed}
    if patient in HAS_T1:
        speeds["t1"] = normal_speed * T1_IMPROVEMENT
    return speeds


def planned_recordings(patient, normal_speed):
    """(timepoint, speed label, take, walking speed) of every recording of one fake patient."""
    plan = []
    for timepoint, speed in walking_speeds(patient, normal_speed).items():
        takes = [(1, 3), (2, 4)] if patient in TWO_REPETITIONS else [(1, 3)]  # (normal take, max take)
        for normal_take, max_take in takes:
            plan.append((timepoint, "normal", normal_take, speed))
            if patient not in NO_MAX_SPEED:
                plan.append((timepoint, "max", max_take, speed * MAX_SPEED_FACTOR))
    return plan


def label_rows(patient, subject, normal_speed, rng):
    """mBI at each timepoint made from the walking speed, plus random metadata. FAKE."""
    if patient == NO_LABEL:
        return []
    metadata = {"age": int(rng.integers(45, 86)), "hemisphere": rng.choice(["left", "right"]),
                "bmi": np.nan if patient in NO_BMI else round(rng.normal(26, 4), 1)}
    return [{"subject": subject, "timepoint": timepoint,
             "mbi": int(np.clip(round(100 * speed + rng.normal(0, 5)), 0, 100)), **metadata}
            for timepoint, speed in walking_speeds(patient, normal_speed).items()]


def check_no_real_data():
    real_tables = [f for f in config.EXTRACTED_DIR.glob("*.csv") if not f.name.startswith(("FAKE", "recordings"))]
    if real_tables:
        sys.exit(f"{config.EXTRACTED_DIR} holds real recordings: move that folder away before building the FAKE demo.")
    if config.LABELS_CSV.exists() and not pd.read_csv(config.LABELS_CSV)["subject"].str.startswith("FAKE").all():
        sys.exit(f"{config.LABELS_CSV} holds real labels: move it away before building the FAKE demo.")
    if not config.EXAMPLE_MVNX.exists():
        sys.exit(f"{config.EXAMPLE_MVNX} not found: the demo needs one real .mvnx (see EXAMPLE_MVNX in config.py).")


def main():
    check_no_real_data()
    config.EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        mvnx_to_csv(config.EXAMPLE_MVNX, Path(tmp) / "example.csv")
        base = pd.read_csv(Path(tmp) / "example.csv")
    base_speed = walked_distance(base) / base["time_s"].iloc[-1]  # mean speed of the example, m/s
    rng = np.random.default_rng(config.SEED)

    recordings, labels = [], []
    for patient in range(1, N_PATIENTS + 1):
        subject, normal_speed = f"FAKE{patient:02d}", rng.uniform(0.3, 0.9)
        for timepoint, speed_label, take, speed in planned_recordings(patient, normal_speed):
            name = f"{subject}_{config.TEST}_{timepoint.upper()}-{take:03d}.csv"
            time_stretch(base, base_speed / speed, rng).to_csv(config.EXTRACTED_DIR / name, index=False, float_format="%.6g")
            date = START + timedelta(days=patient + (60 if timepoint == "t1" else 0), minutes=10 * take)
            recordings.append({"file": name, "source_file": "FAKE (generated)", "subject": subject,
                               "test": config.TEST, "timepoint": timepoint, "take": take, "speed": speed_label,
                               "rec_date": date.isoformat(timespec="milliseconds"), "frame_rate": 60.0})
        labels += label_rows(patient, subject, normal_speed, rng)

    first = next(r for r in recordings if r["subject"] == f"FAKE{DUPLICATED:02d}")  # the same file exported twice
    shutil.copy(config.EXTRACTED_DIR / first["file"], config.EXTRACTED_DIR / first["file"].replace(".csv", "_copy.csv"))
    recordings.append({**first, "file": first["file"].replace(".csv", "_copy.csv")})

    pd.DataFrame(recordings).to_csv(config.EXTRACTED_DIR / "recordings.csv", index=False)
    pd.DataFrame(labels).to_csv(config.LABELS_CSV, index=False)
    print(f"FAKE cohort: {N_PATIENTS} patients, {len(recordings)} recordings -> {config.EXTRACTED_DIR}, "
          f"labels -> {config.LABELS_CSV}")


if __name__ == "__main__":
    main()
