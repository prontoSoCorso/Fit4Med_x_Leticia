"""Step 2: inventory of the recordings: what we have, what is missing, what looks wrong.

Reads:  data/extracted/recordings.csv and the per-recording tables
Writes: results/inventory.csv  (one row per recording with quality numbers and status ok/duplicate,
        plus one row per expected timepoint x speed of config.TEST with no recording: status missing)
"""
import sys

import numpy as np
import pandas as pd

import config
from pipeline.preprocessing import recording_stats


def missing_rows(inventory):
    """Expected timepoint x speed combinations of config.TEST that have no recording."""
    rows = []
    for subject, recs in inventory[inventory["test"] == config.TEST].groupby("subject"):
        for timepoint in config.TIMEPOINTS:
            speeds = set(recs.loc[recs["timepoint"] == timepoint, "speed"])
            if not speeds:
                rows.append({"subject": subject, "timepoint": timepoint, "speed": "any"})
            elif "unknown" not in speeds:  # speeds can only be checked once they are known
                rows += [{"subject": subject, "timepoint": timepoint, "speed": s}
                         for s in config.SPEEDS if s not in speeds]
    return pd.DataFrame(rows, columns=["subject", "timepoint", "speed"]).assign(test=config.TEST, status="missing")


def main():
    index_path = config.EXTRACTED_DIR / "recordings.csv"
    if not index_path.exists():
        sys.exit(f"{index_path} not found: run 01_extract.py first (or 00_make_fake_cohort.py for the demo).")
    recordings = pd.read_csv(index_path)
    stats = pd.DataFrame([recording_stats(pd.read_csv(config.EXTRACTED_DIR / f)) for f in recordings["file"]])
    inventory = pd.concat([recordings, stats], axis=1).sort_values(["subject", "rec_date"])

    # The same recording exported twice (same start time), or two files claiming the same take: keep the first one
    same_start = inventory.duplicated(["subject", "rec_date"]) & inventory["rec_date"].notna()
    copies = same_start | inventory.duplicated(["subject", "test", "timepoint", "take"])
    inventory["status"] = np.where(copies, "duplicate", "ok")
    ok = inventory["status"] == "ok"
    inventory.loc[ok, "repetition"] = inventory[ok].groupby(["subject", "test", "timepoint", "speed"]).cumcount() + 1

    inventory = pd.concat([inventory, missing_rows(inventory)], ignore_index=True).sort_values("subject", kind="stable")
    counts = ["take", "repetition", "n_frames", "n_nan", "n_repeated_times"]
    inventory[counts] = inventory[counts].astype("Int64")  # whole numbers stay whole next to empty cells
    config.RESULTS_DIR.mkdir(exist_ok=True)
    inventory.to_csv(config.RESULTS_DIR / "inventory.csv", index=False)
    print(inventory["status"].value_counts().to_string())
    print(f"Inventory of {len(recordings)} recordings -> {config.RESULTS_DIR / 'inventory.csv'}")


if __name__ == "__main__":
    main()
