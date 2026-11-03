"""Step 1: turn every Xsens .mvnx recording into a table.

Reads:  data/raw/*.mvnx
Writes: data/extracted/<file>.csv  (time_s + all 788 signals, one row per frame)
        data/extracted/recordings.csv  (one row per recording: subject, test, timepoint, take, speed, date)
"""
import csv
import sys

import config
from pipeline.mvnx import mvnx_to_csv, recording_info

INDEX_COLUMNS = ["file", "source_file", "subject", "test", "timepoint", "take", "speed", "rec_date", "frame_rate"]


def main():
    mvnx_files = sorted(config.RAW_DIR.glob("*.mvnx"))
    if not mvnx_files:
        sys.exit(f"No .mvnx files in {config.RAW_DIR}: copy the recordings there "
                 "(for the demo without real data, run 00_make_fake_cohort.py instead).")
    config.EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    if any(config.EXTRACTED_DIR.glob("FAKE*.csv")):
        sys.exit(f"{config.EXTRACTED_DIR} still holds the FAKE demo cohort: delete that folder, then run again.")

    recordings = []
    for path in mvnx_files:
        csv_name = path.stem + ".csv"  # named after the source file, so duplicates never overwrite each other
        header, n_frames = mvnx_to_csv(path, config.EXTRACTED_DIR / csv_name)
        info = recording_info(header, path)
        speed = config.SPEED_BY_TAKE.get(info["take"], "unknown")
        recordings.append({**info, "file": csv_name, "source_file": path.name, "speed": speed})
        print(f"{path.name}: {n_frames} frames, {info['subject']} {info['test']} {info['timepoint']} "
              f"take {info['take']} ({speed} speed)")

    with open(config.EXTRACTED_DIR / "recordings.csv", "w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(recordings)
    print(f"{len(recordings)} recordings -> {config.EXTRACTED_DIR}")


if __name__ == "__main__":
    main()
