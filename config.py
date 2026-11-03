"""All parameters and paths of the pipeline, in one place.

Every numbered script reads its settings from here: change values here, not in the code.
"""
from pathlib import Path

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                # put the .mvnx files here
EXTRACTED_DIR = DATA_DIR / "extracted"    # one CSV per recording (written by 01_extract.py)
WINDOWS_DIR = DATA_DIR / "windows"        # windows and features (written by 03_make_windows.py)
EMBEDDINGS_DIR = DATA_DIR / "embeddings"  # foundation-model embeddings (written by 04_embed_foundation.py)
LABELS_CSV = DATA_DIR / "labels.csv"      # columns: subject, timepoint, mbi (+ optional age, bmi, hemisphere)
RESULTS_DIR = ROOT / "results"
EXAMPLE_MVNX = ROOT / "examples" / "S2_10MWT_T0-004.mvnx"  # one real recording, used by the demo and tests

# ---------- Study design ----------
TEST = "10MWT"                                             # walking test analysed (10-metre walk test)
TIMEPOINTS = ["t0", "t5s", "t10s", "t15s", "t20s", "t1"]   # IMU every 5 sessions; t1 = end of treatment
SPEEDS = ["normal", "max"]                                 # walking-speed configurations
# The .mvnx files do not say which take was at normal or at max speed. Fill this map when the
# protocol is known, e.g. {1: "normal", 2: "normal", 3: "max", 4: "max"}, then run 01_extract.py again.
SPEED_BY_TAKE = {}
