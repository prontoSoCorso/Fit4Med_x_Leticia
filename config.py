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

SEED = 42  # fixed random numbers: every run gives the same results

# ---------- What to predict ----------
INPUT_TIMEPOINT = "t0"                       # IMU recordings used as input
TARGET_TIMEPOINT = "t0"                      # timepoint of the label to predict (later, e.g., "t1")
TARGET = "mbi"                               # label column in labels.csv
SPEEDS_USED = ["normal", "max", "unknown"]   # all recordings of a patient pooled together
METADATA = ["age", "bmi", "hemisphere"]      # optional columns of labels.csv, used only if present

# ---------- Cleaning and windows ----------
STILL_SPEED_THRESHOLD = 0.15  # m/s: a slower pelvis means standing still; still parts at start and end are cut
SMOOTHING_SECONDS = 0.5       # the pelvis speed is averaged over this time before applying the threshold
WINDOW_SECONDS = 4.0          # length of one window (about 2-3 strides)
WINDOW_OVERLAP = 0.5          # consecutive windows share half of their frames
# Signals given to the models: each name selects the columns that start with it (x, y, z axes).
# Joint angles avoid the slide-4 risks (no drift, no heading/yaw, no gravity); the pelvis vertical
# acceleration is gravity-free and does not depend on the walking direction.
CHANNELS = ["jointAngle_jRightHip", "jointAngle_jRightKnee", "jointAngle_jRightAnkle",
            "jointAngle_jLeftHip", "jointAngle_jLeftKnee", "jointAngle_jLeftAnkle", "acceleration_Pelvis_z"]

# ---------- Models and evaluation ----------
# Files in models/ that 05_evaluate.py runs. Add a model = add its file and its name here.
MODELS_ENABLED = ["mean_baseline", "metadata_only", "svr", "knn", "random_forest", "minirocket",
                  "cnn1d", "lstm"]
N_BOOTSTRAP = 2000  # resamplings of the patients for the 95% confidence intervals
# Neural networks: fixed settings (too few patients to tune them); training stops early when the
# validation patients stop improving for `patience` epochs (one epoch = one pass over the training windows)
DEEP = {"max_epochs": 100, "patience": 10, "batch_size": 32, "learning_rate": 1e-3}
