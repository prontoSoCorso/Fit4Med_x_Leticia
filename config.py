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
