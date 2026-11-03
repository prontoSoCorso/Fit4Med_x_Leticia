"""Step 6: one table and one figure that compare all the evaluated models.

Reads:  results/predictions/<model>.csv for every model in MODELS_ENABLED (written by 05_evaluate.py)
Writes: results/summary_table.csv  (one row per model: MAE, RMSE, Spearman, R² with 95% intervals)
        results/comparison.png     (MAE of each model with its 95% interval; lower is better)
"""
import sys

import matplotlib

matplotlib.use("Agg")  # draw straight into a file, no window needed
import matplotlib.pyplot as plt
import pandas as pd

import config
from pipeline.metrics import scores_with_ci

# Reference palette, light theme: one colour for the models, greys for text and lines
SURFACE, INK, MUTED, GRID, AXIS, BLUE = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#2a78d6"


def summary_table():
    """Scores of every enabled model that has predictions, best (lowest MAE) first."""
    rows, missing = [], []
    for name in config.MODELS_ENABLED:
        path = config.RESULTS_DIR / "predictions" / f"{name}.csv"
        if not path.exists():
            missing.append(name)
            continue
        predictions = pd.read_csv(path)
        fake = predictions["subject"].astype(str).str.startswith("FAKE").all()
        rows.append({"model": name, "patients": len(predictions), **scores_with_ci(predictions),
                     "data": "FAKE" if fake else "real"})
    if missing:
        print(f"Not evaluated yet (run 05_evaluate.py {' '.join(missing)}): {missing}")
    if not rows:
        sys.exit("No predictions found: run 05_evaluate.py first.")
    return pd.DataFrame(rows).sort_values("MAE").round(3)


def comparison_figure(table, path):
    """Dot = MAE of a model, line = its 95% interval over patients; best model at the top."""
    table = table.iloc[::-1]
    rows = range(len(table))
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(table) + 1.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.hlines(rows, table["MAE_low"], table["MAE_high"], color=BLUE, linewidth=2)
    ax.plot(table["MAE"], rows, "o", color=BLUE, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
    if "mean_baseline" in set(table["model"]):
        baseline = table.loc[table["model"] == "mean_baseline", "MAE"].iloc[0]
        ax.axvline(baseline, color=MUTED, linewidth=1.5, zorder=0)
        ax.annotate("mean-predictor baseline", (baseline, 1), xycoords=("data", "axes fraction"),
                    xytext=(-4, -2), textcoords="offset points", ha="right", va="top", color=MUTED, fontsize=9)
    ax.set_yticks(list(rows), table["model"])
    ax.set_xlim(left=0)
    ax.set_xlabel("MAE in mBI points (lower is better); line = 95% interval over patients", color=INK)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK, length=0)
    title = f"Leave-one-patient-out error of each model ({table['patients'].max()} patients)"
    if (table["data"] == "FAKE").all():
        title += "\nFAKE DATA: demo of the pipeline, no scientific value"
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    table = summary_table()
    table.to_csv(config.RESULTS_DIR / "summary_table.csv", index=False)
    comparison_figure(table, config.RESULTS_DIR / "comparison.png")
    for row in table.itertuples():
        print(f"{row.model:15s} MAE {row.MAE:5.1f} [{row.MAE_low:4.1f}-{row.MAE_high:4.1f}]  "
              f"R2 {row.R2:5.2f}  Spearman {row.Spearman:5.2f}")
    print(f"-> {config.RESULTS_DIR / 'summary_table.csv'} and {config.RESULTS_DIR / 'comparison.png'}")


if __name__ == "__main__":
    main()
