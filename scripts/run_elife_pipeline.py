"""Run the eLife MLE inference pipeline using repository-relative paths."""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).parent))
from src.MLE import MLE

PROJECT_DIR = Path(
    os.environ.get("PROJECT_DIR", Path(__file__).resolve().parents[1])
).expanduser().resolve()
ELIFE_DIR = PROJECT_DIR / "processed_data" / "elife"
DIST_PATH = Path(os.environ.get("ELIFE_DIST_PATH", ELIFE_DIR / "distribution/elife.parquet"))
INFERENCE_DIR = Path(os.environ.get("ELIFE_INFERENCE_DIR", ELIFE_DIR / "inference_data"))
OUTPUT_DIR = Path(os.environ.get("ELIFE_OUTPUT_DIR", ELIFE_DIR))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print(f"Distribution: {DIST_PATH}")
    model = MLE(str(DIST_PATH))
    results = []

    for parquet_file in sorted(INFERENCE_DIR.glob("*.parquet")):
        year_month = parquet_file.stem
        try:
            alpha, ci = model.inference(str(parquet_file))
            year, month = year_month.split("_")
            results.append(
                {
                    "year": int(year),
                    "month": int(month),
                    "time": int(year) + (int(month) - 1) / 12,
                    "alpha": alpha * 100,
                    "ci": ci * 100,
                    "n_sentences": len(pd.read_parquet(parquet_file)),
                }
            )
        except Exception as exc:
            print(f"{year_month}: failed - {exc}")

    if not results:
        raise RuntimeError("No monthly inference results were produced.")

    df_r = pd.DataFrame(results).sort_values("time")
    fig, ax = plt.subplots(figsize=(12, 6))
    chatgpt_time = 2022 + 10 / 12
    ax.axvline(chatgpt_time, linestyle="--", linewidth=1.5, alpha=0.7)
    ax.errorbar(
        df_r["time"],
        df_r["alpha"],
        yerr=df_r["ci"],
        fmt="o-",
        markersize=6,
        capsize=4,
        elinewidth=1,
        linewidth=1.5,
        label="eLife",
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Estimated alpha (%)")
    ax.set_title("eLife Peer Review — Exploratory MLE Estimate")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left")
    sns.despine(right=True, top=True)
    plt.tight_layout()

    plot_path = OUTPUT_DIR / "elife_pipeline_test.png"
    csv_path = OUTPUT_DIR / "elife_alpha_results.csv"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    df_r.to_csv(csv_path, index=False)

    print(f"Saved plot: {plot_path}")
    print(f"Saved results: {csv_path}")


if __name__ == "__main__":
    main()
