"""
期刊 MLE 推断全链路脚本（通用版）
==================================
步骤：
  1. 构建词频分布 (distribution) → 调用 src/estimation.py
  2. 逐月 MLE 推断 (MLE.py) → α 估计 + 95% CI + Bootstrap
  3. 输出趋势图 + CSV 结果表

用法：
  python run_journal_pipeline.py --journal_name elife
  python run_journal_pipeline.py --journal_name plos_one --project_dir /mnt/data/hermes-workspace/mle-llm-detection

前置条件：
  - preprocess_journal.py 已生成 processed_data/{journal}/inference_data/ 和 human_corpus/
  - generate_ai_journal.py 已生成 processed_data/{journal}/ai_corpus/
"""

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="期刊 MLE 推断全链路")
    parser.add_argument("--journal_name", required=True,
                        help="期刊名称，如 elife")
    parser.add_argument("--project_dir", default=None,
                        help="项目根目录（默认自动检测）")
    parser.add_argument("--no_plot", action="store_true",
                        help="跳过绘图")
    return parser.parse_args()


# ============================================================
# 主流程
# ============================================================
def main():
    args = parse_args()
    journal = args.journal_name

    # 项目目录
    if args.project_dir:
        project_dir = Path(args.project_dir)
    else:
        project_dir = Path(__file__).resolve().parent.parent

    # 将 src/ 加入导入路径
    sys.path.insert(0, str(project_dir / "scripts"))
    from src.MLE import MLE
    from src.estimation import (
        count_human_binary_word_occurrences,
        count_ai_binary_word_occurrences,
        estimate_log_probabilities,
        get_vocabulary_intersection,
        filter_frequent_words,
        calculate_log_probability,
    )

    data_dir = project_dir / "processed_data" / journal
    h_path = data_dir / "human_corpus" / f"{journal}_human.parquet"
    q_path = data_dir / "ai_corpus" / f"{journal}_ai.parquet"
    dist_dir = data_dir / "distribution"
    dist_path = dist_dir / f"{journal}.parquet"
    inference_dir = data_dir / "inference_data"
    dist_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================
    # 步骤 1: 构建分布（复用 build_distribution.py 逻辑）
    # ========================================================
    print("=" * 60)
    print(f"  {journal} — 步骤 1/3: 构建词频分布")
    print("=" * 60)

    # ---- 加载 H 语料 ----
    print(f"\n加载 H 语料: {h_path}")
    df_h = pd.read_parquet(h_path)
    print(f"  H 语料: {len(df_h)} 条")

    # ---- 加载 Q 语料 ----
    print(f"加载 Q 语料: {q_path}")
    df_q = pd.read_parquet(q_path)
    print(f"  Q 语料: {len(df_q)} 条")

    # 对齐
    n = min(len(df_h), len(df_q))
    print(f"对齐 H 和 Q: 各取 {n} 条")

    # 展开为 flat table（一行一句）
    def flatten_sentences(series):
        """将 numpy object array → Python list of lists → flat DataFrame"""
        result = []
        for item in series:
            sentences = [list(s) for s in item]
            sentences = [s for s in sentences if len(s) > 1]
            result.extend(sentences)
        return pd.DataFrame({"sentence": result}).dropna()

    h_flat = flatten_sentences(df_h["human_sentence"].iloc[:n])
    q_flat = flatten_sentences(df_q["ai_sentence"].iloc[:n])
    print(f"  H: {len(h_flat)} 句, Q: {len(q_flat)} 句")

    # 构建分布
    print("计算词频分布...")
    human_word_counts = count_human_binary_word_occurrences(h_flat)
    ai_word_counts = count_ai_binary_word_occurrences(q_flat)
    total_human = len(h_flat)
    total_ai = len(q_flat)

    human_log_probs = estimate_log_probabilities(human_word_counts, total_human)
    ai_log_probs = estimate_log_probabilities(ai_word_counts, total_ai)

    common_vocab = get_vocabulary_intersection(human_word_counts, ai_word_counts)
    freq_human = filter_frequent_words(human_word_counts, 5)
    freq_ai = filter_frequent_words(ai_word_counts, 3)
    freq_common = common_vocab.intersection(freq_human.keys(), freq_ai.keys())

    dist_df = calculate_log_probability(human_log_probs, ai_log_probs, freq_common)
    dist_df.to_parquet(dist_path, index=False)
    print(f"  词典大小: {len(dist_df)} 个词")
    print(f"  分布保存至: {dist_path}")

    # ========================================================
    # 步骤 2: MLE 推断
    # ========================================================
    print(f"\n{'=' * 60}")
    print(f"  {journal} — 步骤 2/3: 逐月 MLE 推断")
    print("=" * 60)

    model = MLE(str(dist_path))
    results = []

    files = sorted(os.listdir(inference_dir))
    for fname in files:
        if not fname.endswith(".parquet"):
            continue
        path = os.path.join(inference_dir, fname)
        year_month = fname.replace(".parquet", "")

        try:
            alpha, ci = model.inference(path)
            year, month = year_month.split("_")
            results.append({
                "year": int(year),
                "month": int(month),
                "time": int(year) + (int(month) - 1) / 12,
                "alpha": alpha * 100,
                "ci": ci * 100,
                "n_sentences": len(pd.read_parquet(path)),
            })
            print(f"  {year_month}: α={alpha*100:.1f}% ± {ci*100:.2f}%  "
                  f"({len(pd.read_parquet(path))} 条)")
        except Exception as e:
            print(f"  {year_month}: 失败 - {e}")

    if not results:
        print("❌ 没有可推断的月份，退出。")
        return

    # ========================================================
    # 步骤 3: 绘图 + 保存
    # ========================================================
    print(f"\n{'=' * 60}")
    print(f"  {journal} — 步骤 3/3: 趋势图 + 结果表")
    print("=" * 60)

    df_r = pd.DataFrame(results).sort_values("time")

    # 保存 CSV
    csv_path = data_dir / f"{journal}_alpha_results.csv"
    df_r.to_csv(csv_path, index=False)
    print(f"结果表: {csv_path}")

    # 全量均值
    mean_alpha_all = df_r["alpha"].mean()
    print(f"\n全量 α 均值: {mean_alpha_all:.2f}%")

    # 年度均值
    yearly = df_r.groupby("year")["alpha"].mean()
    print("\n年度 α 均值:")
    for yr, a in yearly.items():
        n_rows = len(df_r[df_r["year"] == yr])
        print(f"  {yr}: {a:.2f}% ({n_rows} 个月)")

    if args.no_plot:
        print("跳过绘图 (--no_plot)")
        return

    # ---- 绘图 ----
    plt.rcParams.update({"font.size": 14})
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti SC", "PingFang SC"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(12, 6))

    chatgpt_time = 2022 + 10 / 12
    ax.axvline(x=chatgpt_time, color="darkred", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(
        chatgpt_time - 0.05,
        df_r["alpha"].max() * 0.85,
        "ChatGPT\n2022.11",
        color="darkred",
        ha="right",
        va="top",
        fontsize=12,
    )

    ax.errorbar(
        df_r["time"], df_r["alpha"], yerr=df_r["ci"],
        fmt="o-", color="#2b83ba", markersize=6, capsize=4,
        elinewidth=1, linewidth=1.5,
        label=f"{journal} (n={len(df_r)} months)",
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Estimated α (%)")
    ax.set_title(
        f"{journal} Peer Review LLM Usage Trend\n"
        f"(MLE inference, {len(dist_df)}-word distribution)"
    )

    def to_percent(y, pos):
        return f"{y:.0f}%"
    ax.yaxis.set_major_formatter(FuncFormatter(to_percent))

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left")
    sns.despine(right=True, top=True)
    plt.tight_layout()

    png_path = data_dir / f"{journal}_trend.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"趋势图: {png_path}")
    plt.close()

    print(f"\n✅ {journal} 全链路完成。")
    print(f"  分布: {dist_path}")
    print(f"  结果: {csv_path}")
    print(f"  图表: {png_path}")


if __name__ == "__main__":
    main()
