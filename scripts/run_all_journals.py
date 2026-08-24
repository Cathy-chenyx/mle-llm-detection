"""
六刊 MLE Pipeline 主入口
=========================
用法：
  DEEPSEEK_API_KEY=sk-... python run_all_journals.py

执行步骤（每刊串行）：
  1. 从主库 overlay 导出 H 语料 CSV (pre-ChatGPT, top 40 by content_chars)
  2. 全量推断数据导出 CSV
  3. preprocess_journal.py — spaCy 分句分词
  4. preprocess_inference.py — 推断数据按月 parquet
  5. generate_ai_journal.py — DeepSeek API 两阶段生成 Q 语料
  6. run_journal_pipeline.py — 分布构建 + MLE 推断
  7. 跨刊汇总

输出：
  processed_data/{journal_name}/human_corpus/{journal_name}_human.parquet
  processed_data/{journal_name}/ai_corpus/{journal_name}_ai.parquet
  processed_data/{journal_name}/distribution/{journal_name}.parquet
  processed_data/{journal_name}/{journal_name}_alpha_results.csv
  processed_data/{journal_name}/{journal_name}_pipeline.png
  processed_data/cross_journal_summary.csv
  processed_data/cross_journal_trend.png
  processed_data/cross_journal_bar.png
  processed_data/full_report.md
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

import os
import re
import csv
import sqlite3
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from pathlib import Path
import spacy
from src.estimation import (
    count_human_binary_word_occurrences,
    count_ai_binary_word_occurrences,
    estimate_log_probabilities,
    get_vocabulary_intersection,
    filter_frequent_words,
    calculate_log_probability,
)
from src.MLE import MLE


# ============================================================
# 配置
# ============================================================
PROJECT_DIR = Path("/mnt/data/hermes-workspace/mle-llm-detection")
OVERLAY_PATH = "/mnt/data/hermes-workspace/public_review_mining_stack/store/exports/data_repairs/unit-aligned-cross-journal-2020-2024-20260714-v3/research_review_units_overlay.sqlite"

# 六刊配置：journal_name (目录名) → DB journal name
# 注意：peerj_cs 已按会议决定从分析中剔除（2026-08 会议），JOURNALS 仅保留 5 刊
JOURNALS = [
    {"dir": "bmc_med", "db_journal": "BMC Med", "n": 40},
    {"dir": "peerj", "db_journal": "PeerJ", "n": 40},
    {"dir": "f1000research", "db_journal": "F1000Research", "n": 40},
    {"dir": "elife", "db_journal": "eLife", "n": 40},
    {"dir": "nat_commun", "db_journal": "Nat Commun", "n": 40},
]

CHATGPT_CUTOFF = "2022-11-01"  # pre-ChatGPT: review_date or publication_date < this


# ============================================================
# 文本清洗（复制自 preprocess_elife.py）
# ============================================================
def clean_html(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&nbsp;', ' ', text)
    return text

def clean_editor_notes(text):
    text = re.sub(r'\[Editors.? note:.*?\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[#.*?\]', '', text)
    return text

def clean_section_headers(text):
    headers = [
        r'^Acceptance summary:\s*',
        r'^Decision letter after peer review:\s*',
        r'^Summary:\s*',
        r'^Peer Review File\s*',
    ]
    for h in headers:
        text = re.sub(h, '', text, flags=re.IGNORECASE | re.MULTILINE)
    return text

def clean_text(text):
    text = str(text)
    text = clean_html(text)
    text = clean_editor_notes(text)
    text = clean_section_headers(text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = text.strip()
    return text


# ============================================================
# spaCy 分词
# ============================================================
def make_tokenize(nlp):
    def tokenize(text):
        text = text.replace('\n', ' ')
        sentence_list = []
        doc = nlp(text)
        for sent in doc.sents:
            words = re.findall(r'\b\w+\b', sent.text.lower())
            words = [w for w in words if not w.isdigit()]
            if len(words) > 0:
                sentence_list.append(words)
        return sentence_list
    return tokenize


# ============================================================
# Step 1: 导出 H 语料 CSV (pre-ChatGPT, top 40 by content_chars)
# ============================================================
def export_human_csv(journal_cfg, nlp):
    dir_name = journal_cfg["dir"]
    db_journal = journal_cfg["db_journal"]
    n = journal_cfg["n"]

    print(f"\n{'='*60}")
    print(f"Step 1: 导出 H 语料 — {dir_name}")
    print(f"{'='*60}")

    conn = sqlite3.connect(f"file:{OVERLAY_PATH}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    cur = conn.cursor()

    # 查询 pre-ChatGPT 的 review units
    # 统一使用 publication_date < cutoff（会议决定），并限定时间范围 2021-2026
    cur.execute("""
        SELECT unit_id, content, publication_date, review_date
        FROM derived_review_units_v1
        WHERE journal = ?
          AND content IS NOT NULL
          AND LENGTH(content) > 50
          AND publication_date >= '2021-01-01'
          AND publication_date < ?
        ORDER BY LENGTH(content) DESC
        LIMIT ?
    """, (db_journal, CHATGPT_CUTOFF, n))

    rows = cur.fetchall()
    conn.close()

    if not rows:
        print(f"  错误: {db_journal} 没有找到 H 语料")
        return 0

    # 保存 CSV
    export_dir = PROJECT_DIR / "review_text_export"
    export_dir.mkdir(parents=True, exist_ok=True)
    csv_path = export_dir / f"{dir_name}_human_40.csv"

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['unit_id', 'review_text', 'publication_date', 'review_date'])
        for row in rows:
            writer.writerow([row[0], row[1], row[2], row[3]])

    print(f"  导出 {len(rows)} 条 → {csv_path}")
    print(f"  文件大小: {os.path.getsize(csv_path) / 1024:.1f} KB")

    # 预处理：清洗 + spaCy 分词
    df = pd.read_csv(csv_path)
    df['cleaned_text'] = df['review_text'].apply(clean_text)
    df = df[df['cleaned_text'].str.len() > 50].copy()

    if len(df) == 0:
        print(f"  错误: 清洗后无有效条目")
        return 0

    tokenize = make_tokenize(nlp)
    df['human_sentence'] = df['cleaned_text'].apply(tokenize)

    # 保存 parquet
    human_dir = PROJECT_DIR / "processed_data" / dir_name / "human_corpus"
    human_dir.mkdir(parents=True, exist_ok=True)
    human_path = human_dir / f"{dir_name}_human.parquet"
    df[['human_sentence']].to_parquet(human_path, index=False)

    total_sentences = df['human_sentence'].apply(len).sum()
    total_tokens = df['human_sentence'].apply(lambda x: sum(len(s) for s in x)).sum()
    print(f"  H 语料: {len(df)} 条, {total_sentences} 句, {total_tokens} 词")
    print(f"  Parquet: {human_path}")

    return len(df)


# ============================================================
# Step 2: 导出全量推断数据 CSV + 按月 parquet
# ============================================================
def export_inference_data(journal_cfg, nlp):
    dir_name = journal_cfg["dir"]
    db_journal = journal_cfg["db_journal"]

    print(f"\n{'='*60}")
    print(f"Step 2: 导出全量推断数据 — {dir_name}")
    print(f"{'='*60}")

    conn = sqlite3.connect(f"file:{OVERLAY_PATH}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    cur = conn.cursor()

    cur.execute("""
        SELECT unit_id, content, publication_date, review_date
        FROM derived_review_units_v1
        WHERE journal = ?
          AND content IS NOT NULL
        ORDER BY publication_date
    """, (db_journal,))

    rows = cur.fetchall()
    conn.close()

    if not rows:
        print(f"  错误: {db_journal} 没有推断数据")
        return 0

    # 保存 CSV
    export_dir = PROJECT_DIR / "review_text_export"
    csv_path = export_dir / f"{dir_name}_all_inference.csv"

    csv.field_size_limit(sys.maxsize)

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['unit_id', 'review_text', 'publication_date', 'review_date'])
        for row in rows:
            writer.writerow([row[0], row[1], row[2], row[3]])

    print(f"  导出 {len(rows)} 条 → {csv_path}")

    # 预处理：清洗 + spaCy 分词 + 按月分组
    df = pd.read_csv(csv_path)

    # 确定月份分组列：统一使用 publication_date（发表日期），空值回退 review_date
    # 会议决定（2026-08）：六刊日期字段统一为发表日期，Nat Commun 无审稿日期，必须用 publication_date
    df['month_key'] = df.apply(
        lambda row: (str(row['publication_date'])[:7] if pd.notna(row.get('publication_date')) and str(row['publication_date']).strip() and str(row['publication_date']) != 'None'
                     else (str(row['review_date'])[:7] if pd.notna(row.get('review_date')) and str(row['review_date']).strip() and str(row['review_date']) != 'None'
                           else None)),
        axis=1
    )

    null_months = df['month_key'].isna().sum()
    if null_months > 0:
        print(f"  ⚠️  {null_months} 条记录无法确定月份，将跳过")
        df = df.dropna(subset=['month_key'])

    # 时间范围过滤：会议决定限定 2021-2026
    before_range = len(df)
    df['year_num'] = df['month_key'].str[:4].astype(int)
    df = df[(df['year_num'] >= 2021) & (df['year_num'] <= 2026)].copy()
    df = df.drop(columns=['year_num'])
    print(f"  ⏱️  时间范围限定 2021-2026: {before_range} → {len(df)} (移除 {before_range - len(df)} 条)")
    if len(df) == 0:
        print("  错误: 时间范围 2021-2026 内无数据")
        return 0

    df['cleaned_text'] = df['review_text'].apply(clean_text)
    df = df[df['cleaned_text'].str.len() > 50].copy()

    if len(df) == 0:
        print(f"  错误: 清洗后无有效条目")
        return 0

    tokenize = make_tokenize(nlp)
    df['inference_sentence'] = df['cleaned_text'].apply(tokenize)

    # 按月输出 parquet
    inference_dir = PROJECT_DIR / "processed_data" / dir_name / "inference_data"
    inference_dir.mkdir(parents=True, exist_ok=True)

    month_counts = df['month_key'].value_counts().sort_index()
    for month_key, count in month_counts.items():
        if count == 0:
            continue
        year, month = month_key.split('-')
        out_path = inference_dir / f"{year}_{month}.parquet"
        group = df[df['month_key'] == month_key]
        group[['inference_sentence']].to_parquet(out_path, index=False)

    total_sentences = df['inference_sentence'].apply(len).sum()
    total_tokens = df['inference_sentence'].apply(lambda x: sum(len(s) for s in x)).sum()
    print(f"  推断数据: {len(df)} 条, {total_sentences} 句, {total_tokens} 词")
    print(f"  月份数: {len(month_counts)}")
    print(f"  Parquet 目录: {inference_dir}")

    return len(df)


# ============================================================
# Step 3: 生成 AI 语料 (DeepSeek API)
# ============================================================
def generate_ai_corpus(journal_cfg):
    dir_name = journal_cfg["dir"]
    n = journal_cfg["n"]

    print(f"\n{'='*60}")
    print(f"Step 3: 生成 AI 语料 — {dir_name}")
    print(f"{'='*60}")

    # 检查是否已有 AI 语料
    ai_path = PROJECT_DIR / "processed_data" / dir_name / "ai_corpus" / f"{dir_name}_ai.parquet"
    if ai_path.exists():
        df_q = pd.read_parquet(ai_path)
        if len(df_q) >= n * 0.5:  # 至少 50% 成功率才跳过
            print(f"  AI 语料已存在: {len(df_q)} 条，跳过生成")
            return len(df_q)

    # 调用 generate_ai_journal.py
    import subprocess
    env = os.environ.copy()
    env["DEEPSEEK_API_KEY"] = os.environ.get("DEEPSEEK_API_KEY", "")
    env["DEEPSEEK_BASE_URL"] = "https://115.190.192.101"
    env["DEEPSEEK_MODEL"] = "deepseek-v4-flash"

    result = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "generate_ai_journal.py"),
         "--journal_name", dir_name],
        env=env,
        capture_output=True,
        text=True,
        timeout=600
    )

    print(result.stdout)
    if result.stderr:
        print(f"  STDERR: {result.stderr[:500]}")

    if ai_path.exists():
        df_q = pd.read_parquet(ai_path)
        return len(df_q)
    return 0


# ============================================================
# Step 4: 构建分布 + MLE 推断
# ============================================================
def run_pipeline(journal_cfg):
    dir_name = journal_cfg["dir"]

    print(f"\n{'='*60}")
    print(f"Step 4: 构建分布 + MLE 推断 — {dir_name}")
    print(f"{'='*60}")

    import subprocess
    env = os.environ.copy()

    # Skip if results already exist
    alpha_path = PROJECT_DIR / "processed_data" / dir_name / f"{dir_name}_alpha_results.csv"
    if alpha_path.exists():
        print(f"  ⏭️  已有结果，跳过: {alpha_path}")
        return True

    result = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "run_journal_pipeline.py"),
         "--journal_name", dir_name],
        env=env,
        capture_output=True,
        text=True,
        timeout=600
    )

    print(result.stdout)
    if result.stderr:
        print(f"  STDERR: {result.stderr[:500]}")

    return result.returncode == 0


# ============================================================
# Step 5: 跨刊汇总
# ============================================================
def cross_journal_summary():
    print(f"\n{'='*60}")
    print(f"Step 5: 跨刊汇总")
    print(f"{'='*60}")

    all_results = []
    for j in JOURNALS:
        dir_name = j["dir"]
        csv_path = PROJECT_DIR / "processed_data" / dir_name / f"{dir_name}_alpha_results.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df['journal'] = dir_name
            all_results.append(df)
            print(f"  {dir_name}: {len(df)} 个月, α 均值={df['alpha'].mean():.2f}%")
        else:
            print(f"  {dir_name}: 无结果")

    if not all_results:
        print("  错误: 没有任何结果")
        return

    df_all = pd.concat(all_results, ignore_index=True)

    # 1. 汇总 CSV
    summary_path = PROJECT_DIR / "processed_data" / "cross_journal_summary.csv"
    summary = df_all.groupby('journal').agg(
        n_months=('alpha', 'count'),
        mean_alpha=('alpha', 'mean'),
        latest_alpha=('alpha', 'last'),
        latest_month=('month', 'last'),
        total_sentences=('n_sentences', 'sum'),
    ).reset_index()
    summary.to_csv(summary_path, index=False)
    print(f"\n✅ 汇总 CSV: {summary_path}")
    print(summary.to_string(index=False))

    # 2. 趋势图
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']

    chatgpt_time = 2022 + 10/12
    ax.axvline(x=chatgpt_time, color='darkred', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(chatgpt_time + 0.02, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 25,
            "ChatGPT\n2022.11", color='darkred', ha='left', va='top', fontsize=11)

    for i, j in enumerate(JOURNALS):
        dir_name = j["dir"]
        df_j = df_all[df_all['journal'] == dir_name].sort_values(by='time')
        if len(df_j) == 0:
            continue
        ax.errorbar(df_j['time'], df_j['alpha'], yerr=df_j['ci'],
                    fmt='o-', color=colors[i % len(colors)], markersize=4, capsize=3,
                    elinewidth=0.8, linewidth=1.2, label=dir_name, alpha=0.85)

    ax.set_xlabel('Year', fontsize=13)
    ax.set_ylabel('Estimated α (%)', fontsize=13)
    ax.set_title('Cross-Journal LLM Usage Trend in Peer Review (MLE)', fontsize=14)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y:.0f}%"))
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='upper left', fontsize=10, ncol=2)
    sns.despine(right=True, top=True)
    plt.tight_layout()

    trend_path = PROJECT_DIR / "processed_data" / "cross_journal_trend.png"
    fig.savefig(str(trend_path), dpi=150, bbox_inches='tight')
    print(f"✅ 趋势图: {trend_path}")
    plt.close()

    # 3. 柱状图
    fig, ax = plt.subplots(figsize=(10, 6))
    journals_with_data = summary[summary['n_months'] > 0]
    bars = ax.bar(journals_with_data['journal'], journals_with_data['mean_alpha'],
                  color=colors[:len(journals_with_data)], alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Journal', fontsize=13)
    ax.set_ylabel('Mean α (%)', fontsize=13)
    ax.set_title('Cross-Journal Mean LLM Usage (MLE α)', fontsize=14)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y:.0f}%"))
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    for bar, val in zip(bars, journals_with_data['mean_alpha']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11)
    sns.despine(right=True, top=True)
    plt.tight_layout()

    bar_path = PROJECT_DIR / "processed_data" / "cross_journal_bar.png"
    fig.savefig(str(bar_path), dpi=150, bbox_inches='tight')
    print(f"✅ 柱状图: {bar_path}")
    plt.close()

    # 4. 报告
    report_path = PROJECT_DIR / "processed_data" / "full_report.md"
    with open(report_path, 'w') as f:
        f.write("# 六刊 MLE Pipeline 执行报告\n\n")
        f.write(f"**生成时间**: 2026-07-25\n")
        f.write(f"**API**: DeepSeek (deepseek-v4-flash)\n")
        f.write(f"**H 语料条数**: 每刊 40 条\n\n")

        f.write("## 六刊结果汇总\n\n")
        f.write("| 期刊 | 月份数 | α 均值(%) | 最新月 α(%) | 总句数 |\n")
        f.write("|---|---|---|---|---|\n")
        for _, row in summary.iterrows():
            f.write(f"| {row['journal']} | {row['n_months']} | {row['mean_alpha']:.2f} | {row['latest_alpha']:.2f} | {row['total_sentences']} |\n")

        f.write("\n## 逐刊详细结果\n\n")
        for j in JOURNALS:
            dir_name = j["dir"]
            df_j = df_all[df_all['journal'] == dir_name]
            if len(df_j) == 0:
                f.write(f"### {dir_name}\n\n无数据\n\n")
                continue
            f.write(f"### {dir_name}\n\n")
            f.write("| year | month | alpha(%) | ci | n_sentences |\n")
            f.write("|---|---|---|---|---|\n")
            for _, r in df_j.iterrows():
                f.write(f"| {r['year']} | {r['month']} | {r['alpha']:.1f} | {r['ci']:.2f} | {r['n_sentences']} |\n")
            f.write("\n")

        f.write("## 注意事项\n\n")
        f.write("- α 为 MLE 估计的 AI 参与比例，CI 为 Bootstrap 95% 置信区间半宽\n")
        f.write("- 部分月份样本量极少（<5 句），α 估计不稳定\n")
        f.write("- eLife 的 publication_date 存在批量异常（1261 条集中在 2017-11）\n")
        f.write("- Nat Commun 无真审稿日期，只能做发表 cohort 分析\n")
        f.write("- 趋势图和柱状图见同目录下的 PNG 文件\n")

    print(f"✅ 报告: {report_path}")


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='六刊 MLE Pipeline 主入口')
    parser.add_argument('--skip_ai', action='store_true', help='跳过 AI 语料生成（使用已有 Q）')
    parser.add_argument('--skip_export', action='store_true', help='跳过数据导出（已有 CSV）')
    parser.add_argument('--only_summary', action='store_true', help='仅生成跨刊汇总')
    parser.add_argument('--only_journal', type=str, default=None, help='仅执行指定期刊')
    args = parser.parse_args()

    if args.only_summary:
        cross_journal_summary()
        return

    # 加载 spaCy
    print("加载 spaCy 模型 (en_core_web_lg)...")
    nlp = spacy.load("en_core_web_lg")

    journals_to_run = JOURNALS
    if args.only_journal:
        journals_to_run = [j for j in JOURNALS if j["dir"] == args.only_journal]

    for j in journals_to_run:
        dir_name = j["dir"]

        # Step 1: 导出 H 语料
        if not args.skip_export:
            n_h = export_human_csv(j, nlp)

        # Step 2: 导出全量推断数据
        if not args.skip_export:
            n_inf = export_inference_data(j, nlp)

        # Step 3: 生成 AI 语料
        if not args.skip_ai:
            n_q = generate_ai_corpus(j)

        # Step 4: 构建分布 + MLE 推断
        run_pipeline(j)

    # Step 5: 跨刊汇总
    cross_journal_summary()

    print(f"\n{'='*60}")
    print(f"✅ 全部完成！")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
