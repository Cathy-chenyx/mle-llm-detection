"""
多 Prompt 分层 Pipeline：A/B/C/D 四个级别完整体验
=====================================================
对每个 prompt 级别：生成 Q 语料 → 构建分布词典 → MLE 推断 → 汇总对比

用法：
  python run_multi_prompt_pipeline.py
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from src.estimation import (
    count_human_binary_word_occurrences,
    count_ai_binary_word_occurrences,
    estimate_log_probabilities,
    filter_frequent_words,
    get_vocabulary_intersection,
    calculate_log_probability,
)
from src.MLE import MLE

# ============================================================
# 路径配置
# ============================================================
PROJECT_DIR = Path("/Users/cathy/Documents/学习相关/老段课题组/AI_project")
ELIFE_DIR = PROJECT_DIR / "processed_data/elife"
H_PATH = ELIFE_DIR / "human_corpus/elife_human.parquet"
AI_DIR = ELIFE_DIR / "ai_corpus"
INFERENCE_DIR = ELIFE_DIR / "inference_data"
DIST_DIR = ELIFE_DIR / "distribution"
OUTPUT_DIR = ELIFE_DIR

LEVELS = ["A", "B", "C", "D"]

# ============================================================
# 步骤 1：读取 / 分词 Q 语料（复用 原 build_distribution 逻辑）
# ============================================================
import re
import spacy
nlp = spacy.load("en_core_web_lg")

def tokenize(text):
    """与 tokenize_demo.ipynb / build_distribution.py 一致的分句+分词逻辑"""
    text = text.replace('\n', ' ')
    sentence_list = []
    doc = nlp(text)
    for sent in doc.sents:
        words = re.findall(r'\b\w+\b', sent.text.lower())
        words = [w for w in words if not w.isdigit()]
        if len(words) > 0:
            sentence_list.append(words)
    return sentence_list


def to_list_of_lists(sentence_series):
    """将 parquet 中的句子列转为 Python list of lists"""
    result = []
    for item in sentence_series:
        if isinstance(item, (list, np.ndarray)):
            result.append([list(w) for w in item])
        else:
            result.append([])
    return result


def build_distribution(h_sentences, q_texts, level_label):
    """从 H 句子列表 + Q 文本列表构建分布词典（对齐 build_distribution.py 模式）"""
    # Q 分句+分词（per document → list of sentence token lists）
    q_sentence_list = []
    for text in q_texts:
        sentences = tokenize(text)
        q_sentence_list.append(sentences)
    
    # 对齐
    n = min(len(h_sentences), len(q_sentence_list))
    h_sentences = h_sentences[:n]
    q_sentence_list = q_sentence_list[:n]
    
    # 展开为 flat table（一行一句）
    h_flat = pd.DataFrame({
        'human_sentence': [s for doc in h_sentences for s in doc]
    }).dropna()
    q_flat = pd.DataFrame({
        'ai_sentence': [s for doc in q_sentence_list for s in doc]
    }).dropna()
    print(f"  [{level_label}] H: {len(h_flat)} 句, Q: {len(q_flat)} 句")
    
    # 直接调用 estimation.py 内层函数（与 build_distribution.py 完全一致）
    human_word_counts = count_human_binary_word_occurrences(h_flat)
    ai_word_counts = count_ai_binary_word_occurrences(q_flat)
    total_human_sentences = len(h_flat)
    total_ai_sentences = len(q_flat)
    
    human_log_probs = estimate_log_probabilities(human_word_counts, total_human_sentences)
    ai_log_probs = estimate_log_probabilities(ai_word_counts, total_ai_sentences)
    
    common_vocab = get_vocabulary_intersection(human_word_counts, ai_word_counts)
    frequent_human_words = filter_frequent_words(human_word_counts, 5)
    frequent_ai_words = filter_frequent_words(ai_word_counts, 3)
    frequent_common_vocab = common_vocab.intersection(
        frequent_human_words.keys(), frequent_ai_words.keys()
    )
    
    dist_df = calculate_log_probability(human_log_probs, ai_log_probs, frequent_common_vocab)
    
    dist_path = DIST_DIR / f"elife_{level_label}.parquet"
    dist_df.to_parquet(dist_path, index=False)
    print(f"  [{level_label}] 分布词典: {len(dist_df)} 词 → {dist_path}")
    return dist_path, len(dist_df)


# ============================================================
# 步骤 2：MLE 推断
# ============================================================
def run_mle(dist_path, level_label):
    """对分布词典 run MLE 推断"""
    mle = MLE(str(dist_path))
    results = []
    
    for parquet_file in sorted(INFERENCE_DIR.glob("*.parquet")):
        fname = parquet_file.stem  # e.g. "2024_12"
        year, month = fname.split("_")
        
        try:
            alpha, ci = mle.inference(str(parquet_file), exploded_data=False)
            results.append({
                'year': int(year),
                'month': int(month),
                'alpha': alpha,
                'ci': ci
            })
        except Exception as e:
            print(f"    [{fname}] MLE 失败: {e}")
            continue
    
    df_res = pd.DataFrame(results)
    out_csv = OUTPUT_DIR / f"elife_alpha_{level_label}.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"  [{level_label}] MLE 完成 → {out_csv}")
    return df_res


# ============================================================
# 步骤 3：汇总对比图
# ============================================================
def plot_comparison(all_results):
    """四级别对比图"""
    colors = {'A': '#2ecc71', 'B': '#3498db', 'C': '#f39c12', 'D': '#e74c3c'}
    labels = {'A': 'A: Proofread', 'B': 'B: Rewrite', 'C': 'C: Expand Outline', 'D': 'D: Act as Reviewer'}
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for level in LEVELS:
        if level not in all_results or all_results[level].empty:
            continue
        
        df = all_results[level].sort_values(['year', 'month'])
        x_labels = [f"{int(r['year'])}-{int(r['month']):02d}" for _, r in df.iterrows()]
        x = range(len(x_labels))
        
        alphas = df['alpha'].values * 100
        cis = df['ci'].values * 100
        
        ax.plot(x, alphas, 'o-', color=colors[level], label=labels[level], linewidth=1.5, markersize=4)
        ax.fill_between(x, alphas - cis, alphas + cis, color=colors[level], alpha=0.1)
    
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(x=7.5, color='gray', linestyle=':', linewidth=0.5, label='ChatGPT release (~2022.11)')
    
    ax.set_xlabel('Month')
    ax.set_ylabel('Estimated AI Contribution α (%)')
    ax.set_title('eLife — Multi-Prompt α Comparison (DeepSeek-v4-Flash)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # X 轴标签采样
    all_labels = sorted(set([
        f"{int(r['year'])}-{int(r['month']):02d}"
        for level in all_results.values()
        for _, r in level.iterrows()
    ]))
    ax.set_xticks(range(len(all_labels)))
    ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=8)
    
    plt.tight_layout()
    out_png = OUTPUT_DIR / "elife_multi_prompt_comparison.png"
    fig.savefig(out_png, dpi=150)
    print(f"\n对比图已保存: {out_png}")
    return out_png


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 60)
    print("多 Prompt 分层 Pipeline")
    print("=" * 60)
    
    # 加载 H 语料句子
    print(f"\n加载 H 语料: {H_PATH}")
    df_h = pd.read_parquet(H_PATH)
    h_sentences = to_list_of_lists(df_h['human_sentence'])
    print(f"  H 句子总数: {sum(len(s) for s in h_sentences)} (来自 {len(h_sentences)} 条审稿)")
    
    all_results = {}
    dist_sizes = {}
    
    for level in LEVELS:
        q_path = AI_DIR / f"elife_ai_{level}.parquet"
        if not q_path.exists():
            print(f"\n[{level}] Q 语料不存在，跳过: {q_path}")
            continue
        
        print(f"\n{'='*40}")
        print(f"[{level}] 加载 Q 语料: {q_path}")
        df_q = pd.read_parquet(q_path)
        q_texts = df_q['ai_review_text'].tolist()
        print(f"  Q 语料: {len(q_texts)} 条")
        
        # 构建分布
        print(f"\n[{level}] 构建分布词典...")
        dist_path, n_words = build_distribution(h_sentences, q_texts, level)
        dist_sizes[level] = n_words
        
        # MLE 推断
        print(f"\n[{level}] MLE 推断...")
        df_res = run_mle(dist_path, level)
        all_results[level] = df_res
    
    # 汇总表
    print(f"\n{'='*60}")
    print("结果汇总")
    print(f"{'='*60}")
    print(f"{'Level':<8} {'Name':<22} {'Vocab':<8} {'Latest α':<10}")
    print("-" * 54)
    
    level_names = {'A': 'Proofread', 'B': 'Rewrite', 'C': 'Expand Outline', 'D': 'Act as Reviewer'}
    for level in LEVELS:
        name = level_names.get(level, "?")
        vocab = dist_sizes.get(level, 0)
        latest = ""
        if level in all_results and not all_results[level].empty:
            last = all_results[level].iloc[-1]
            latest = f"{last['alpha']*100:.1f}% ± {last['ci']*100:.1f}%"
        print(f"{level:<8} {name:<22} {vocab:<8} {latest}")
    
    # 对比图
    if all_results:
        plot_comparison(all_results)
    
    print("\n完成。")


if __name__ == "__main__":
    main()
