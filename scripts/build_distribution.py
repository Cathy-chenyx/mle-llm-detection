"""
构建 eLife 审稿意见专属分布 (distribution parquet)
====================================================
步骤：
  1. 加载 H 语料 (已 tokenize 的 human_sentence)
  2. 加载 Q 语料 (AI 生成的原始文本 → tokenize)
  3. 对齐格式 (human_sentence + ai_sentence)
  4. 调用 estimation.py 计算 logP / logQ / log(1-P) / log(1-Q)
  5. 输出 distribution/elife.parquet

前置条件：先运行 generate_ai_corpus.py 生成 Q 语料
"""

import sys
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import re
import spacy
from pathlib import Path
from src.estimation import (
    count_human_binary_word_occurrences,
    count_ai_binary_word_occurrences,
    estimate_log_probabilities,
    get_vocabulary_intersection,
    filter_frequent_words,
    calculate_log_probability,
)

# ============================================================
# 配置
# ============================================================
PROJECT_DIR = Path("/Users/cathy/Documents/学习相关/老段课题组/AI_project")
H_PATH = PROJECT_DIR / "processed_data/elife/human_corpus/elife_human.parquet"
Q_RAW_PATH = PROJECT_DIR / "processed_data/elife/ai_corpus/elife_ai.parquet"
DIST_OUT_DIR = PROJECT_DIR / "processed_data/elife/distribution"
DIST_OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. 加载 spaCy（与 preprocess_elife.py 一致的配置）
# ============================================================
print("加载 spaCy...")
nlp = spacy.load("en_core_web_lg")

def tokenize(text):
    """与 tokenize_demo.ipynb 完全一致"""
    text = text.replace('\n', ' ')
    sentence_list = []
    doc = nlp(text)
    for sent in doc.sents:
        words = re.findall(r'\b\w+\b', sent.text.lower())
        words = [w for w in words if not w.isdigit()]
        if len(words) > 0:
            sentence_list.append(words)
    return sentence_list

# ============================================================
# 2. 加载 H 语料
# ============================================================
print(f"\n加载 H 语料: {H_PATH}")
df_h = pd.read_parquet(H_PATH)
print(f"  H 语料: {len(df_h)} 条, 列名: {df_h.columns.tolist()}")
# human_sentence 列已存在，无需处理

# ============================================================
# 3. 加载 Q 语料并 tokenize
# ============================================================
print(f"\n加载 Q 语料: {Q_RAW_PATH}")
df_q = pd.read_parquet(Q_RAW_PATH)
print(f"  Q 语料: {len(df_q)} 条")

print("对 Q 语料进行 spaCy 分词...")
df_q['ai_sentence'] = df_q['ai_review_text'].apply(tokenize)

# 统计
q_sentences = df_q['ai_sentence'].apply(len).sum()
q_tokens = df_q['ai_sentence'].apply(lambda x: sum(len(s) for s in x)).sum()
print(f"  Q 语料: {q_sentences} 句, {q_tokens} 词")

# ============================================================
# 4. 对齐 H 和 Q（取较短的进行配对）
# ============================================================
n = min(len(df_h), len(df_q))
print(f"\n对齐 H 和 Q: 各取 {n} 条")

df_combined = pd.DataFrame({
    'human_sentence': df_h['human_sentence'].iloc[:n].values,
    'ai_sentence': df_q['ai_sentence'].iloc[:n].values
})

# ============================================================
# 5. 转为 Python list 并过滤短句（同 estimate_text_distribution 逻辑）
# ============================================================
def to_list_of_lists(series):
    """将 numpy object array 转为 Python list of lists"""
    result = []
    for item in series:
        sentences = [list(s) for s in item]
        sentences = [s for s in sentences if len(s) > 1]
        result.append(sentences)
    return result

print("转换 H 语料格式...")
h_sentences_list = to_list_of_lists(df_combined['human_sentence'])
q_sentences_list = to_list_of_lists(df_combined['ai_sentence'])

# 展开为 flat table（一行一句）
h_flat = pd.DataFrame({
    'human_sentence': [s for doc in h_sentences_list for s in doc]
}).dropna()
q_flat = pd.DataFrame({
    'ai_sentence': [s for doc in q_sentences_list for s in doc]
}).dropna()
print(f"  H: {len(h_flat)} 句, Q: {len(q_flat)} 句")

# ============================================================
# 6. 直接调用 estimation.py 内层函数（避免 parquet 回读 numpy 问题）
# ============================================================
out_path = DIST_OUT_DIR / "elife.parquet"
print(f"\n计算词频分布...")

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
dist_df.to_parquet(out_path, index=False)
print(f"  词典大小: {len(dist_df)} 个词")
print(f"  列名: {dist_df.columns.tolist()}")

print("\n高频词 Top 10:")
print(dist_df.head(10)[['Word', 'logP', 'logQ']].to_string(index=False))

print(f"\n分布已保存至: {out_path}")

# 显示 logP 和 logQ 差异最大的词
dist_df['diff'] = abs(dist_df['logP'] - dist_df['logQ'])
top_diff = dist_df.nlargest(20, 'diff')[['Word', 'logP', 'logQ', 'diff']]
print(f"\n区分度最高的 20 个词 (|logP - logQ| 最大):")
print(top_diff.to_string(index=False))

print(f"\n下一步: 修改 run_elife_pipeline.py 中的 DIST_PATH 为:")
print(f"  {out_path}")
print(f"  然后重新运行 run_elife_pipeline.py")
