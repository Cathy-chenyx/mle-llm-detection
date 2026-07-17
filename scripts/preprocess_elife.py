"""
eLife 审稿意见预处理脚本
=========================
功能：读取 eLife CSV → 清洗 HTML → spaCy 分句分词 → 按月输出 parquet
输出：
  1. inference_data/elife/{year}_{month}.parquet  用于 MLE 推断
  2. inference_data/elife/all_cleaned.parquet      全部清洗后的数据（备用）
  3. human_corpus/elife_human.parquet              ChatGPT前（≤2022.10）人类参考语料
"""

import pandas as pd
import re
import os
import spacy
from pathlib import Path

# ============================================================
# 配置
# ============================================================
PROJECT_DIR = Path("/Users/cathy/Documents/学习相关/老段课题组/AI_project")
DATA_DIR = PROJECT_DIR / "review_text_export"
OUT_DIR = PROJECT_DIR / "processed_data" / "elife"

INPUT_FILE = DATA_DIR / "eLife_reviews_2021_2024.csv"
INFERENCE_DIR = OUT_DIR / "inference_data"
HUMAN_CORPUS_DIR = OUT_DIR / "human_corpus"
INFERENCE_DIR.mkdir(parents=True, exist_ok=True)
HUMAN_CORPUS_DIR.mkdir(parents=True, exist_ok=True)

CHATGPT_CUTOFF = (2022, 11)  # ChatGPT 发布时间：2022年11月30日

# ============================================================
# 1. 文本清洗
# ============================================================
def clean_html(text):
    """去除 HTML 标签，保留纯文本"""
    text = re.sub(r'<[^>]+>', '', text)          # 去除 <b>, <i> 等所有 HTML 标签
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&nbsp;', ' ', text)
    return text

def clean_editor_notes(text):
    """清理 [Editors' note: ...] 等编辑注释（保留后面正文）"""
    # 去除方括号编辑器注释，保留后面的正文
    text = re.sub(r'\[Editors.? note:.*?\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[#.*?\]', '', text)
    return text

def clean_section_headers(text):
    """清理无意义的节标题（单独成行的那种）"""
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
    """综合清洗"""
    text = str(text)
    text = clean_html(text)
    text = clean_editor_notes(text)
    text = clean_section_headers(text)
    # 压缩多余空白
    text = re.sub(r'\n{3,}', '\n\n', text)         # 多个换行 → 双换行
    text = re.sub(r' {2,}', ' ', text)             # 多个空格 → 单空格
    text = text.strip()
    return text

# ============================================================
# 2. spaCy 分词（完全复用 Liang 论文的 tokenize 逻辑）
# ============================================================
print("加载 spaCy 模型 (en_core_web_lg)...")
nlp = spacy.load("en_core_web_lg")

def tokenize(text):
    """
    与 tokenize_demo.ipynb 完全一致的逻辑：
    1. 用 spaCy 分句
    2. 每句提取纯字母词（小写）
    3. 去除纯数字词
    """
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
# 3. 主流程
# ============================================================
def main():
    print(f"读取数据: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f"原始条目数: {len(df)}")

    # 清洗文本
    print("清洗 HTML 标签和编辑器注释...")
    df['cleaned_text'] = df['review_text'].apply(clean_text)

    # 过滤清洗后为空的条目
    before = len(df)
    df = df[df['cleaned_text'].str.len() > 50].copy()
    print(f"清洗后过滤空条目: {before} → {len(df)} (移除 {before - len(df)} 条)")

    # 分词
    print("spaCy 分句/分词中...（可能需要几分钟）")
    df['inference_sentence'] = df['cleaned_text'].apply(tokenize)

    # 统计
    total_sentences = df['inference_sentence'].apply(len).sum()
    total_tokens = df['inference_sentence'].apply(lambda x: sum(len(s) for s in x)).sum()
    print(f"总计: {total_sentences} 句, {total_tokens} 词")
    print(f"月均: {total_sentences / (48 if len(df['year'].unique()) == 4 else 36):.0f} 句")

    # ---- 3a. 保存推断数据：按年月分组 ----
    print("\n--- 保存月度推断数据 ---")
    for (year, month), group in df.groupby(['year', 'month']):
        out_path = INFERENCE_DIR / f"{year}_{month}.parquet"
        group[['inference_sentence']].to_parquet(out_path, index=False)
        print(f"  {year}_{month}: {len(group)} 条 → {out_path.name}")

    # ---- 3b. 保存人类参考语料（≤ 2022.10）----
    print("\n--- 生成人类参考语料 (ChatGPT 发布前) ---")
    human_mask = (
        (df['year'] < CHATGPT_CUTOFF[0]) |
        ((df['year'] == CHATGPT_CUTOFF[0]) & (df['month'] < CHATGPT_CUTOFF[1]))
    )
    human_df = df[human_mask].copy()
    human_path = HUMAN_CORPUS_DIR / "elife_human.parquet"
    # 列名匹配 estimation.py 期望的列名
    human_df = human_df.rename(columns={'inference_sentence': 'human_sentence'})
    human_df[['human_sentence']].to_parquet(human_path, index=False)
    print(f"  人类参考语料: {len(human_df)} 条 → {human_path}")

    # ---- 3c. 年份分布总览 ----
    print("\n--- 年份分布---")
    summary = df.groupby('year').agg(
        条目数=('cleaned_text', 'count'),
        平均字数=('cleaned_text', lambda x: int(x.str.len().mean())),
        总句子数=('inference_sentence', lambda x: x.apply(len).sum())
    )
    print(summary.to_string())

    print("\n✅ 预处理完成。输出文件：")
    print(f"  月度推断数据: {INFERENCE_DIR}/")
    print(f"  人类参考语料: {human_path}")
    print(f"\n下一步：")
    print(f"  1. 生成 AI 参考语料 (Q) → 运行 generate_ai_corpus.py")
    print(f"  2. 估计分布 → estimation.py")
    print(f"  3. MLE 推断 → MLE.py 或 increasing_temporal.ipynb")

if __name__ == "__main__":
    main()
