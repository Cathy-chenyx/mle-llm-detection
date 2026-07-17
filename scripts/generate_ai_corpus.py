"""
两阶段生成 AI 参考语料 (Q) — Anthropic 格式 API 版
====================================================
用法：
  python generate_ai_corpus.py

输入：processed_data/elife/human_corpus/elife_human.parquet
输出：processed_data/elife/ai_corpus/elife_ai.parquet
"""

import pandas as pd
import os
import time
from pathlib import Path
import anthropic

# ============================================================
# API 配置
# ============================================================
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
BASE_URL = "http://115.190.192.101:3000"
MODEL = "deepseek-v4-flash"  # 或 deepseek-v4-pro

client = anthropic.Anthropic(
    api_key=API_KEY,
    base_url=BASE_URL,
    timeout=120.0
)

# ============================================================
# 路径配置
# ============================================================
PROJECT_DIR = Path("/Users/cathy/Documents/学习相关/老段课题组/AI_project")
HUMAN_PATH = PROJECT_DIR / "processed_data/elife/human_corpus/elife_human.parquet"
AI_OUT_DIR = PROJECT_DIR / "processed_data/elife/ai_corpus"
AI_OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = None  # None = 全用 40 条

# ============================================================
# 提示词（Anthropic 格式：system 单独传入）
# ============================================================
STAGE1_SYSTEM = "You are a research methodologist. Extract the key factual points from a peer review as a numbered bullet list. Each bullet must be a short factual statement. Strip all stylistic language, hedging, and evaluative phrases. Focus on methodological concerns, statistical issues, interpretation points, and factual corrections. Do NOT add any information not present in the original review. Output ONLY the bullet list."

STAGE2_SYSTEM = "You are writing a peer review for an academic journal. Expand the given bullet outline into a full peer review paragraph. Use formal academic language and complete sentences. Naturally incorporate common academic phrases. Cover ALL bullets from the outline, in order. Do NOT add new factual content beyond what is in the outline. Do NOT fabricate citations or data. Output ONLY the review paragraph, no preamble."

# ============================================================
# 工具函数
# ============================================================
import numpy as np

def reconstruct_text(sentence_lists):
    """将 token 列表还原为原始文本"""
    sentences = []
    for sent in sentence_lists:
        if isinstance(sent, (list, np.ndarray)):
            words = [str(w) for w in sent]
            sentences.append(" ".join(words))
        elif isinstance(sent, str):
            sentences.append(sent)
    return " ".join(sentences)


def generate_ai_review(human_text, idx, total):
    """两阶段生成一条 AI 审稿 (Anthropic API)"""
    
    text_truncated = human_text[:8000]
    
    # Stage 1: 摘要 (指令嵌入 user message)
    stage1_prompt = (
        "Extract the key factual points from this peer review as a numbered bullet list. "
        "Output ONLY the bullet list, no preamble.\n\n"
        f"{text_truncated}"
    )
    try:
        resp1 = client.messages.create(
            model=MODEL,
            max_tokens=1000,
            temperature=0.3,
            messages=[{"role": "user", "content": stage1_prompt}]
        )
        outline = resp1.content[0].text.strip()
    except Exception as e:
        print(f"  [{idx}/{total}] Stage 1 失败: {e}")
        return None
    
    # Stage 2: 扩写 (指令嵌入 user message)
    stage2_prompt = (
        "Expand the following bullet outline into a full peer review paragraph. "
        "Use formal academic language and complete sentences. Naturally incorporate common "
        "academic phrases. Cover ALL bullets from the outline, in order. Do NOT add new "
        "factual content beyond what is in the outline. Do NOT fabricate citations or data. "
        "Output ONLY the review paragraph, no preamble.\n\n"
        f"{outline}"
    )
    try:
        resp2 = client.messages.create(
            model=MODEL,
            max_tokens=2000,
            temperature=0.7,
            messages=[{"role": "user", "content": stage2_prompt}]
        )
        ai_review = resp2.content[0].text.strip()
    except Exception as e:
        print(f"  [{idx}/{total}] Stage 2 失败: {e}")
        return None
    
    return ai_review


# ============================================================
# 主流程
# ============================================================
def main():
    print(f"API: {BASE_URL}  |  模型: {MODEL}")
    
    # 加载 H 语料
    print(f"\n加载 H 语料: {HUMAN_PATH}")
    df_h = pd.read_parquet(HUMAN_PATH)
    print(f"  H 语料: {len(df_h)} 条")
    
    if SAMPLE_SIZE and SAMPLE_SIZE < len(df_h):
        df_h = df_h.sample(n=SAMPLE_SIZE, random_state=42)
        print(f"  采样 {SAMPLE_SIZE} 条")
    
    # 还原文本
    print("还原人类文本...")
    df_h['human_text'] = df_h['human_sentence'].apply(reconstruct_text)
    
    total = len(df_h)
    print(f"\n两阶段生成开始 (共 {total} 条，预计 {total * 5:.0f} 秒)...\n")
    
    ai_texts = []
    for i, row in df_h.iterrows():
        idx = len(ai_texts) + 1
        print(f"  [{idx}/{total}] 生成中...", end=" ", flush=True)
        ai_text = generate_ai_review(row['human_text'], idx, total)
        if ai_text:
            ai_texts.append(ai_text)
            preview = ai_text[:80].replace('\n', ' ')
            print(f"OK | {preview}...")
        else:
            print("跳过")
        if idx < total:
            time.sleep(0.5)
    
    n_success = len(ai_texts)
    print(f"\n生成完成: {n_success}/{total} 成功")
    
    if n_success == 0:
        print("错误: 没有成功生成任何 AI 语料")
        return
    
    df_q = pd.DataFrame({'ai_review_text': ai_texts})
    out_path = AI_OUT_DIR / "elife_ai.parquet"
    df_q.to_parquet(out_path, index=False)
    print(f"Q 语料已保存: {out_path}")
    
    # 样例对比
    print(f"\n--- 样例对比 (第 1 条) ---")
    print(f"人类原文 ({len(df_h.iloc[0]['human_text'])} chars):")
    print(df_h.iloc[0]['human_text'][:250])
    print(f"\nAI 改写 ({len(ai_texts[0])} chars):")
    print(ai_texts[0][:250])
    
    print(f"\n下一步: python build_distribution.py")


if __name__ == "__main__":
    main()
