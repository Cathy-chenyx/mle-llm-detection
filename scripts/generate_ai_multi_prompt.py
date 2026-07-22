"""
多 Prompt 分层 AI 语料生成 — Anthropic 格式 API 版
=====================================================
生成 A/B/C/D 四个级别的 Q 语料，用于测试梯度 AI 介入效应。

用法：
  python generate_ai_multi_prompt.py

输出：processed_data/elife/ai_corpus/elife_ai_{A,B,C,D}.parquet
"""

import pandas as pd
import os
import time
import sys
from pathlib import Path
import numpy as np
import anthropic

# ============================================================
# API 配置
# ============================================================
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
BASE_URL = "http://115.190.192.101:3000"
MODEL = "deepseek-v4-flash"

client = anthropic.Anthropic(
    api_key=API_KEY,
    base_url=BASE_URL,
    timeout=120.0
)

# ============================================================
# 路径配置
# ============================================================
sys.path.insert(0, str(Path(__file__).parent))
PROJECT_DIR = Path("/Users/cathy/Documents/学习相关/老段课题组/AI_project")
HUMAN_PATH = PROJECT_DIR / "processed_data/elife/human_corpus/elife_human.parquet"
AI_OUT_DIR = PROJECT_DIR / "processed_data/elife/ai_corpus"
AI_OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Prompt 定义
# ============================================================
PROMPTS = {
    "A": {
        "name": "proofread",
        "description": "Proofread only — fix spelling/grammar, preserve all words and structure",
        "temperature": 0.3,
        "max_tokens": 16384,
        "build_messages": lambda text: [{
            "role": "user",
            "content": (
                "Proofread the following peer review. Fix ONLY spelling, grammar, and punctuation errors.\n"
                "Do NOT change any word choices, sentence structures, or content.\n"
                "Output ONLY the corrected review, no preamble.\n\n"
                f"{text[:8000]}"
            )
        }]
    },
    "B": {
        "name": "rewrite",
        "description": "Rewrite — change wording and structure, preserve all factual content",
        "temperature": 0.5,
        "max_tokens": 4096,
        "build_messages": lambda text: [{
            "role": "user",
            "content": (
                "Rewrite the following peer review. Preserve ALL factual content, criticisms, and key points.\n"
                "Change sentence structures, word choices, and phrasing to express the same ideas differently.\n"
                "Use formal academic language. Do NOT add new factual content.\n"
                "Output ONLY the rewritten review, no preamble.\n\n"
                f"{text[:8000]}"
            )
        }]
    },
    "C": {
        "name": "expand_outline",
        "description": "Two-stage: extract bullet points → expand into full review (Liang 2025 method)",
        "temperature_stage1": 0.3,
        "max_tokens_stage1": 4096,
        "temperature_stage2": 0.7,
        "max_tokens_stage2": 4096,
        "build_stage1_messages": lambda text: [{
            "role": "user",
            "content": (
                "Extract the key factual points from this peer review as a numbered bullet list.\n"
                "Each bullet must be a short factual statement. Strip all stylistic language, hedging,\n"
                "and evaluative phrases. Focus on methodological concerns, statistical issues,\n"
                "interpretation points, and factual corrections. Output ONLY the bullet list.\n\n"
                f"{text[:8000]}"
            )
        }],
        "build_stage2_messages": lambda outline: [{
            "role": "user",
            "content": (
                "Expand the following bullet outline into a full peer review paragraph.\n"
                "Use formal academic language and complete sentences. Naturally incorporate\n"
                "common academic phrases. Cover ALL bullets from the outline, in order.\n"
                "Do NOT add new factual content beyond what is in the outline.\n"
                "Output ONLY the review paragraph, no preamble.\n\n"
                f"{outline}"
            )
        }]
    },
    "D": {
        "name": "act_as_reviewer",
        "description": "Act as reviewer — write independent review using human review as reference only",
        "temperature": 0.7,
        "max_tokens": 4096,
        "build_messages": lambda text: [{
            "role": "user",
            "content": (
                "You are a peer reviewer for an academic journal. Write a critical peer review for a manuscript.\n"
                "Below is a review written by another reviewer for the same paper.\n"
                "Use it as REFERENCE ONLY for understanding the paper's content — do NOT copy its structure,\n"
                "phrasing, or word choices. Write your OWN independent review covering methodological\n"
                "concerns, statistical issues, interpretation points, and factual corrections.\n"
                "Use formal academic language. Output ONLY the review paragraph, no preamble.\n\n"
                f"{text[:8000]}"
            )
        }]
    }
}


# ============================================================
# 工具函数
# ============================================================
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


def generate_single_stage(human_text, level, idx, total):
    """单阶段生成（A, B, D）"""
    cfg = PROMPTS[level]
    messages = cfg["build_messages"](human_text)
    
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=cfg["max_tokens"],
            temperature=cfg["temperature"],
            messages=messages
        )
        return resp.content[0].text.strip()
    except Exception as e:
        print(f"  [{idx}/{total}] {level} 失败: {e}")
        return None


def generate_two_stage(human_text, idx, total):
    """两阶段生成（C）"""
    cfg = PROMPTS["C"]
    
    # Stage 1
    messages1 = cfg["build_stage1_messages"](human_text)
    try:
        resp1 = client.messages.create(
            model=MODEL,
            max_tokens=cfg["max_tokens_stage1"],
            temperature=cfg["temperature_stage1"],
            messages=messages1
        )
        outline = resp1.content[0].text.strip()
    except Exception as e:
        print(f"  [{idx}/{total}] C-Stage1 失败: {e}")
        return None
    
    # Stage 2
    messages2 = cfg["build_stage2_messages"](outline)
    try:
        resp2 = client.messages.create(
            model=MODEL,
            max_tokens=cfg["max_tokens_stage2"],
            temperature=cfg["temperature_stage2"],
            messages=messages2
        )
        return resp2.content[0].text.strip()
    except Exception as e:
        print(f"  [{idx}/{total}] C-Stage2 失败: {e}")
        return None


# ============================================================
# 主流程
# ============================================================
def main():
    print(f"API: {BASE_URL}  |  模型: {MODEL}")
    
    # 加载 H 语料
    print(f"\n加载 H 语料: {HUMAN_PATH}")
    df_h = pd.read_parquet(HUMAN_PATH)
    print(f"  H 语料: {len(df_h)} 条")
    
    # 还原文本
    print("还原人类文本...")
    df_h['human_text'] = df_h['human_sentence'].apply(reconstruct_text)
    
    total = len(df_h)
    levels = ["A", "B", "C", "D"]
    results = {}
    
    for level in levels:
        cfg = PROMPTS[level]
        is_two_stage = (level == "C")
        
        print(f"\n{'='*60}")
        print(f"Level {level} — {cfg.get('name', cfg['name'])} ({'两阶段' if is_two_stage else '单阶段'})")
        print(f"{'='*60}")
        
        ai_texts = []
        for i, row in df_h.iterrows():
            idx = len(ai_texts) + 1
            print(f"  [{idx}/{total}] {level} 生成中...", end=" ", flush=True)
            
            if is_two_stage:
                ai_text = generate_two_stage(row['human_text'], idx, total)
            else:
                ai_text = generate_single_stage(row['human_text'], level, idx, total)
            
            if ai_text:
                ai_texts.append(ai_text)
                preview = ai_text[:80].replace('\n', ' ')
                print(f"OK | {preview}...")
            else:
                print("跳过")
            
            if idx < total:
                time.sleep(0.5)
        
        n_success = len(ai_texts)
        print(f"\n  {level}: {n_success}/{total} 成功")
        
        if n_success > 0:
            df_q = pd.DataFrame({'ai_review_text': ai_texts})
            out_path = AI_OUT_DIR / f"elife_ai_{level}.parquet"
            df_q.to_parquet(out_path, index=False)
            print(f"  已保存: {out_path}")
            results[level] = n_success
        else:
            print(f"  警告: {level} 全部失败")
            results[level] = 0
    
    # 最终汇总
    print(f"\n{'='*60}")
    print("生成汇总")
    print(f"{'='*60}")
    for level in levels:
        print(f"  Level {level} ({PROMPTS[level].get('name', PROMPTS[level]['name'])}): {results.get(level, 0)}/{total}")


if __name__ == "__main__":
    main()
