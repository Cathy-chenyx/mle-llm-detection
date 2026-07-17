"""
用 eLife 数据 + 论文 CS 分布（占位）跑通 MLE 推断→绘图全链路
注意：结果无科学意义（分布来自 CS 论文摘要，不是医学审稿），仅用于验证 pipeline
"""

import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.MLE import MLE
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# ============================================================
# 配置
# ============================================================
DIST_PATH = "/Users/cathy/Documents/学习相关/老段课题组/AI_project/processed_data/elife/distribution/elife.parquet"
INFERENCE_DIR = "/Users/cathy/Documents/学习相关/老段课题组/AI_project/processed_data/elife/inference_data"
OUTPUT_DIR = "/Users/cathy/Documents/学习相关/老段课题组/AI_project/processed_data/elife"

# ============================================================
# 1. 加载分布（用论文 CS 占位）
# ============================================================
print(f"加载分布: {DIST_PATH}")
model = MLE(DIST_PATH)

# ============================================================
# 2. 对每个月跑 MLE 推断
# ============================================================
print("\n逐月推断:")
results = []

for fname in sorted(os.listdir(INFERENCE_DIR)):
    if not fname.endswith(".parquet"):
        continue
    path = os.path.join(INFERENCE_DIR, fname)
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
            "n_sentences": len(pd.read_parquet(path))
        })
        print(f"  {year_month}: α={alpha*100:.1f}% ± {ci*100:.2f}%  ({len(pd.read_parquet(path))} 条)")
    except Exception as e:
        print(f"  {year_month}: 失败 - {e}")

# ============================================================
# 3. 绘图
# ============================================================
df_r = pd.DataFrame(results)
df_r = df_r.sort_values("time")

plt.rcParams.update({'font.size': 14})
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti SC', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots(figsize=(12, 6))

# ChatGPT 发布时间线
chatgpt_time = 2022 + 10/12  # 2022年11月
ax.axvline(x=chatgpt_time, color='darkred', linestyle='--', linewidth=1.5, alpha=0.7)
ax.text(chatgpt_time - 0.05, ax.get_ylim()[1]*0.85 if ax.get_ylim()[1] > 0 else 20,
        "ChatGPT\n2022.11", color='darkred', ha='right', va='top', fontsize=12)

ax.errorbar(df_r['time'], df_r['alpha'], yerr=df_r['ci'],
            fmt='o-', color='#2b83ba', markersize=6, capsize=4,
            elinewidth=1, linewidth=1.5, label='eLife (占位分布)')

ax.set_xlabel('Year')
ax.set_ylabel('Estimated α (%)')
ax.set_title('eLife Peer Review LLM Usage Trend (CS distribution placeholder, pipeline test only)')

def to_percent(y, pos):
    return f"{y:.0f}%"
ax.yaxis.set_major_formatter(FuncFormatter(to_percent))

ax.grid(True, linestyle='--', alpha=0.4)
ax.legend(loc='upper left')
sns.despine(right=True, top=True)
plt.tight_layout()

out_path = f"{OUTPUT_DIR}/elife_pipeline_test.png"
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\n✅ 趋势图保存至: {out_path}")
plt.close()

# ============================================================
# 4. 保存结果表
# ============================================================
csv_path = f"{OUTPUT_DIR}/elife_alpha_results.csv"
df_r.to_csv(csv_path, index=False)
print(f"✅ 结果表保存至: {csv_path}")
print(f"\n结果概览:")
print(df_r[['year', 'month', 'alpha', 'ci', 'n_sentences']].to_string(index=False))
