---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 3ee4bef11db4abe8e7f99813efcba22e_e07b7d957f2611f1ae905254006c9bbf
    ReservedCode1: 72s8v+TySEDpNZCxISvNpBJR4OO5Z+ONBVDjCw/FVoDwrHTrAXP3vQ+qFdj5xHn+p1lt21je1V9PowWVPzc/9vov+kEaQOXv06lXUrUkGAARGEv2wqr4f0Prgf19Bnrfk6T1F7/EMo6RiF3tEhNdgEV3mjLNWlRumWAjHFNJv9onhDWQ5St/Tt/MSb8=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 3ee4bef11db4abe8e7f99813efcba22e_e07b7d957f2611f1ae905254006c9bbf
    ReservedCode2: 72s8v+TySEDpNZCxISvNpBJR4OO5Z+ONBVDjCw/FVoDwrHTrAXP3vQ+qFdj5xHn+p1lt21je1V9PowWVPzc/9vov+kEaQOXv06lXUrUkGAARGEv2wqr4f0Prgf19Bnrfk6T1F7/EMo6RiF3tEhNdgEV3mjLNWlRumWAjHFNJv9onhDWQ5St/Tt/MSb8=
---

# Pipeline 全程操作手册：LLM 使用率检测（Liang 2025 MLE 方法）

**技能名称**: `run-mle-pipeline`  
**适用范围**: 将医学/生命科学期刊审稿意见文本按月做 MLE 推断，估计 LLM 修改/生成比例 α。  
**前提**: 已完成 validation.ipynb 验证，conda 环境 `llm-detection` 可用。

---

## 一、环境与依赖

### 1.1 Conda 环境

```bash
conda activate llm-detection
# Python 3.8.19, pandas 2.0.3, numpy 1.24.4, scipy 1.10.1, swifter 1.4.0
```

### 1.2 核心模块（scripts/src/ 内嵌）

Pipeline 脚本已收束至 `scripts/` 目录，核心推断模块内嵌于其 `src/` 子目录，不再依赖外部仓库：

```
/Users/cathy/Documents/学习相关/老段课题组/AI_project/scripts/src/
├── MLE.py            ← MLE 推断核心：MLE 类构造函数加载分布词典，estimate_alpha() 做对数似然优化 + 1000 次 Bootstrap
├── estimation.py     ← 词频分布计算：二元计数 → 对数概率 → logP/logQ/log(1-P)/log(1-Q)
```

各脚本通过 `sys.path.insert(0, str(Path(__file__).parent))` 自动将 `scripts/` 加入搜索路径，因此 `from src.MLE import MLE` 等 import 在 `scripts/` 目录下均可正常工作。

### 1.3 API 配置（Q 语料生成用）

```
Base URL:  <INTERNAL_API_BASE_URL>
API Key:   <REDACTED_API_KEY>
协议格式:  Anthropic Messages API (anthropic SDK v0.72.0)
默认模型:  deepseek-v4-flash
```

---

## 二、输入数据格式

### 2.1 原始数据 CSV

**进入 Pipeline 的第一份数据**，由抓取/获取阶段产出。

| 必填字段 | 类型 | 说明 |
|----------|------|------|
| `text`（或等效字段名） | string | 审稿意见全文 |
| `year` | int | 年份 |
| `month` | int | 月份（1-12） |

**格式要求**: UTF-8 编码 CSV。其他字段（期刊名、文章 DOI 等）可选，但不会进入 Pipeline。

**样例**（`review_text_export/eLife_reviews_2021_2024.csv` 的实际结构）：
```
journal,year,month,review_content
eLife,2021,12,"This manuscript investigates..."
eLife,2021,12,"The authors present a novel..."
```

**⚠️ 注意**: 如果你的 CSV 列名不同（如 `review_content` 而非 `text`），必须在 `preprocess_*.py` 的映射字典里做对应调整。

### 2.2 数据量要求

| 数据 | 最低要求 | 理想值 |
|------|----------|--------|
| ChatGPT 前 H 语料（2022.10 之前） | ≥ 20 条 | ≥ 100 条 |
| 月度推断数据 | 每月 ≥ 5 句 | 每月 ≥ 50 句 |
| 时间跨度 | 至少覆盖 ChatGPT 前后各 1 年 | 2021-2024 全覆盖 |

**H 语料太少会导致**: 分布词典不可靠，α 估计方差大。

---

## 三、Pipeline 四步执行

### 步骤 0：数据预处理 → 产出月度 parquet + H 语料

**执行**: `python scripts/preprocess_XXX.py`（XXX = 期刊名，如 `elife`）

**输入**: 原始 CSV（含 review 文本 + 年月）

**处理过程**:
1. 清洗 HTML 标签、版权头部等杂质
2. spaCy `en_core_web_lg` 分句 + 正则分词（去除纯数字 token）
3. 按年月分组，写入 `{YEAR}_{MONTH}.parquet`
4. 筛选 ChatGPT 前（≤ 2022-10）的子集作为 H 语料

**输出文件**:

```
processed_data/{journal}/inference_data/
├── 2021_5.parquet
├── 2021_10.parquet
├── ...
└── 2024_12.parquet

processed_data/{journal}/human_corpus/
└── {journal}_human.parquet
```

**月度 parquet 列结构**:

| 列名 | 类型 | 说明 |
|------|------|------|
| `human_sentence` | list of list of str (numpy array | 每个元素是一个句子的 token 列表 |

**H 语料 parquet 列结构**:

| 列名 | 类型 | 说明 |
|------|------|------|
| `human_sentence` | list of list of str | 每条审稿意见的所有句子 token |

---

### 步骤 1：生成 Q 语料 → 产出 AI 参考语料

**执行**: `python scripts/generate_ai_corpus.py`

**输入**: `processed_data/{journal}/human_corpus/{journal}_human.parquet`（H 语料）

**处理过程**:
1. 对每条 H 文本截断到 8000 字符
2. **Stage 1**: 调用 LLM 提取事实要点 → 输出编号 bullet list
3. **Stage 2**: 调用 LLM 将 bullet list 扩写为完整审稿风格段落
4. 温度为 0.3（Stage 1）/ 0.7（Stage 2）
5. 每条间隔 0.5 秒避免限流

**配置修改项**: `generate_ai_corpus.py` 中的 `HUMAN_PATH` 和 `AI_OUT_DIR` 需指向目标期刊。

**输出文件**:

```
processed_data/{journal}/ai_corpus/
└── {journal}_ai.parquet
```

**Q 语料 parquet 列结构**:

| 列名 | 类型 | 说明 |
|------|------|------|
| `ai_review_text` | str | 完整 AI 生成的审稿意见段落文本 |

---

### 步骤 2：构建分布词典 → 产出 logP/logQ/log(1-P)/log(1-Q)

**执行**: `python scripts/build_distribution.py`

**输入**:
- H 语料: `processed_data/{journal}/human_corpus/{journal}_human.parquet`
- Q 语料: `processed_data/{journal}/ai_corpus/{journal}_ai.parquet`

**处理过程**:
1. 对 Q 语料做 spaCy 分词（与步骤 0 相同的 tokenize 逻辑）
2. 对齐 H 和 Q（取 min 条数）
3. 转为 Python list of lists（避免 parquet numpy object array 问题）
4. 调用 `estimation.py` 内层函数：
   - `count_human_binary_word_occurrences()` / `count_ai_binary_word_occurrences()`：二元计数（词在句中是否出现）
   - `estimate_log_probabilities()`：log(出现次数 / 总句数)
   - `filter_frequent_words()`：H 至少 5 次，Q 至少 3 次
   - `get_vocabulary_intersection()`：取交集
   - `calculate_log_probability()`：计算 logP, logQ, log(1-P), log(1-Q)

**配置修改项**: `build_distribution.py` 中的 `H_PATH`, `Q_RAW_PATH`, `DIST_OUT_DIR`。

**输出文件**:

```
processed_data/{journal}/distribution/
└── {journal}.parquet
```

**分布词典列结构**:

| 列名 | 类型 | 说明 |
|------|------|------|
| `Word` | str | 词典中的词 |
| `logP` | float | log(P)，P = 该词出现在人类句子中的概率 |
| `log1-P` | float | log(1-P) |
| `logQ` | float | log(Q)，Q = 该词出现在 AI 句子中的概率 |
| `log1-Q` | float | log(1-Q) |

**核心公式**: 混合模型 `P(w) = α·Q(w) + (1-α)·H(w)`，MLE 推断时用这四个对数概率计算对数似然。

---

### 步骤 3：MLE 推断 → 产出月度 α 序列 + 趋势图

**执行**: `python scripts/run_{journal}_pipeline.py`

**输入**:
- 分布词典: `processed_data/{journal}/distribution/{journal}.parquet`
- 推断数据: `processed_data/{journal}/inference_data/*.parquet`

**处理过程**:
1. `MLE(DIST_PATH)` 加载分布词典 → 构造函数内部做 `precompute_log_probabilities`（预计算 log(1-P)/log(1-Q)，未登录词默认 -13.8）
2. 遍历每个月度 parquet：
   - 提取所有句子
   - 对每个句子的 token 计算（α·Q + (1-α)·H）下的对数似然
   - scipy `minimize_scalar` 在 [0, 1] 区间搜索最优 α
   - 1000 次 Bootstrap 计算 95% CI
3. 汇总结果，matplotlib 绘制时间序列折线图 ± CI 着色

**配置修改项**:
- `DIST_PATH`: 分布词典路径
- `INFERENCE_DIR`: 月度 parquet 目录
- `OUTPUT_DIR`: 输出目录

**输出文件**:

```
processed_data/{journal}/
├── {journal}_alpha_results.csv
└── {journal}_pipeline_test.png
```

**结果 CSV 列结构**:

| 列名 | 类型 | 说明 |
|------|------|------|
| `year` | int | 年份 |
| `month` | int | 月份 |
| `alpha` | float | MLE 估计的 AI 贡献比例（0-1 之间） |
| `ci` | float | 95% Bootstrap 置信区间半宽 |
| `n_sentences` | int | 该月句子总数 |

**趋势图**: 横轴为时间（年月），纵轴为 α（百分比），着色区域为 95% CI。

---

## 四、完整执行清单（以新期刊为例）

假设新期刊数据 CSV 为 `review_text_export/JournalX_reviews_2021_2024.csv`。

### 前置检查

```bash
conda activate llm-detection
cd /Users/cathy/Documents/学习相关/老段课题组/AI_project
```

### [ ] 步骤 0：编写/修改 `preprocess_journalX.py`

1. 改 CSV 文件路径
2. 改列名映射（如原文列叫 `review_content` → 脚本内映射）
3. 改输出目录名 `journalX`
4. 执行：`python scripts/preprocess_journalX.py`

**验收**: `processed_data/journalX/inference_data/` 下有月度 parquet 文件。

### [ ] 步骤 1：修改 `generate_ai_corpus.py` 并执行

1. 改 `HUMAN_PATH` → `processed_data/journalX/human_corpus/journalX_human.parquet`
2. 改 `AI_OUT_DIR` → `processed_data/journalX/ai_corpus/`
3. 执行：`python scripts/generate_ai_corpus.py`

**验收**: Q 语料行数 = H 语料行数，无 "bullet outline was not provided" 错误。

### [ ] 步骤 2：修改 `build_distribution.py` 并执行

1. 改 `H_PATH` → H 语料路径
2. 改 `Q_RAW_PATH` → Q 语料路径
3. 改 `DIST_OUT_DIR` → `processed_data/journalX/distribution/`
4. 执行：`python scripts/build_distribution.py`

**验收**: 输出词典大小 > 500 词，区分度最高的词以过渡词为主（furthermore/additionally 等）。

### [ ] 步骤 3：编写/修改 `run_journalX_pipeline.py`

1. 改 `DIST_PATH`
2. 改 `INFERENCE_DIR`
3. 改 `OUTPUT_DIR`（或直接用前缀名）
4. 执行：`python scripts/run_journalX_pipeline.py`

**验收**: 产出 CSV 有 α 值，png 有趋势折线，2024 年 CI 不能过宽。

---

## 五、常见问题排查

| 问题 | 原因 | 解决 |
|------|------|------|
| `unhashable type: 'numpy.ndarray'` | parquet 回读时嵌套 list 转 numpy array | build_distribution.py 已修复（内存直传），若重现在 `to_list_of_lists()` 处加 `list(s)` 转换 |
| `bullet outline was not provided` | Stage 1 prompt 过长，模型误解 | 截断文本到 8000 字符，使用短 prompt 嵌入 user message |
| Stage 1 输出被截断 | max_tokens 不够 | 增大到 3000 |
| α = 0.0% 所有月份 | 分布跨域（如 CS 分布跑医学数据） | 确认用了正确的分布 parquet |
| CI 极宽（± 30%+） | 当月句子太少 | 正常，小样本结果应在报告中标注不可靠 |
| matplotlib 中文警告 | 无 CJK 字体 | 不影响英文图表，可忽略或用 `plt.rcParams['font.sans-serif']` 指定 |
| 生成 Q 语料 API 超时 | 自有端点不稳定 | 重试或切换模型（deepseek-v4-pro） |

---

## 六、输出物用途说明

| 文件 | 用途 |
|------|------|
| 月度 parquet | MLE 推断的输入；可跨项目复用 |
| H 语料 parquet | Q 语料生成 + 分布构建；是所有估计的基准 |
| Q 语料 parquet | 与 H 语料配对构建分布词典 |
| 分布词典 parquet | `MLE()` 构造函数加载的核心输入；决定哪些词纳入模型、logP/logQ 的值 |
| alpha_results.csv | Excel/Tableau 绘图、LaTeX 论文表格的直接数据源 |
| pipeline_test.png | 展示用趋势图；可嵌入 PPT 或论文 |
