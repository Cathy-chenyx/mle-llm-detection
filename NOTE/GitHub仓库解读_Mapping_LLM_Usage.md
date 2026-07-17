---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 3ee4bef11db4abe8e7f99813efcba22e_918786b5793111f1a7da5254006c9bbf
    ReservedCode1: Fb7xvYoH8JhET6LSlqtZVZvn8kGbEEATddbH2hdaivM5DVntHGXZgHAi51FdvCIOCDYDtNMVybM0YhU6sMa4PReHcgP8QwnxXVZWMrT1RPuDIOplEkkAjHmUpIXsaNqcgUoGAT4fQlXCZnIt5RF9sguoa1MQtaNDV6+KoYqXbT7SBG9i6P6L2CVX68g=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 3ee4bef11db4abe8e7f99813efcba22e_918786b5793111f1a7da5254006c9bbf
    ReservedCode2: Fb7xvYoH8JhET6LSlqtZVZvn8kGbEEATddbH2hdaivM5DVntHGXZgHAi51FdvCIOCDYDtNMVybM0YhU6sMa4PReHcgP8QwnxXVZWMrT1RPuDIOplEkkAjHmUpIXsaNqcgUoGAT4fQlXCZnIt5RF9sguoa1MQtaNDV6+KoYqXbT7SBG9i6P6L2CVX68g=
---

# GitHub仓库解读：Mapping the Increasing Use of LLMs in Scientific Papers

> **仓库地址**：https://github.com/Weixin-Liang/Mapping-the-Increasing-Use-of-LLMs-in-Scientific-Papers
> **作者**：Weixin Liang（斯坦福大学）
> **论文**：Liang et al., 2024 (ICML) — 就是你正在读的那篇
> **编写时间**：2026年7月

---

## 一、仓库是什么：一句话总结

这个仓库是论文 **"Monitoring AI-Modified Content at Scale"** 的**完整代码和数据的开源复现包**。你论文里看到的每一张图、每一个数字，都可以用这个仓库重新跑出来。

---

## 二、仓库结构：五句话读懂

```
Mapping-the-Increasing-Use-of-LLMs-in-Scientific-Papers/
│
├── src/MLE.py              ← 核心引擎：MLE 类（约 186 行）
├── distribution/           ← 预计算的词频分布（人类 vs AI）
├── data/inference_data/    ← 各期刊/会议的月度摘要数据（parquet格式）
├── data/validation_data/   ← 人工混合验证数据（已知 α 真值）
├── increasing_temporal.ipynb  ← 复现论文 Figure 1（时间趋势图）
├── validation.ipynb        ← 复现论文 Figure 3（方法验证）
├── tokenize_demo.ipynb     ← 演示：如何把原始文本变成模型能用的格式
└── environment.yml         ← Python 依赖配置（conda）
```

---

## 三、代码在做什么：逐文件解读

### 3.1 核心引擎：`src/MLE.py`

这是整个仓库的**心脏**，只有一个类 `MLE`，约 186 行代码。它做的事情跟我们之前学的统计理论完全对应：

| 方法 | 对应统计概念 | 做什么 |
|---|---|---|
| `__init__()` | 加载参考分布 | 读入预计算的词频 parquet 文件，存下 logP（人类）、logQ（AI）、log(1-P)、log(1-Q) |
| `precompute_log_probabilities()` | 计算句子对数概率 | 对每条句子分别算它在人类分布下和 AI 分布下的对数概率 |
| `optimized_log_likelihood()` | 极大似然估计 | 给定 α，算混合模型的负对数似然 $\ell(\alpha) = -\frac{1}{n}\sum \log[(1-\alpha) + \alpha \cdot e^{\log Q_i - \log P_i}]$ |
| `inference()` | 完整推断流程 | 读数据 → 预处理 → 调 `bootstrap_alpha_inference()` → 返回 α̂ 和 95% CI |
| `bootstrap_alpha_inference()` | Bootstrap 置信区间 | 1000 次重抽样 + 每次做 MLE，取 2.5% 和 97.5% 分位数 |

**关键细节**：论文中处理了"未登录词"（out-of-vocabulary words）——对于在参考分布中没有见过的词，赋予默认的对数概率 -13.8，相当于给它一个非常小的概率。

### 3.2 数据格式

数据以 Apache Parquet 格式存储（一种高效列式存储格式，比 CSV 快得多）。每条数据是一个句子，已经被分词成词汇列表：

```
inference_sentence
["This", "is", "an", "example"]
["Another", "sentence", "for", "you"]
```

Nature 文件夹里有 **2021年1月到2024年9月** 逐月的 parquet 文件（每月最多 2000 篇论文摘要）。论文发现 Nature 期刊的 α 始终很低（~1.6%），与 arXiv 上 AI 领域（CS 最高到 ~15%）形成鲜明对比。

### 3.3 三个 Notebook 的作用

| Notebook | 功能 | 对应论文 |
|---|---|---|
| `increasing_temporal.ipynb` | 遍历所有月份的 parquet 文件，对每个月份用 MLE 估计 α，然后画时间趋势折线图 | Figure 1 |
| `validation.ipynb` | 在已知 α 真值的人工混合数据上验证方法精度 | Figure 3 |
| `tokenize_demo.ipynb` | 演示如何用 spaCy 把原始文本分词成模型需要的格式 | 数据预处理 |

---

## 四、和你的 AI 项目的关系

### 4.1 直接关系：这就是论文的官方代码

你导师给你这篇论文，大概率就是让你基于这个框架开展工作。这个仓库是你项目的**"代码地基"**——你可以直接 clone 下来，在自己的数据上跑一遍，理解整个流程。

### 4.2 你可以做的延展方向

| 方向 | 具体做什么 | 对你的价值 |
|---|---|---|
| **方向 1：复现验证** | clone 仓库，跑通 `validation.ipynb`，验证你理解正确 | 建立对方法精度的第一手认知 |
| **方向 2：领域迁移** | 把这个框架搬到**生物医学论文摘要**或**临床试验报告**上 | 这是你最有可能做的项目方向——把方法应用到你自己熟悉的领域 |
| **方向 3：信号扩展** | 论文只用了形容词——你能不能加入**医学术语偏好**、**统计方法表述模式**等新信号？ | 结合你的生物统计背景，做差异化创新 |
| **方向 4：R 语言移植** | 把 Python 的 MLE 类用 R 重写一遍 | 加深对方法本身的理解，同时产出可复用的 R 包 |

### 4.3 重要提示：Nature 文件夹的意义

你给的链接指向 `data/inference_data/nature`。Nature 在论文中是**对照组**——它是综合性科学期刊（不是 AI 会议），作者用它来验证"非 AI 领域的论文摘要没有被 LLM 大规模渗透"。结果也确实如此（α ≈ 1.6%，无显著增长趋势）。这对你的启示是：**同样的方法你可以用到生物医学期刊上，看中国/国际生物医学论文在 ChatGPT 发布后有没有类似趋势。**

---

## 五、如何学习这个仓库

### 5.1 学习路径（从浅入深，约 3~5 天）

```
第 1 步（1 小时）：通读 README 和本文档
    ↓
第 2 步（2 小时）：读 src/MLE.py，和我之前的统计理论笔记对照着看
    ↓
第 3 步（2 小时）：clone 仓库，装好环境，跑通 validation.ipynb
    ↓
第 4 步（3 小时）：读 increasing_temporal.ipynb，理解逐月推断流程
    ↓
第 5 步（半天）：读 tokenize_demo.ipynb，理解数据预处理
    ↓
第 6 步（自由探索）：尝试在 Nature 数据上自己跑一次推断
```

### 5.2 第一步：读代码时对照理论

你在读 `src/MLE.py` 时，可以参考我上一份笔记中的对应章节：

| MLE.py 中的代码 | 对应笔记章节 |
|---|---|
| `optimized_log_likelihood()` | 第一部分 1.2 "数学定义" |
| 混合模型公式 `(1-α) + α·exp(logQ-logP)` | 第二部分 2.2 "数学形式" |
| `bootstrap_alpha_inference()` | 第三部分 3.3 "Bootstrap 置信区间" |

你会发现，**论文中的统计推导和仓库里的 Python 代码是一一对应的**。这不是巧合——代码就是数学公式的工程实现。

### 5.3 需要克服的技术门槛：Python

这个仓库是 Python 写的，而你主要用 R。这不是障碍，而是好机会——正好借这个机会入门 Python 的数据科学生态（numpy、pandas、scipy）。以下是快速上手的对照表：

| R 操作 | Python 等价 |
|---|---|
| `data.frame` | `pandas.DataFrame` |
| `read.csv()` | `pd.read_csv()` |
| `readRDS()` / `load()` | `pd.read_parquet()` |
| `optim()` | `scipy.optimize.minimize()` |
| `dplyr::mutate()` / `apply()` | `.apply()` 或 `pandas` 向量化操作 |
| `ggplot2` | `matplotlib` + `seaborn` |

如果你坚持只用 R 跑分析，也可以直接读 parquet 文件（R 的 `arrow` 包支持 `read_parquet()`），然后用 R 重写 MLE 逻辑。

---

## 六、如何在自己的电脑上使用

### 6.1 安装步骤

```bash
# 第 1 步：clone 仓库
cd /Users/cathy/Documents/学习相关/老段课题组/AI_project
git clone https://github.com/Weixin-Liang/Mapping-the-Increasing-Use-of-LLMs-in-Scientific-Papers.git
cd Mapping-the-Increasing-Use-of-LLMs-in-Scientific-Papers

# 第 2 步：创建 conda 环境（需要先装好 conda）
conda env create -f environment.yml
conda activate llm-detection
```

> **Mac M2 注意事项**：conda 环境下安装的 numpy/pandas/scipy 对 Apple Silicon 支持良好，不需要额外配置。

### 6.2 最小可用示例

创建环境后，在 Python 中运行：

```python
from src.MLE import MLE

# 加载计算机科学领域的词频分布
model = MLE("distribution/CS.parquet")

# 对验证数据做推断（已知 α 真值 = 0.10，验证方法精度）
estimated, ci = model.inference(
    "data/validation_data/CS/ground_truth_alpha_0.1.parquet",
    exploded_data=True
)

print(f"估计值: {estimated:.3f}, 真值: 0.100, 误差: {abs(estimated - 0.100):.3f}")
# 期望输出：误差 < 3%
```

### 6.3 在 Nature 数据上跑一次

```python
from src.MLE import MLE

# 加载 Nature 的词频分布
model = MLE("distribution/nature.parquet")

# 对 Nature 2023 年 6 月的数据做推断
alpha, ci = model.inference("data/inference_data/nature/2023_6.parquet")
print(f"Nature 2023年6月 α = {alpha:.3%}, 95% CI = [{ci[0]:.3%}, {ci[1]:.3%}]")
```

---

## 七、总结：三个关键收获

| 收获 | 具体内容 |
|---|---|
| **方法论** | 你学了 MLE + 混合模型的理论，现在看到了它们在真实研究中的代码实现 |
| **工具链** | 你知道了如何从原始文本 → 分词 → parquet → MLE推断 → 可视化这一整条管线 |
| **可迁移性** | 你可以把这个框架搬到生物医学领域——换一包数据、换一套词频分布，方法完全通用 |

---

> **给研0同学的建议**：这个仓库虽然只有约 186 行核心代码，但它是一个完整的、可发表的研究项目的缩影。clone 下来、跑通、读懂每一行——这件事本身就会让你对"如何做一个数据驱动的 AI 研究项目"有一个非常具体的认知。比读十篇综述都管用。
*（内容由AI生成，仅供参考）*
