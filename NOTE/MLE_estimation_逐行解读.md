---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 3ee4bef11db4abe8e7f99813efcba22e_c34b852d793511f1a7da5254006c9bbf
    ReservedCode1: lwOPRZe66qbcRegkfsUDeURK6nXutpQf5mEbrOwFMo2JWjQyvA7jevPLblBrA7/NZtc3VR/+JAHR60fW8pOaQZI+WwA1SK4gm6b0FPpRBib003n2BA4xFHNTTyB3JUnFfGGL4SiSRKCdLAdVlB5sgy3g+WvrhpMpa8UTr8TLVvhHb/XKzLmb02NKf/4=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 3ee4bef11db4abe8e7f99813efcba22e_c34b852d793511f1a7da5254006c9bbf
    ReservedCode2: lwOPRZe66qbcRegkfsUDeURK6nXutpQf5mEbrOwFMo2JWjQyvA7jevPLblBrA7/NZtc3VR/+JAHR60fW8pOaQZI+WwA1SK4gm6b0FPpRBib003n2BA4xFHNTTyB3JUnFfGGL4SiSRKCdLAdVlB5sgy3g+WvrhpMpa8UTr8TLVvhHb/XKzLmb02NKf/4=
---

# MLE.py 和 estimation.py 逐行解读

> **面向人群**：熟悉 R 语言、对 Python 不熟的生物统计专业学生
> **配套理论**：请对照阅读《统计理论基础_极大似然估计_混合模型_群体推断.md》
> **编写时间**：2026年7月

---

## 阅读前：Python vs R 速查表

先记住几个 Python 和 R 的关键差异，读代码时不会卡住：

| Python | R | 说明 |
|---|---|---|
| `import numpy as np` | `library(...)` | 导入包 |
| `def f(x):` | `f <- function(x) {}` | 定义函数，注意 Python 用冒号+缩进，没有花括号 |
| `[1, 2, 3]` | `c(1, 2, 3)` | 列表/向量 |
| `{a: 1, b: 2}` | `list(a=1, b=2)` | 字典（键值对） |
| `x = 5` | `x <- 5` | 赋值 |
| `len(x)` | `length(x)` | 长度 |
| `range(10)` | `1:10` | 生成整数序列 |
| `[i*2 for i in x]` | `sapply(x, function(i) i*2)` | 列表推导式 |
| `lambda x: x+1` | `function(x) x+1` | 匿名函数 |
| `True / False` | `TRUE / FALSE` | 布尔值，注意大小写 |
| `None` | `NULL` | 空值 |
| `.apply(func)` | `sapply()` / `lapply()` | 对每行/列应用函数 |
| `self.xxx` | `xxx`（在引用列表元素时） | 类的成员变量 |

---

---

# 第一部分：estimation.py 逐行解读

> 这个文件的作用是：**从原始的人类文本和 AI 文本中，估计每个词在两类文本里出现的概率分布**。输出是一个 parquet 文件（即 `distribution/*.parquet`），供 MLE.py 使用。

---

## 0. 文件开头：导入依赖

```python
import pandas as pd        # 数据处理，相当于 R 的 data.frame + dplyr
import numpy as np         # 数值计算，相当于 R 的基础数学函数
from collections import Counter  # 计数器，相当于 R 的 table()
```

**Python 知识点**：`import X as Y` 就是给包起一个短别名，相当于 R 的 `library(X)` 之后你可以直接用函数名。但 Python 必须写 `pd.xxx` 或 `np.xxx` 前缀。

---

## 1. `get_vocabulary_intersection()`

```python
def get_vocabulary_intersection(human_counts, ai_counts):
    return set(human_counts.keys()).intersection(ai_counts.keys())
```

### 做什么？
取两个词频字典的**交集**——只保留既在人类文本中出现过、又在 AI 文本中出现过的词。

### 逐行拆解
- `human_counts.keys()`：取字典的所有键（即所有单词），相当于 R 的 `names(human_counts)`
- `set(...)`：转为集合（集合是无序、不重复的），因为接下来要做集合运算
- `.intersection(...)`：取两个集合的交集，相当于 R 的 `intersect()`

### R 等价写法
```r
get_vocabulary_intersection <- function(human_counts, ai_counts) {
  intersect(names(human_counts), names(ai_counts))
}
```

---

## 2. `filter_frequent_words()`

```python
def filter_frequent_words(word_counts, min_occurrences):
    return {word: count for word, count in word_counts.items() if count >= min_occurrences}
```

### 做什么？
过滤低频词——只保留出现次数 ≥ `min_occurrences` 的词。

### 逐行拆解
- `word_counts.items()`：把字典拆成 `(词, 次数)` 的键值对列表
- `{word: count for word, count in ... if count >= min_occurrences}`：**字典推导式**——遍历每个键值对，满足条件就保留。相当于 R 的 `word_counts[word_counts >= min_occurrences]`

### R 等价写法
```r
filter_frequent_words <- function(word_counts, min_occurrences) {
  word_counts[word_counts >= min_occurrences]
}
```

---

## 3. `count_human_binary_word_occurrences()`

```python
def count_human_binary_word_occurrences(human_data):
    word_counts = Counter(word for sent in human_data['human_sentence'] for word in set(sent))
    return dict(word_counts)
```

### 做什么？
统计每个词在**多少句**人类文本中出现过（二元计数：同一句中多次出现也只算 1 次）。

> 问：为什么是二元（binary）而非频率？答：因为论文的方法是基于"这个词至少出现一次的概率 P(w)"来建模，不是基于 TF-IDF 那种频率加权。这样更稳健，不受单句中重复使用同一个词的影响。

### 逐行拆解
- `human_data['human_sentence']`：取数据框中列名为 `human_sentence` 的列，这是一个 Series（类似 R 的数据框单列），每行是一个句子（词列表）
- `for sent in human_data['human_sentence']`：外层循环，遍历每个句子
- `for word in set(sent)`：内层循环，遍历这个句子中**去重后**的词（`set()` 去重，保证同一句中一个词只计一次）
- `Counter(...)`：统计每个词出现了多少次（多少个句子中含有该词），相当于 R 的 `table()`
- `dict(word_counts)`：Counter 本质是字典的子类，这里显式转成普通字典

### R 等价写法
```r
count_binary_word_occurrences <- function(data, col_name) {
  result <- integer(0)
  for (sent in data[[col_name]]) {
    unique_words <- unique(sent)
    for (word in unique_words) {
      result[word] <- (result[word] %||% 0) + 1
    }
  }
  result
}
```
> Python 的写法比 R 简洁很多——这是 Python 在数据处理上的一个优势。

---

## 4. `count_ai_binary_word_occurrences()`

与第 3 节完全对称，只是列名从 `human_sentence` 变成 `ai_sentence`，逻辑一模一样。

---

## 5. `estimate_log_probabilities()`

```python
def estimate_log_probabilities(word_counts, total_sents):
    log_probabilities = {word: np.log(count / total_sents) for word, count in word_counts.items()}
    return log_probabilities
```

### 做什么？
把"某词在多少句中出现"转成**对数概率**。

数学上：$P(w) = \dfrac{\text{含有 w 的句子数}}{\text{总句子数}}$，然后取 $\log$。

### 逐行拆解
- `count / total_sents`：出现概率的估计
- `np.log(...)`：自然对数。Python 里 `np.log` = R 的 `log`（都是以 e 为底）
- `{word: ..., for ...}`：字典推导式，构建 `{词: 对数概率}` 的映射

### R 等价写法
```r
estimate_log_probabilities <- function(word_counts, total_sents) {
  sapply(word_counts, function(count) log(count / total_sents))
}
```

---

## 6. `calculate_log_probability()` — 核心函数

这是 estimation.py 中**最长、最关键的**函数。它把前面几个函数的结果整合起来，计算每个词的完整概率分布。

```python
def calculate_log_probability(human_probs, ai_probs, common_vocab):
    data = []
    for word in common_vocab:
        log_human_prob = human_probs.get(word, -np.inf)
        log_ai_prob = ai_probs.get(word, -np.inf)
        
        log_one_minus_human_prob = np.log1p(-np.exp(log_human_prob))
        log_one_minus_ai_prob = np.log1p(-np.exp(log_ai_prob))
          
        human_log_odds = log_human_prob - log_one_minus_human_prob
        ai_log_odds = log_ai_prob - log_one_minus_ai_prob
        log_odds_ratio = human_log_odds - ai_log_odds
        
        if np.isinf(log_odds_ratio) or np.isnan(log_odds_ratio):
            continue

        data.append({"Word": word,  
                     "logP": log_human_prob,
                     'log1-P': log_one_minus_human_prob,
                     "logQ": log_ai_prob,
                     'log1-Q': log_one_minus_ai_prob,
                     "Log Odds Ratio": log_odds_ratio})
    
    df = pd.DataFrame(data)
    df = df.sort_values(by='Log Odds Ratio', ascending=True)
    df.reset_index(drop=True, inplace=True)
    df = df.drop(columns=['Log Odds Ratio'])
    return df
```

### 做什么？
为每个共同词汇计算五个值，按对数优势比（Log Odds Ratio）排序后输出。最终输出的列是：`Word`, `logP`, `log1-P`, `logQ`, `log1-Q`。

其中：
- **logP**：人类文本中出现该词的对数概率
- **logQ**：AI 文本中出现该词的对数概率
- **log1-P**：人类文本中**不**出现该词的对数概率
- **log1-Q**：AI 文本中**不**出现该词的对数概率

> "Log Odds Ratio" 只用于排序（按"这个词在人类中相对于 AI 的优势"从小排到大），排完后被删掉，不进入最终输出。

### 逐行拆解

| 行 | 代码 | 解释 |
|---|---|---|
| 1 | `data = []` | 初始化空列表，准备装字典 |
| 2 | `for word in common_vocab:` | 遍历每个共同词 |
| 3 | `log_human_prob = human_probs.get(word, -np.inf)` | 从人类概率字典中取该词的对数概率；如果没有，默认 `-∞` |
| 4 | `log_ai_prob = ai_probs.get(word, -np.inf)` | 同上，从 AI 概率字典中取 |
| 6 | `np.log1p(-np.exp(log_human_prob))` | 关键操作：已知 log(p)，求 log(1-p)。公式：$\log(1-p) = \log(1 - e^{\log p})$。`log1p(x)` = $\log(1+x)$，比直接算 `log(1 - exp(...))` 更数值稳定 |
| 9 | `human_log_odds` | $\log\frac{p}{1-p}$，即对数优势 |
| 10 | `ai_log_odds` | 同上 |
| 11 | `log_odds_ratio` | $\log\frac{p/(1-p)}{q/(1-q)}$。正值 = 人类更可能用该词；负值 = AI 更可能用该词 |
| 13 | `if np.isinf(...) or np.isnan(...):` | 如果对数优势比是无穷大或 NaN（通常因为 p 或 q 是 0 或 1），跳过这个词 |
| 17 | `data.append({...})` | 把该词的所有信息打包成字典，加入列表 |
| 24 | `pd.DataFrame(data)` | 列表 → 数据框 |
| 25 | `df.sort_values(...)` | 按 Log Odds Ratio 升序排列。越靠前的词，AI 越倾向于使用（相对于人类） |
| 26 | `reset_index(drop=True)` | 重排行号，`drop=True` 表示不保留旧行号。相当于 R 的 `rownames(df) <- NULL` |
| 27 | `df.drop(columns=['Log Odds Ratio'])` | 删除排序列，只保留最终需要的五列 |

### Python 知识点

- **`.get(key, default)`**：字典方法。如果 key 存在，返回对应值；如果不存在，返回 `default`。相当于 R 里先 `if (word %in% names(dict))` 再取值。
- **`np.log1p(x)`**：计算 `log(1+x)`。当 x 很小时比直接算更精确。这里 `x = -exp(log_p)`，所以实际算的是 `log(1 - exp(log_p))` = `log(1-p)`。
- **`np.isinf()` / `np.isnan()`**：判断是否无穷大 / 非数值。
- **`.append()`**：列表方法，在末尾添加元素。
- **`pd.DataFrame(data)`**：Python 数据框构造函数。当 `data` 是字典列表时，每个字典变成一行，键变成列名。
- **`.sort_values(by=..., ascending=...)`**：数据框排序。
- **`.drop(columns=[...])`**：删除列。

---

## 7. `estimate_text_distribution()` — 总调度函数

```python
def estimate_text_distribution(human_source_path, ai_source_path, save_file_path="Word.parquet"):
    human_data = pd.read_parquet(human_source_path)
    ai_data = pd.read_parquet(ai_source_path)
    
    if 'human_sentence' not in human_data.columns:
        raise ValueError("human_sentence column not found in human data")
    if 'ai_sentence' not in ai_data.columns:
        raise ValueError("ai_sentence column not found in ai data")

    human_data = human_data[human_data['human_sentence'].apply(len) > 1]
    ai_data = ai_data[ai_data['ai_sentence'].apply(len) > 1]
    human_data.dropna(subset=['human_sentence'], inplace=True)
    ai_data.dropna(subset=['ai_sentence'], inplace=True)
    
    human_word_counts = count_human_binary_word_occurrences(human_data)
    ai_word_counts = count_ai_binary_word_occurrences(ai_data)
    
    total_human_sentences = len(human_data)
    total_ai_sentences = len(ai_data)
    
    human_log_probs = estimate_log_probabilities(human_word_counts, total_human_sentences)
    ai_log_probs = estimate_log_probabilities(ai_word_counts, total_ai_sentences)
    
    common_vocab = get_vocabulary_intersection(human_word_counts, ai_word_counts)
    frequent_human_words = filter_frequent_words(human_word_counts, 5)
    frequent_ai_words = filter_frequent_words(ai_word_counts, 3)
    frequent_common_vocab = common_vocab.intersection(frequent_human_words.keys(), frequent_ai_words.keys())

    log_likelihood_df = calculate_log_probability(human_log_probs, ai_log_probs, frequent_common_vocab)
    log_likelihood_df.to_parquet(save_file_path, index=False)
```

### 做什么？
把前面的所有函数串起来的**总调度函数**。输入两个 parquet 文件（人类文本 + AI 文本），输出一个 parquet 文件（词频分布）。

### 处理流程（七步）

```
人类 parquet → 读入 → 清洗 → 统计词频 → 估计对数概率 ┐
                                                      ├→ 取交集 → 过滤低频 → 算完整分布 → 保存
AI parquet   → 读入 → 清洗 → 统计词频 → 估计对数概率 ┘
```

### 逐行拆解

| 步骤 | 行 | 解释 |
|---|---|---|
| 读入 | `pd.read_parquet(...)` | 读 parquet 文件。Parquet 是一种高效的列式存储格式，比 CSV 快很多。R 用户用 `arrow::read_parquet()` 对应 |
| 校验 | `if 'human_sentence' not in ...` | 检查必需的列是否存在。`raise ValueError(...)` 相当于 R 的 `stop(...)` |
| 清洗 | `.apply(len) > 1` | 过滤掉长度 ≤ 1 的词列表（太短的句子没有分析价值） |
| 清洗 | `.dropna(...)` | 删除缺失值（NaN = Not a Number） |
| 计数 | `count_human_binary_word_occurrences(...)` | 调用第 3 节函数，得到每个词在多少句中出现过 |
| 句子数 | `len(human_data)` | 数据框的行数 = 句子总数 |
| 对数概率 | `estimate_log_probabilities(...)` | 调用第 5 节函数，count → log-probability |
| 交集 | `get_vocabulary_intersection(...)` | 调用第 1 节函数，只保留两类文本中都出现的词 |
| 低频过滤 | `filter_frequent_words(..., 5)` | 人类文本：只保留出现在 ≥5 句里的词 |
| 低频过滤 | `filter_frequent_words(..., 3)` | AI 文本：只保留出现在 ≥3 句里的词 |
| 三维交集 | `.intersection(A, B, C)` | 同时满足：在共同词汇中、在人类高频词中、在 AI 高频词中 |
| 算分布 | `calculate_log_probability(...)` | 调用第 6 节函数，算 logP, logQ, log(1-P), log(1-Q) |
| 保存 | `.to_parquet(..., index=False)` | 存为 parquet。`index=False` 表示不保存行号 |

### Python 知识点

- **`raise ValueError(...)`**：抛出异常，停止程序。相当于 R 的 `stop()`
- **`inplace=True`**：直接在原数据框上修改，不返回新对象
- **`.apply(len)`**：对 Series 的每个元素应用 `len()` 函数
- **`.intersection(A, B, C)`**：Python 集合的 `intersection()` 方法**可以接受多个参数**，一次性取多个集合的交集

---

---

# 第二部分：MLE.py 逐行解读

> 这个文件是核心推理引擎。它用 estimation.py 产出的词频分布文件，对任何新的文本语料库估计 AI 生成比例 α。

---

## 0. 文件开头：导入依赖

```python
import numpy as np              # 数值计算
import pandas as pd             # 数据处理
from scipy.optimize import minimize   # 数值优化（找极小值），相当于 R 的 optim()
import swifter                  # pandas 的加速器，让 .apply() 并行运行
```

---

## 1. `class MLE():` — 类定义

```python
class MLE():
```

### Python 知识点：什么是 class（类）？

你之前可能没用过 Python 的面向对象编程。简单理解：

- **类（class）** 是一个"蓝图"或"模板"
- **实例（instance）** 是用蓝图造出来的具体对象
- **方法（method）** 是类里面定义的函数，调用时用 `对象.方法()`
- **`self`** 指代"这个实例自己"，类似 R 的 S3/S4 对象中的 `self` 概念

举个例子：

```python
model = MLE("distribution/CS.parquet")   # 用蓝图造一个具体的模型实例
# 相当于 R 的: model <- MLE$new("distribution/CS.parquet")

alpha, ci = model.inference("data.parquet")   # 调用实例的方法
```

---

## 2. `__init__()` — 构造函数（初始化）

```python
def __init__(self, word_df_path):
    df = pd.read_parquet(word_df_path)
    word_df = df.copy()
    self.all_tokens_set = set(word_df['Word'].tolist())
    self.log_p_hat = {row['Word']: row['logP'] for index, row in word_df.iterrows()}
    self.log_q_hat = {row['Word']: row['logQ'] for index, row in word_df.iterrows()}
    self.log_one_minus_p_hat = {row['Word']: row['log1-P'] for index, row in word_df.iterrows()}
    self.log_one_minus_q_hat = {row['Word']: row['log1-Q'] for index, row in word_df.iterrows()}
```

### 做什么？
创建 MLE 实例时自动执行。读入 estimation.py 产出的 parquet 文件，把里面的数据拆成四个快速查询的字典。

### 逐行拆解

| 行 | 代码 | 解释 |
|---|---|---|
| 1 | `df = pd.read_parquet(word_df_path)` | 读取 parquet 文件（包含 Word, logP, logQ, log1-P, log1-Q 五列） |
| 2 | `word_df = df.copy()` | 复制一份，避免后续操作影响原始数据 |
| 3 | `self.all_tokens_set = set(word_df['Word'].tolist())` | 提取所有词的集合，用于后面判断"一个词是不是在词典里" |
| 4 | `self.log_p_hat = {row['Word']: row['logP'] for ...}` | 构建字典：`{"commendable": -4.6, "meticulous": -3.9, ...}`。遍历每一行，取 Word 和 logP。这个字典存的是：**每个词在人类文本中出现的对数概率** |
| 5 | `self.log_q_hat = {...}` | 同上，存 AI 文本中的对数概率 |
| 6 | `self.log_one_minus_p_hat = {...}` | 存人类文本中**不出现**的对数概率 |
| 7 | `self.log_one_minus_q_hat = {...}` | 存 AI 文本中不出现的对数概率 |

> **这四本"词典"是整个方法的基础。** 它们存储了"人类 vs AI"在每一个词上的用词差异。

### Python 知识点

- **`__init__`**：双下划线开头和结尾的是 Python 的特殊方法（"魔术方法"）。`__init__` 是构造函数，在 `MLE(...)` 时自动调用。
- **`self.xxx`**：`self` 代表"这个实例"。赋给 `self.xxx` 的变量成为**实例属性**，在类的其他方法中都能通过 `self.xxx` 访问。
- **`.iterrows()`**：逐行遍历数据框，每次返回 `(行号, 行数据)`。比较慢，但因为只在初始化时用一次，所以无所谓。

---

## 3. `optimized_log_likelihood()` — 对数似然函数

```python
def optimized_log_likelihood(self, alpha, log_p_values, log_q_values):
    alpha = alpha[0]
    ll = np.mean(np.log((1 - alpha) + alpha * np.exp(log_q_values - log_p_values)))
    return -ll
```

### 做什么？
计算给定 α 下混合模型的**负对数似然**。`scipy.optimize.minimize` 会反复调用这个函数，尝试不同的 α 值，找到使返回值最小的 α。

### 数学推导（关键的几步）

对于第 i 条句子，它在人类分布下的对数概率为 $\log P_i$，在 AI 分布下的对数概率为 $\log Q_i$。

在混合模型下，这条句子的概率是：
$$P(\text{sentence}_i) = (1-\alpha) \cdot e^{\log P_i} + \alpha \cdot e^{\log Q_i}$$

取 log：
$$\log P(\text{sentence}_i) = \log\left[(1-\alpha) \cdot e^{\log P_i} + \alpha \cdot e^{\log Q_i}\right]$$
$$= \log P_i + \log\left[(1-\alpha) + \alpha \cdot e^{\log Q_i - \log P_i}\right]$$

平均对数似然：
$$\ell(\alpha) = \frac{1}{n}\sum_{i=1}^n \left[\log P_i + \log\left((1-\alpha) + \alpha \cdot e^{\log Q_i - \log P_i}\right)\right]$$

因为 $\log P_i$ 是常数（不随 α 变化），最大化 $\ell(\alpha)$ 等价于最大化：
$$\frac{1}{n}\sum \log\left((1-\alpha) + \alpha \cdot e^{\log Q_i - \log P_i}\right)$$

代码里的 `np.mean(np.log((1-alpha) + alpha * np.exp(log_q_values - log_p_values)))` 正是这个。返回值取负号是因为 `minimize` 默认找**最小值**，而我们要找**最大**似然——取负号把最大化问题转化成最小化问题。

### 逐行拆解

| 行 | 代码 | 解释 |
|---|---|---|
| 1 | `alpha = alpha[0]` | `minimize` 传入的参数是列表 `[α]`（即使只有一个元素），所以取第一个元素得到标量 α |
| 2 | `np.exp(log_q_values - log_p_values)` | 计算每个句子的 $e^{\log Q_i - \log P_i}$，即 $\frac{Q_i}{P_i}$（似然比） |
| 3 | `(1 - alpha) + alpha * ...` | 混合模型的核心公式 |
| 4 | `np.log(...)` | 取对数 |
| 5 | `np.mean(...)` | 对所有句子取平均 |
| 6 | `return -ll` | 返回负值（因为 `minimize` 找最小值） |

### Python 知识点

- **`alpha[0]`**：列表索引。Python 索引从 0 开始，`[0]` 取第一个元素。
- **`np.exp(x)`**：$e^x$，相当于 R 的 `exp(x)`。
- **`np.log(x)`**：自然对数，相当于 R 的 `log(x)`。
- **`np.mean(x)`**：算术平均，相当于 R 的 `mean(x)`。

---

## 4. `precompute_log_probabilities()` — 预计算每个句子的对数概率

```python
def precompute_log_probabilities(self, data):
    total_log_one_minus_p = sum(self.log_one_minus_p_hat.values())
    total_log_one_minus_q = sum(self.log_one_minus_q_hat.values())
    
    log_p_values = data.swifter.progress_bar(False).apply(
        lambda x: sum(self.log_p_hat.get(t, -13.8) for t in x) +
                 (total_log_one_minus_p - sum(self.log_one_minus_p_hat[t] for t in x if t in self.all_tokens_set))
    )
    
    log_q_values = data.swifter.progress_bar(False).apply(
        lambda x: sum(self.log_q_hat.get(t, -13.8) for t in x) +
                 (total_log_one_minus_q - sum(self.log_one_minus_q_hat[t] for t in x if t in self.all_tokens_set))
    )
    
    return np.array(log_p_values), np.array(log_q_values)
```

### 做什么？
输入一个句子列表（每条句子是一个词的集合），输出两个数组：
1. 每条句子在**人类分布**下的对数概率 logP
2. 每条句子在 **AI 分布**下的对数概率 logQ

### 核心数学

一条句子 s 里含有的词是 {w₁, w₂, ...}，不在句子里的词是 {..., 其他所有词}。

在给定分布下，这条句子的概率 = "句中出现的词都出现了"且"句中没出现的词都没出现"的联合概率。

对数形式：
$$\log P(\text{sentence}) = \underbrace{\sum_{w \in \text{sentence}} \log P(w)}_{\text{出现的词}} + \underbrace{\sum_{w \notin \text{sentence}} \log(1-P(w))}_{\text{没出现的词}}$$

第二项不好直接算（因为"没出现的词"太多了）。但可以变形：
$$\sum_{w \notin \text{sentence}} \log(1-P(w)) = \underbrace{\sum_{w \in \text{所有词}} \log(1-P(w))}_{\text{可以预计算的总和}} - \underbrace{\sum_{w \in \text{sentence}} \log(1-P(w))}_{\text{句中出现的词不用减}}$$

这就是代码中 `total_log_one_minus_p - sum(self.log_one_minus_p_hat[t] for t in x if t in self.all_tokens_set)` 的含义。

### 逐行拆解

| 行 | 代码 | 解释 |
|---|---|---|
| 1 | `total_log_one_minus_p = sum(self.log_one_minus_p_hat.values())` | 所有词的 log(1-P) 之和（预计算，对所有句子都一样） |
| 2 | `total_log_one_minus_q = ...` | 所有词的 log(1-Q) 之和 |
| 3 | `data.swifter.progress_bar(False).apply(lambda x: ...)` | 对 data 的每一行（每个句子）应用后面的 lambda 函数。`swifter.progress_bar(False)` 关闭进度条，`.apply()` 是并行化的 apply |
| 4 | `sum(self.log_p_hat.get(t, -13.8) for t in x)` | 对句子 x 中每个词 t，查它在人类分布中的 logP。如果 t 不在词典里（罕见词），给一个默认值 **-13.8** |
| 5 | `total_log_one_minus_p - sum(self.log_one_minus_p_hat[t] for t in x if t in self.all_tokens_set)` | 所有词的不出现概率总和 减去 句中词的不出现概率 = 句中未出现词的不出现概率总和 |
| 8 | `np.array(log_p_values)` | 转为 numpy 数组（高效数值运算的前提） |

### Python 知识点

- **`-13.8` 的含义**：$\log(10^{-6}) \approx -13.8$。即对于未登录词，假设它出现的概率是百万分之一。这是一种常见的平滑策略。
- **`lambda x: ...`**：匿名函数。`lambda x: x+1` 相当于 R 的 `function(x) x+1`。
- **`.swifter.progress_bar(False).apply(...)`**：swifter 是 pandas 的加速器，自动判断用 `apply`（串行）还是并行处理。对大数据集显著提速。
- **`sum(... for t in x)`**：生成器表达式。在 `sum()` 内部直接写循环，不需要先创建列表，更省内存。相当于 `sum(sapply(x, function(t) ...))` 但更高效。

---

## 5. `bootstrap_alpha_inference()` — Bootstrap MLE

```python
def bootstrap_alpha_inference(self, data, n_bootstrap=1000):
    full_log_p_values, full_log_q_values = self.precompute_log_probabilities(data)
    alpha_values_bootstrap = []
    for i in range(n_bootstrap):
        sample_indices = np.random.choice(len(data), size=len(data), replace=True)
        sample_log_p_values = full_log_p_values[sample_indices]
        sample_log_q_values = full_log_q_values[sample_indices]
        result = minimize(self.optimized_log_likelihood, x0=[0.5],
                         args=(sample_log_p_values, sample_log_q_values),
                         method='L-BFGS-B', bounds=[(0, 1)])
        if result.success:
            min_loss_alpha = result.x[0]
            alpha_values_bootstrap.append(min_loss_alpha)
    return np.percentile(alpha_values_bootstrap, [2.5, 97.5])
```

### 做什么？
用 Bootstrap 方法估计 α 的 95% 置信区间。步骤：

1. 对全部数据预计算一次 logP 和 logQ（所有 bootstrap 轮次共享）
2. 重复 1000 次：
   - 从原始数据中有放回地抽样
   - 取对应的 logP 和 logQ 值
   - 用 `minimize` 做 MLE，找到最优 α
3. 取 1000 个 α 估计值的 2.5% 和 97.5% 分位数作為 95% CI

### 逐行拆解

| 行 | 代码 | 解释 |
|---|---|---|
| 1 | `full_log_p_values, full_log_q_values = ...` | 预计算所有句子的对数概率，只做一次 |
| 2 | `alpha_values_bootstrap = []` | 初始化空列表，存储 1000 次 Bootstrap 的 α 估计值 |
| 3 | `for i in range(n_bootstrap):` | `range(1000)` 生成 0, 1, 2, ..., 999 |
| 4 | `np.random.choice(len(data), size=len(data), replace=True)` | 有放回地抽 `len(data)` 个索引。`replace=True` = 有放回（Bootstrap 的核心） |
| 5 | `full_log_p_values[sample_indices]` | 用索引数组取对应元素（numpy 的"花式索引"） |
| 7 | `minimize(...)` | 数值优化函数。几个参数：`x0=[0.5]` 初始猜测 α=0.5；`args=(...)` 传给目标函数的额外参数；`method='L-BFGS-B'` 使用带边界约束的拟牛顿法；`bounds=[(0,1)]` 限制 α∈[0,1] |
| 9 | `if result.success:` | 检查优化是否收敛。如果不收敛，跳过这轮 |
| 10 | `result.x[0]` | 最优 α 值 |
| 13 | `np.percentile(..., [2.5, 97.5])` | 取分位数，相当于 R 的 `quantile(x, c(0.025, 0.975))` |

### Python 知识点

- **`np.random.choice(n, size=m, replace=True)`**：从 0 到 n-1 中随机抽 m 个数。`replace=True` = 有放回。相当于 R 的 `sample(1:n, m, replace=TRUE)`。
- **`minimize(fn, x0, args, method, bounds)`**：`scipy.optimize.minimize`，相当于 R 的 `optim(par, fn, method, lower, upper)`。`L-BFGS-B` 算法支持参数边界约束。
- **`result.x`**：优化结果中，`.x` 属性存储最优参数值。

---

## 6. `inference()` — 对外主接口

```python
def inference(self, inference_file_path, exploded_data=False):
    inference_data = pd.read_parquet(inference_file_path)
    if 'inference_sentence' not in inference_data.columns:
        raise ValueError("inference_sentence column not found in inference data")
    if not exploded_data:
        inference_data = inference_data.explode('inference_sentence')
        inference_data.dropna(subset=['inference_sentence'], inplace=True)
    inference_data = inference_data[inference_data['inference_sentence'].apply(len) > 1]
    inference_data.dropna(subset=['inference_sentence'], inplace=True)
    inference_data.reset_index(drop=True, inplace=True)
    data = inference_data['inference_sentence'].swifter.progress_bar(False).apply(
        lambda x: set(token for token in x if token in self.all_tokens_set))
    confidence_interval = self.bootstrap_alpha_inference(data)
    solution = round(np.mean(confidence_interval), 3)
    half_width = (confidence_interval[1] - confidence_interval[0]) / 2
    half_width = round(half_width, 3)
    return solution, half_width
```

### 做什么？
这是用户直接调用的主接口。输入一个 parquet 文件路径，输出 (α̂, CI半宽) 两个数。

### 逐行拆解

| 行 | 代码 | 解释 |
|---|---|---|
| 1 | `pd.read_parquet(...)` | 读入待推断的文本数据 |
| 2-3 | `if ... raise ValueError(...)` | 校验必须有 `inference_sentence` 列 |
| 4-6 | `if not exploded_data: ... .explode()` | `explode()` 把嵌套的句子列表展开成多行。例如一行 `[["word1","word2"], ["word3"]]` 变成两行 `["word1","word2"]` 和 `["word3"]` |
| 7 | `.apply(len) > 1` | 过滤掉只有一个词的句子 |
| 8 | `.dropna(...)` | 删除缺失值 |
| 9 | `.reset_index(drop=True)` | 重排行号 |
| 10-11 | `.swifter...apply(lambda x: set(...))` | 对每个句子，只保留在词典中的词，然后去重转集合 |
| 12 | `self.bootstrap_alpha_inference(data)` | 调用第 5 节函数，返回 [CI下界, CI上界] |
| 13 | `solution = round(np.mean(confidence_interval), 3)` | **α 的点估计 = CI 的中点**。这是一种稳健做法——直接用 bootstrap 分布的中心位置作为点估计 |
| 14-15 | `(CI[1] - CI[0]) / 2` | CI 半宽。如果 CI 是 [0.088, 0.124]，半宽 = (0.124-0.088)/2 = 0.018 |
| 16 | `return solution, half_width` | 返回 (α̂, half_width)。用户拿到后可以自己写成 α̂ ± half_width |

### Python 知识点

- **`.explode(col)`**：把嵌套列表列展开。Pandas 1.3+ 支持。相当于 R `tidyr::unnest()`。
- **`round(x, 3)`**：四舍五入保留 3 位小数。相当于 R 的 `round(x, 3)`。
- **`np.mean(confidence_interval)`**：取 CI 均值 = (下界+上界)/2。

### 为什么用 CI 中点代替代点估计？

传统的 MLE 是直接返回使似然最大的 α̂。这里的做法更稳健——先用 Bootstrap 得到 CI，然后取 CI 中点作为点估计。两种方式在大样本下等价，但这种方式天然保证了点估计和区间估计的一致性。

---

---

## 第三部分：两张图总结两个文件的关系

```
estimation.py                           MLE.py
─────────────                           ──────
                                          
人类 parquet ─┐                         inference parquet ─┐
              ├→ estimate_text_         MLE.__init__() ←──┘ 读取 distribution/*.parquet
AI parquet ──┘   distribution()         MLE.inference()
                   ↓                        ↓
              distribution/*.parquet    (α̂, half_width)
              (Word, logP, logQ,        例如：(0.106, 0.018)
               log1-P, log1-Q)          即 α = 10.6% ± 1.8%
```

---

## 第四部分：带着统计眼光看代码

读完全部代码后，回顾我们学的三个统计概念：

| 统计概念 | 在代码中的位置 |
|---|---|
| **极大似然估计** | `optimized_log_likelihood()` — 定义似然函数；`minimize(..., method='L-BFGS-B')` — 数值求解 |
| **混合分布模型** | `optimized_log_likelihood()` 中的 `(1-alpha) + alpha * exp(logQ - logP)` — 两分量混合 |
| **群体推断** | 整个 `MLE` 类的设计就是群体推断——输入一个语料库，输出一个 α，不判断单篇 |
| **Bootstrap 置信区间** | `bootstrap_alpha_inference()` — 1000 次重抽样 + 百分位法求 CI |

---

> **给 R 用户的建议**：如果你想加深理解，最好的练习是用 R 重写 MLE.py 的核心逻辑。你会发现大部分代码可以直接用 base R + `optim()` 实现，`precompute_log_probabilities()` 那几行是唯一需要稍微动脑子翻译的地方。写完之后，你对这个方法的理解会再上一个台阶。
*（内容由AI生成，仅供参考）*
