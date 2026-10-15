# 第 2 课：Self-Attention & Transformer

> 课次：2/6
> 难度：★★☆☆☆~★★★☆☆

---

## 一、直觉引入（~300 字）

想象你在读一本技术书。读到第 50 页时，你突然看到一个术语"多态"，但你记不清它的定义了。你会怎么做？

**人类的做法**：快速翻回第 10 页（那里第一次提到这个术语），扫一眼定义，然后翻回第 50 页继续阅读。整个过程不到 2 秒，而且你精准地知道该看哪里。

**传统 RNN/ LSTM 的做法**：从第 1 页开始，一个字一个字读到第 50 页，希望中间的信息没有忘记。但读到第 50 页时，第 10 页的内容可能已经模糊了——这就是"长距离依赖"问题。

**Transformer 的做法**：在第 50 页的每个字，都可以直接"看到"前面所有页的任意位置。它不是顺序阅读，而是并行地建立所有位置之间的连接。这种机制叫 **Self-Attention（自注意力）**。

**直觉理解**：Attention 就是一个"可微分的数据库查询"。对于每个 token，模型学习三个向量：
- **Query（Q）**：我想找什么信息？
- **Key（K）**：我有什么信息可以提供？
- **Value（V）**：我实际存储的内容是什么？

Query 和 Key 匹配，决定"关注度"；Value 是实际传递的内容。这和数据库的 `SELECT * WHERE key MATCHES query` 几乎一样，只是全部可微分、可以端到端训练。

---

## 二、核心机制（~800 字）

### 2.1 Scaled Dot-Product Attention

Attention 的核心公式（必须记住）：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**逐步拆解**：

1. **QK^T**：计算 Query 和所有 Key 的点积相似度
   - 如果有 n 个 token，每个 token 的维度是 d_k
   - Q 形状：(n, d_k)，K 形状：(n, d_k)
   - QK^T 形状：(n, n)，表示每个 token 对其他所有 token 的关注度

2. **除以 √d_k**：缩放因子，防止点积过大导致 softmax 梯度消失
   - 为什么需要？假设 d_k=1024，每个维度值在 [-1,1]，点积可能达到 1024
   - softmax(1024) → 接近 one-hot，梯度几乎为 0，无法训练
   - 除以 √1024≈32，把值拉回到合理范围

3. **softmax**：把相似度归一化为概率分布
   - 每一行的和为 1，表示"对所有 token 的注意力权重"
   - 例如：[0.05, 0.02, 0.60, 0.03, 0.30] 表示最关注第 3 个 token

4. **乘以 V**：按注意力权重加权求和 Value
   - 输出形状：(n, d_v)，每个 token 的输出是所有 token Value 的加权和

**例子**（简化版，d_k=4）：
```
句子："我 喜欢 吃 苹果"

假设第 2 个 token"喜欢"的 Query 是 [0.8, 0.6, 0.2, 0.1]
所有 token 的 Key 矩阵：
我   [0.7, 0.5, 0.3, 0.2]
喜欢 [0.8, 0.6, 0.2, 0.1]
吃   [0.6, 0.4, 0.5, 0.3]
苹果 [0.9, 0.7, 0.1, 0.0]

Q·K^T 计算：
喜欢·我   = 0.8*0.7 + 0.6*0.5 + 0.2*0.3 + 0.1*0.2 = 0.94
喜欢·喜欢 = 0.8*0.8 + 0.6*0.6 + 0.2*0.2 + 0.1*0.1 = 1.05
喜欢·吃   = 0.8*0.6 + 0.6*0.4 + 0.2*0.5 + 0.1*0.3 = 0.85
喜欢·苹果 = 0.8*0.9 + 0.6*0.7 + 0.2*0.1 + 0.1*0.0 = 1.16

缩放后（d_k=4, √4=2）：[0.47, 0.525, 0.425, 0.58]
softmax 后：[0.23, 0.25, 0.22, 0.30]

→ "喜欢"最关注"苹果"(0.30)，其次是"喜欢"自己 (0.25)
```

### 2.2 Multi-Head Attention

单个 Attention 头只能学习一种注意力模式。Multi-Head 让模型同时学习多种模式。

**公式**：
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$$

其中 $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$

**直觉**：
- 头 1：学习语法关系（主谓宾）
- 头 2：学习语义关系（同义词、上下位）
- 头 3：学习指代关系（代词→先行词）
- 头 4：学习位置关系（相邻、相隔 n 个 token）
- ...

**为什么有效**：不同头可以 specialization（专业化），就像 CNN 的不同卷积核学习边缘、纹理、形状等不同特征。

**典型配置**：
- GPT-2 Small：12 头，每头 64 维 → 总维度 768
- Llama 3 8B：32 头，每头 128 维 → 总维度 4096
- 头数越多，模型能捕捉的关系类型越丰富

### 2.3 Transformer Block 完整结构

一个标准的 Transformer Decoder Block（GPT 风格）：

```
输入 x
  ↓
[LayerNorm 1] → [Multi-Head Self-Attention] → [+ Residual]
  ↓
[LayerNorm 2] → [FFN (MLP)] → [+ Residual]
  ↓
输出
```

**关键组件**：

1. **LayerNorm（层归一化）**：
   - 公式：$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sigma} + \beta$
   - 作用：稳定训练，防止梯度爆炸/消失
   - 位置：现代架构用 Pre-LN（在 Attention/FFN 之前），比 Post-LN 更稳定

2. **Residual Connection（残差连接）**：
   - 公式：$y = \text{LayerNorm}(x) + x$
   - 作用：让梯度可以直接流过，深层网络可训练
   - 直觉：如果 Attention 学不到东西，至少输出等于输入（恒等映射）

3. **FFN（Feed-Forward Network）**：
   - 结构：Linear → GELU → Linear
   - 维度：通常是 hidden_dim 的 4 倍（如 4096→16384→4096）
   - 作用：增加非线性，让模型能学习更复杂的变换

### 2.4 Causal Mask（因果掩码）

GPT 是 Decoder-only 架构，只能看到前面的 token，不能看到未来的 token。这是通过 Causal Mask 实现的。

**Mask 矩阵**（4 token 示例）：
```
    我  喜欢  吃  苹果
我   0  -∞   -∞  -∞
喜欢  0   0   -∞  -∞
吃   0   0    0  -∞
苹果  0   0    0   0
```

- 0 表示可以关注，-∞ 表示不能关注（softmax 后为 0）
- 下三角矩阵，保证每个 token 只能看到自己及之前的 token

**为什么需要**？语言建模是预测下一个 token，如果看到未来信息，训练和推理不一致，模型会失效。

---

## 三、代码级拆解（~500 字）

```python
import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Q, K, V 投影矩阵
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x, causal_mask=None):
        batch_size, seq_len, _ = x.shape
        
        # 1. 投影 Q, K, V
        Q = self.q_proj(x)  # (batch, seq_len, embed_dim)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # 2. 分成多头 (batch, seq_len, num_heads, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 3. 计算注意力分数 Q·K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # scores 形状：(batch, num_heads, seq_len, seq_len)
        
        # 4. 应用因果掩码（如果是 GPT 风格）
        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        
        # 5. softmax + V
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)  # (batch, num_heads, seq_len, head_dim)
        
        # 6. 合并多头 + 输出投影
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(attn_output)
        
        return output, attn_weights

# 使用示例
embed_dim = 768
num_heads = 12
seq_len = 512
batch_size = 4

# 创建因果掩码
causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
# 形状：(1, 1, seq_len, seq_len)，广播到 batch 和 head

model = SelfAttention(embed_dim, num_heads)
x = torch.randn(batch_size, seq_len, embed_dim)  # 模拟输入

output, attn_weights = model(x, causal_mask)
print(f"输出形状：{output.shape}")  # (4, 512, 768)
print(f"注意力权重形状：{attn_weights.shape}")  # (4, 12, 512, 512)

# 可视化注意力（看第 1 个样本、第 1 个头、最后一个 token 关注哪里）
import matplotlib.pyplot as plt
last_token_attn = attn_weights[0, 0, -1, :].cpu().numpy()
plt.bar(range(seq_len), last_token_attn)
plt.title("最后一个 token 的注意力分布")
plt.xlabel("Token 位置")
plt.ylabel("注意力权重")
plt.show()
```

**关键注释**：
- `transpose(1, 2)`：把 (batch, seq_len, heads, dim) 变成 (batch, heads, seq_len, dim)，方便并行计算所有头
- `masked_fill`：把不能关注的位置设为 -∞，softmax 后变成 0
- `contiguous()`：transpose 后内存不连续，view 之前需要重新排列内存
- 注意力权重可以用于可解释性分析（看模型关注哪里）

---

## 四、关键翻转（~200 字）

**反直觉洞察：Attention 不是"聚焦重点"，而是"信息路由"。**

很多人以为 Attention 是让模型"关注重要的词"，比如名词、动词。但这不是全部真相。

**Attention 的真正作用**：让每个 token 可以从任意位置"拉取"它需要的信息。有时候是聚焦（如代词找先行词），有时候是分散（如句法结构需要多个词的信息），有时候甚至是抑制（如否定词需要降低后续词的权重）。

**关键证据**：
- 有些头的注意力分布很均匀，没有明显的"聚焦"
- 有些头专门关注固定距离的 token（如 +1 位置），像卷积核
- 有些头在句法任务上激活，但在语义任务上沉默

**工程含义**：
- 不要过度解读注意力权重（它不是重要性分数）
- Multi-Head 的多样性比单个头的"可解释性"更重要
- 长上下文中，注意力会稀释——这是 RAG 的根本原因

---

## 五、前沿动态（~300 字）

### 1. FlashAttention-3（2024-2025）

FlashAttention 系列是近年来最重要的注意力优化技术：

**FlashAttention-1（2022）**：
- 核心思想：IO 感知，避免把整个注意力矩阵写入 HBM（显存）
- 技术：分块计算（tiling），在 SRAM（片上缓存）中完成 QK^T 和 softmax
- 效果：比标准 Attention 快 3-7 倍，显存占用减少 10-20 倍

**FlashAttention-2（2023）**：
- 优化：更好的并行策略，减少线程间同步
- 支持：可变序列长度，动态 batching
- 效果：比 FA-1 再快 2 倍，成为 HuggingFace、vLLM 等框架的默认后端

**FlashAttention-3（2024-2025）**：
- 新特性：支持 FP8 精度，利用 H100/B200 的 FP8 Tensor Core
- 稀疏注意力：动态跳过低注意力权重的计算
- 分布式：跨多 GPU 的 Ring Attention 集成
- 效果：在 H100 上比 FA-2 再快 1.5-2 倍，长序列（32k+）优势更明显

**为什么重要**：没有 FlashAttention，长上下文 LLM 的成本会高 5-10 倍。它是 RAG 和长文档处理的使能技术。

### 2. Mamba / SSM（State Space Models）（2024-2026）

Mamba 是 2024 年最热门的 Transformer 替代架构：

**核心思想**：用状态空间模型（SSM）替代 Attention，实现线性复杂度 O(n)。

**关键创新**：
- **选择机制（Selection Mechanism）**：SSM 的参数是输入依赖的，可以动态选择记住/忘记什么
- **硬件感知算法**：类似 FlashAttention 的分块策略，在 GPU 上高效并行
- **混合架构**：Mamba-2 结合 Attention 和 SSM，兼顾长程依赖和局部精度

**性能对比**（Mamba-2 2024）：
- 推理速度：比 Transformer 快 3-5 倍（长序列）
- 显存占用：线性增长，Transformer 是平方增长
- 效果：在语言建模任务上接近 Llama 2，但在某些推理任务上仍有差距

**2025-2026 进展**：
- **Jamba（AI21, 2024）**：Attention + Mamba 混合，128k 上下文
- **Bolt（2025）**：纯 SSM 架构，在代码生成任务上超越 Llama 3
- **Griffin（Google DeepMind, 2024）**：门控线性注意力，线性复杂度

**趋势**：Attention 不会完全被替代，但混合架构（Attention + SSM）会成为长上下文场景的主流。

### 3. Ring Attention（2024-2025）

**问题**：单 GPU 显存有限，如何训练/推理 100k+ 上下文的模型？

**方案**：Ring Attention 把序列分片到多个 GPU，每个 GPU 只存一部分 KV，通过环形通信交换 QK^T 的结果。

**效果**：
- 上下文长度线性扩展：8 卡 = 8 倍上下文
- 通信开销小：只传输 KV，不传输整个激活
- 成为长上下文训练的标准方案（如 Llama 3.1 405B 的 128k 上下文）

---

## 六、工程连接（~200 字）

**这和你要建的个人知识系统有什么关系**？

1. **理解 RAG 的必要性**：
   - Attention 机制在长上下文下会"稀释"：100k token 中，每个 token 平均注意力权重只有 0.00001
   - 即使有 FlashAttention，计算成本也随长度线性增长
   - **RAG 的本质**：把"大海捞针"变成"先定位再精读"——先用向量检索找到相关 chunk（10-20 个），再让 LLM 用 Attention 精细处理

2. **模型选型**：
   - 短上下文（<8k）：标准 Transformer（Llama 3、Qwen 2.5）
   - 长上下文（8k-128k）：选支持 FlashAttention-2/3 的模型
   - 超长上下文（>128k）：考虑混合架构（Jamba、Mamba-2）或 RAG

3. **推理优化**：
   - 必须用支持 FlashAttention 的推理框架（vLLM、TGI）
   - KV Cache 是显存瓶颈：4096 维度、32 层、8k 上下文 ≈ 8GB 显存
   - 量化（FP16→INT4）可以减半显存，但可能影响长上下文性能

4. **Chunk 长度设计**：
   - 太短（<200 字）：语义不完整，Attention 无法建立跨 chunk 连接
   - 太长（>2000 字）：注意力稀释，关键信息被淹没
   - 推荐：500-800 字/块，重叠 100-150 字

**下一步**：第 3 课讲训练范式（Pre-training → SFT → RLHF），你会理解为什么 LLM"知道但不一定正确"，以及为什么你的系统必须有外部验证机制。
