# 第 4 课：Inference 工程

> 课次：4/6
> 难度：★★☆☆☆

---

## 一、直觉引入（~300 字）

想象你在写代码时，IDE 每次只给你一个字符的建议——你敲 `f`，它提示 `o`；你敲 `o`，它提示 `r`……这显然太慢了。实际上，IDE 会一次性给你一整段补全：`for i in range(10):`。

LLM 推理面临同样的问题。传统推理是「自回归」的：生成 100 个 token 需要模型跑 100 遍，每遍只吐一个词。这就像让法拉利在城市里开——引擎很强，但一直在等红绿灯。

**Inference 工程的核心任务**：让模型在保持输出质量的前提下，尽可能「并行生成」多个 token，同时用更少的内存装下更多的请求。

本课我们拆解三个关键技术：
1. **KV Cache**——记住之前的计算结果，不用重复算
2. **Decoding 策略**——决定是「贪心选最优」还是「采样求多样」
3. **量化**——把模型从 FP16 压到 INT4，显存占用减 4 倍

这些技术直接决定你的个人知识系统：是用云端 API（便宜但慢），还是本地部署（快但吃显存）。

---

## 二、核心机制（~800 字）

### 1. KV Cache：用空间换时间

Transformer 生成第 `t` 个 token 时，需要看到前面所有 `t-1` 个 token 的注意力计算结果。如果每次都重新算，复杂度是 O(n²)。

**KV Cache 的核心思想**：把之前 token 的 Key 和 Value 矩阵缓存起来，新 token 只需要算自己的 QKV，然后和缓存做注意力。

```
传统方式（每步重算）:
token_1: 算 QKV → 注意力 → 输出
token_2: 重新算 token_1 和 token_2 的 QKV → 注意力 → 输出  ❌ 浪费

KV Cache 方式:
token_1: 算 QKV → 缓存 K₁,V₁ → 注意力 → 输出
token_2: 只算 Q₂,K₂,V₂ → 读取缓存 K₁,V₁ → 注意力 → 输出  ✅ 高效
```

**数学表达**：
注意力公式是 `Attention(Q, K, V) = softmax(QK^T / √d) · V`

生成第 `t` 个 token 时：
- `Q_t` 是当前 token 的 Query
- `K_{1:t}` 是缓存的 1 到 t 所有 Key
- `V_{1:t}` 是缓存的 1 到 t 所有 Value

**显存占用计算**：
对于一个 7B 模型（如 LLaMA-7B），FP16 精度下：
- 每个 token 的 KV Cache ≈ `2 × 层数 × 隐藏层维度 × 序列长度 × 2(字节)`
- 以 LLaMA-7B 为例：32 层 × 4096 维度 × 2048 长度 × 2 字节 ≈ **512MB/请求**

这就是为什么长上下文会爆显存——不是模型本身，是 KV Cache 太大。

### 2. Decoding 策略：确定性 vs 多样性

生成 token 时，模型输出一个概率分布。怎么选下一个 token？

| 策略 | 公式 | 适用场景 |
|------|------|----------|
| **Greedy** | `argmax(P)` | 代码生成、数学题（需要确定性） |
| **Top-k** | 从概率最高的 k 个中采样 | 通用对话 |
| **Top-p (Nucleus)** | 从累积概率≥p 的最小集合中采样 | 创意写作 |
| **Temperature** | `P' = softmax(log(P)/T)` | T>1 更随机，T<1 更确定 |

**Top-p 采样示例**：
假设模型输出概率：`{"的": 0.4, "了": 0.3, "是": 0.15, "在": 0.1, "有": 0.05}`
- Top-k=3：只从 `{"的", "了", "是"}` 中选
- Top-p=0.8：累积概率 0.4+0.3+0.15=0.85≥0.8，所以从 `{"的", "了", "是"}` 中选

### 3. 量化：从 FP16 到 INT4

量化是把模型权重从高精度（FP16/FP32）转成低精度（INT8/INT4）的技术。

**核心公式**：
```
W_int = round(W_fp / scale) + zero_point
W_fp ≈ (W_int - zero_point) × scale
```

其中 `scale = (max(W) - min(W)) / (2^n - 1)`，n 是目标位宽。

**量化级别对比**：

| 精度 | 每权重字节 | 7B 模型显存 | 精度损失 |
|------|-----------|-----------|----------|
| FP16 | 2 字节 | 14GB | 无 |
| INT8 | 1 字节 | 7GB | <1% |
| INT4 (AWQ) | 0.5 字节 | 3.5GB | 1-3% |

**AWQ (Activation-aware Weight Quantization)** 的关键洞察：不是所有权重都同等重要。它通过激活值大小识别「重要权重」，对这些权重保留更高精度，从而实现 INT4 量化但损失更小。

---

## 三、代码级拆解（~500 字）

下面是一个简化的 KV Cache 实现，展示核心机制：

```python
import torch
import torch.nn.functional as F

class SimpleKVCache:
    """简化版 KV Cache 实现"""
    
    def __init__(self, num_layers, num_heads, head_dim, max_seq_len):
        # 预分配缓存空间 [层数，批次，头数，序列长度，头维度]
        self.k_cache = torch.zeros(num_layers, 1, num_heads, max_seq_len, head_dim)
        self.v_cache = torch.zeros(num_layers, 1, num_heads, max_seq_len, head_dim)
        self.current_pos = 0
    
    def update(self, layer_idx, k_new, v_new):
        """更新当前 token 的 KV 缓存"""
        # k_new 形状：[1, num_heads, 1, head_dim]
        batch_size, num_heads, seq_len, head_dim = k_new.shape
        
        # 将新 token 的 KV 写入缓存的当前位置
        self.k_cache[layer_idx, :, :, self.current_pos:self.current_pos+seq_len] = k_new
        self.v_cache[layer_idx, :, :, self.current_pos:self.current_pos+seq_len] = v_new
        
        return (
            self.k_cache[layer_idx, :, :, :self.current_pos+seq_len],
            self.v_cache[layer_idx, :, :, :self.current_pos+seq_len]
        )
    
    def reset(self):
        """重置缓存（新请求开始时调用）"""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.current_pos = 0


def attention_with_cache(q, k_cache, v_cache, mask=None):
    """带 KV Cache 的注意力计算"""
    # q: [batch, heads, seq_len, head_dim]
    # k_cache, v_cache: [batch, heads, cache_len, head_dim]
    
    head_dim = q.shape[-1]
    
    # 计算注意力分数：Q · K^T / √d
    scores = torch.matmul(q, k_cache.transpose(-2, -1)) / (head_dim ** 0.5)
    
    # 应用掩码（防止看到未来 token）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # softmax 得到注意力权重
    attn_weights = F.softmax(scores, dim=-1)
    
    # 加权求和 V
    output = torch.matmul(attn_weights, v_cache)
    
    return output


# 使用示例
if __name__ == "__main__":
    # 模拟 LLaMA 配置
    NUM_LAYERS = 32
    NUM_HEADS = 32
    HEAD_DIM = 128
    MAX_SEQ_LEN = 2048
    
    cache = SimpleKVCache(NUM_LAYERS, NUM_HEADS, HEAD_DIM, MAX_SEQ_LEN)
    
    # 模拟生成过程
    for token_idx in range(5):
        # 假设这是模型某一层计算出的新 K 和 V
        k_new = torch.randn(1, NUM_HEADS, 1, HEAD_DIM)
        v_new = torch.randn(1, NUM_HEADS, 1, HEAD_DIM)
        
        # 更新缓存并获取完整 KV
        k_full, v_full = cache.update(0, k_new, v_new)
        
        # 模拟 Query（当前 token）
        q = torch.randn(1, NUM_HEADS, 1, HEAD_DIM)
        
        # 计算注意力
        output = attention_with_cache(q, k_full, v_full)
        
        print(f"Token {token_idx}: 输出形状 {output.shape}")
        cache.current_pos += 1
    
    print("✓ KV Cache 演示完成")
```

**运行结果**：
```
Token 0: 输出形状 torch.Size([1, 32, 1, 128])
Token 1: 输出形状 torch.Size([1, 32, 1, 128])
Token 2: 输出形状 torch.Size([1, 32, 1, 128])
Token 3: 输出形状 torch.Size([1, 32, 1, 128])
Token 4: 输出形状 torch.Size([1, 32, 1, 128])
✓ KV Cache 演示完成
```

---

## 四、关键翻转（~200 字）

**你以为**：推理加速主要是让模型「算得更快」。

**实际上**：现代推理系统的瓶颈不在计算，而在**显存带宽**和**显存管理**。

- GPU 算力过剩：A100 的算力远超过 LLM 推理需求
- 真正卡脖子的：每次生成 token 都要从显存读取全部权重（7B 模型=14GB），显存带宽成了瓶颈
- **PagedAttention 的核心洞察**：KV Cache 的内存碎片化严重，借用操作系统的「分页」思想，把 KV Cache 切成固定大小的页，按需分配，显存利用率从 60% 提升到 96%

这就是为什么 vLLM 能比传统服务系统快 2-4 倍——不是模型变聪明了，是显存管理变聪明了。

---

## 五、前沿动态（~300 字）

### 1. vLLM PagedAttention（SOSP 2023）

**论文**：[Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)

**核心贡献**：
- 将 KV Cache 分块成固定大小的「页」（类似操作系统虚拟内存）
- 支持跨请求共享 KV Cache 页（相同前缀的请求不用重复存储）
- 吞吐量提升 2-4 倍，显存浪费从 40% 降到<4%

**工业界采用**：
- Fireworks AI、Together.ai 等云端推理服务底层都用 vLLM
- 2024 年推出 vLLM 0.4.0，支持多 GPU 张量并行

### 2. Speculative Decoding（ICML 2023 Oral）

**论文**：[Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)

**核心思想**：
- 用一个小模型（Draft Model）快速生成多个候选 token
- 大模型一次性验证这些 token，接受或拒绝
- 理论加速比 = 小模型与大模型的速度比 × 接受率

**实测效果**：
- T5-XXL 上实现 2-3 倍加速，输出分布完全不变
- 2024 年演进为「Medusa」、「Lookahead Decoding」等变体

### 3. MLC-LLM（持续更新）

**项目**：[mlc-ai/mlc-llm](https://github.com/mlc-ai/mlc-llm)

**定位**： universal LLM 部署引擎，一套编译后端通吃所有硬件

**支持平台**：
- NVIDIA/AMD/Intel GPU（CUDA/Vulkan/ROCm）
- Apple Silicon（Metal）
- 移动端（iOS/Android）
- 浏览器（WebGPU）

**技术栈**：基于 TVM 编译器，自动优化算子实现，INT4 量化原生支持。

---

## 六、工程连接（~200 字）

**这和你建个人知识系统有什么关系？**

1. **API vs 本地部署决策**：
   - 如果你每天调用<100 次 → 用云端 API（OpenAI/Anthropic/月之暗面）
   - 如果你需要高频调用（>1000 次/天）或数据敏感 → 本地部署 7B/13B 模型 + vLLM

2. **模型选型**：
   - 中文任务优先：Qwen2.5-7B/14B、Yi-1.5-9B
   - 英文任务优先：Llama-3.1-8B、Mistral-7B
   - 显存<8GB：必须用 INT4 量化版（AWQ/GGUF）

3. **上下文长度权衡**：
   - 长上下文=更多 KV Cache=更少并发请求
   - 如果你的知识库 chunk 是 500 字，没必要开 32k 上下文，8k 足够

**架构建议**：
```
用户提问 → 检索模块（向量 + 关键词）→ 拼接 context → 
本地 vLLM 服务（Qwen2.5-7B-INT4）→ 流式返回
```

---

## 七、🆕 常见坑（~150 字）

```
1. 「量化会让模型变傻」→ 其实是「AWQ/GGUF 的 INT4 量化损失<3%，人类几乎无法察觉」
2. 「显存够就能跑大模型」→ 其实是「KV Cache 会吃掉额外 20-50% 显存，长上下文更严重」
3. 「Temperature=0 就是 Greedy」→ 其实是「大部分 API 的 temperature=0 实际是 temperature=1e-8，仍有微小随机性」
```

---

## 八、🆕 本周小实验（可选，10-15 分钟）

### 实验：用 Ollama 对比不同量化级别的推理速度

**目标**：体验 INT4 vs FP16 的显存占用和速度差异

**前置条件**：
- 安装 Ollama：`curl -fsSL https://ollama.com/install.sh | sh`（Mac/Linux）
- 或下载 Windows 版：https://ollama.com/download

**步骤**：

```bash
# 1. 拉取 FP16 版本（约 14GB）
ollama pull qwen2.5:7b

# 2. 拉取 INT4 量化版本（约 4GB）
ollama pull qwen2.5:7b-instruct-q4_K_M

# 3. 运行并观察显存占用
ollama run qwen2.5:7b-instruct-q4_K_M "用一句话解释什么是 KV Cache"
```

**预期输出**：
```
KV Cache 是一种优化技术，在生成文本时缓存之前 token 的 Key 和 Value 矩阵，
避免重复计算注意力，从而加快推理速度并减少显存占用。
```

**验证速度差异**：
```bash
# 记录生成 100 个 token 的时间
time ollama run qwen2.5:7b-instruct-q4_K_M "写一首关于春天的五言绝句"
```

**如果报错怎么办**：
- `command not found: ollama` → 检查是否安装成功，或尝试重启终端
- `out of memory` → 换更小的模型：`qwen2.5:3b-instruct-q4_K_M`
- 下载太慢 → 用国内镜像：编辑 `/etc/hosts` 添加 GitHub 加速

---

**本课完**

> 下节课预告：第 5 课《幻觉与约束解码》——为什么 LLM 会一本正经地胡说八道？如何用技术手段强制它说真话？
