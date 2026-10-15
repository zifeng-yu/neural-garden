# Day 8｜一个 type hint 为什么拦不住错误输入

> 日期：2026-09-15
> 预计阅读：约 22 分钟
> 今日只回答一个问题：当你想把 VectorRetriever 换成另一套实现时，Python 里到底靠什么保证调用方不用改一行代码？

## 开场：事情从哪里开始失控

**假设场景。** 一个团队正准备给检索评估 runner 加第二条通道。初级工程师写了 `KeywordRetriever`，接口声明得清清楚楚：

```python
def search(self, query: str, k: int) -> list[SearchResult]: ...
```

mypy 全绿，CI 通过。评估 runner 跑到第二批 query 时崩了：

```
AttributeError: 'dict' object has no attribute 'score'
```

他很困惑：类型注解明明写着 `list[SearchResult]`，里面怎么会混进 dict？因为 Python 把元素装进 list 的那一刻不做任何检查；他的实现里有一行调用了无类型标注的第三方方法，mypy 拿不到信息，把整个表达式当作 `Any` 放行。type hint 不是墙，它更像写给检查员看的一张纸条——而检查员（mypy）只在 CI 里上班，运行时从来不出勤。

昨天的巡游结束时留过一个尾巴：要把 BM25 通道接进 RRF，你怎么保证两路输出的形状一致？今天正面回答它。

## 先给结论

三个判断，先摆出来：

1. **Python 的 type hint 不参与执行。** 解释器不强制、不拒绝，运行时没有"自动校验"这回事。违反 contract 的代码照样通过编译期注解，除非有静态检查器（mypy / pyright）在发布前替你抓出来。"类型写对了"和"跑起来是对的"是两件事。
2. **Dataclass 解决"数据形状"。** 它给返回值一个名字、一个构造函数、比较和打印能力；但默认情况下，它同样不校验字段类型。
3. **Protocol 解决"行为契约"。** 只要实现了 `search(query, k)` 这个形状，你就*是*一个 Retriever，不需要显式继承或注册。这是 Python 对 Java interface 的回答，但机制相反：Java 认血统（名义类型），Python 看形状（结构类型）。

对 AI Backend 来说这件事很具体：retriever、reranker、embedding client、vector store，全都是天然的"多实现组件"。没有显式 contract，评估 runner、融合层、线上链路就各自依赖实现细节，换一路实现就全线震荡。

## 事情为什么会这样

沿着链路走：问题 → 输入 → 处理 → 输出 → 失败位置 → 判断依据。

### 鸭子类型的代价：失败会迟到

Python 接受任何对象，只要调用发生前方法存在。好处是耦合度极低；代价是**错误不在"传错东西"的位置暴露，而在几十行后"访问一个不存在的属性"时暴露**。

对照你的 Java 经验：`Retriever r = new KeywordRetriever()` 如果没写 `implements Retriever`，编译器直接拒绝，运行时 JVM 也维护着类型体系。Python 的注解在代码被执行前就已经"蒸发"了。纸条类比必须映射回真实对象：纸条 = 源码里的注解；检查员 = CI 中运行的 mypy；门口那个强制执行的人——Python 根本没有，你得自己造（构造器校验、显式 assert、或者测试）。

### Dataclass 到底加了什么

不用 dataclass 时，`SearchResult` 要手写 `__init__` 赋值、`__repr__`、`__eq__`。`@dataclass` 根据字段注解生成这些方法。官方文档列出的开关包括 `frozen`、`eq`、`order` 等。三个要点：

- 注解依然只是提示：`score: float` 的字段，构造时传字符串 `"0.8"` 不会报错；
- 可变默认值必须走 `field(default_factory=list)`，直接写 `items: list = []` 会被 dataclass 主动拦下（这正是你 Extra Skill Day 2 学的可变默认参数坑的语言级版本）；
- `frozen=True` 让搜索结果变成不可变值——评估管道里并发传递它，不用担心有人偷偷改分数字段。

### Protocol：按行为结契，不按血统

PEP 544（Python 3.8 起进入标准库 typing）定义了 Protocol：

```python
class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[SearchResult]: ...
```

任何拥有兼容 `search` 方法的对象，静态检查时就满足这个契约——**不需要继承，不需要注册，甚至不需要知道 Protocol 的存在**。

与 Java interface 的关键差异值得说透：Java 是 nominal typing（名义类型），你必须写明"我实现了谁"，编译器按声明认亲；Python Protocol 是 structural typing（结构类型），检查器只比对方法名和签名形状。代价是这份契约完全活在静态世界里：运行时默认连 `isinstance` 都不支持；加上 `@runtime_checkable` 装饰器后 `isinstance` 可用，但**只检查属性名是否存在，不检查参数签名，更不检查返回值内容**。

所以工程上的正确姿势是分层防御：

```python
from dataclasses import dataclass
from typing import Protocol

@dataclass(frozen=True)
class SearchResult:
    query: str
    doc_id: str
    score: float
    source: str              # "vector" | "bm25"
    snippet: str = ""

class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[SearchResult]: ...

class VectorRetriever:                       # 注意：没有继承任何类
    def __init__(self, store) -> None:
        self.store = store
    def search(self, query: str, k: int) -> list[SearchResult]:
        return [SearchResult(query, h["id"], h["score"], "vector")
                for h in self.store.query(query, top_k=k)]

def run_eval(r: Retriever, queries: list[str], k: int = 10):
    return {q: r.search(q, k) for q in queries}
```

`VectorRetriever` 没有声明任何血统，`run_eval` 却能接受它——将来加 `KeywordRetriever`、`HybridRetriever`，同样只要形状匹配。反过来，如果哪天有人在 `run_eval` 里写 `if type(r) is VectorRetriever: ...`，契约就已经破了。

一个迷你输入输出例子：`run_eval(vec, ["梯度裁剪", "RAG 评估"], k=3)` 返回两个键，每个值是一串 `SearchResult`；若某实现把 dict 混进列表，静态检查放行、取值时崩溃——这就是开场场景的全部机理。

收一句本节主线：**type hint 承诺"长什么样"，dataclass 承诺"有什么字段"，Protocol 承诺"能做什么动作"，只有测试承诺"结果是对的"。前三者运行时都可绕过。** 还记得 Day 1 那句话吗——看起来不错不算证据，这里同理：mypy 全绿也不算证据。

## 把系统画出来

| 机制 | 何时生效 | 校验什么 | 不校验什么 |
|---|---|---|---|
| type hint | 仅静态（mypy/pyright） | 调用点与声明是否一致 | 任何运行时行为 |
| `@dataclass` | 运行时（构造时） | 字段齐全、默认值、`__eq__`/`__repr__` | 字段类型 |
| `Protocol` | 仅静态 | 方法名 + 签名的结构匹配 | 语义、运行时兼容 |
| `isinstance` + `runtime_checkable` | 运行时 | 属性**名字**是否存在 | 参数签名、返回内容 |
| 行为测试 | 运行时 | 输入输出的语义正确性 | 未覆盖的调用点 |

读表方式：每一行都只堵一个角度的洞。真正的系统是把前三行放进 CI、第四行放进入口校验、第五行当作最终证据。

## 放回三个项目中看

### ML/DL 概念层

ML pipeline 里 contract 的对应物是 **shape**。forward 时维度不匹配，报错常常发生在第五层之后，而不是"传错张量"的那一刻——和鸭子类型的延迟失败一模一样。学 tensor shape，本质上是在学一个运行时才生效的类型系统：对每一层先写清"输入 shape、输出 shape、错了会在哪里暴露"。静态快照里你尚未有 shape 独立掌握的证据，今天这个 contract 视角正好是一根可借用的扶手。另外 Day 7 的 LTR 也有同构问题：feature 向量的列顺序就是样本构造与训练之间的 contract，错位不报错，只会悄悄把指标劣化——比类型错误更危险，因为它连崩溃都不给你。

### Python / AI Backend 工程层

三个直接后果。其一，异步：如果 retriever 需要并发调用，contract 就该写成 `async def search(...)`；sync 和 async 实现在 mypy 下可区分，但极易被忽略，混用时的报错发生在 `await` 一行。其二，返回类型：你 Extra Skill 学过的 generator 在这里回来了——声明 `list[SearchResult]` 却 `return` 一个 generator 是静态违规；反过来，把 contract 声明为 `Iterator[SearchResult]` 就能合法地流式返回大候选集。其三，contract 也包含失败语义：一个实现吞掉异常返回 `[]`，另一个抛 `ConnectionError`，两者静态上都满足 Protocol，但对调用方是两种完全不同的系统——明天整篇讲这个。

### Neural Garden 项目层

根据任务建立时提供的项目状态快照：默认搜索链路只有 Knowledge Unit 向量检索，BM25、Hybrid、RRF、Reranker 均未实现；SQLite 是结构化事实来源，ChromaDB 存向量。高 ROI 路线的第二步（评估 runner + Vector baseline）一旦落地，第一件事就是用同一套 query 集比较不同检索策略——那正是今天这套 contract 的最佳施用场景：一个 `Retriever` Protocol，几路实现，一个 runner，比较代码只写一遍。需要什么证据再决定？做一次真实试点：让向量路和 BM25 路各自实现同一接口，看返回形状是否真的能统一——向量路顺手能带 snippet，BM25 路可能只有 doc_id 和分数，此时 contract 需要加可选字段还是改为"返回 id 再统一取文"，这是一个要靠写一遍代码才能判断的设计取舍，现在下结论为时过早。另外提醒：10 张表、54 项测试是任务建立时的快照，今天不假设有变。

## 最容易踩的两个坑

**坑一：把 mypy 绿灯当运行时安全。** mypy 看不穿无标注代码、第三方库调用和 `Any` 的传播——类型信息一旦经过一个无注解的边界就蒸发。所以对外部来源（LLM 返回、向量库响应、用户上传文件）的解析点，需要显式校验：用 dataclass 手工构造并捕获字段错误，或者引入校验库。术语叫 **trust boundary（信任边界）**：从那里进来的所有东西，静态上是 `Any`，工程上是"未验毒品"。

**坑二：拿 `@runtime_checkable` 的 isinstance 当架构守护。** 它只确认名字存在。一个 `search(self, q)`（参数不同）的对象能顺利通过 `isinstance(obj, RetrieverCheckable)`，然后在调用那一行炸给你看。需要运行时硬保证时，在工厂函数里做显式检查——检查方法签名（`inspect.signature`）或者干脆注册表白名单——而不是假装 isinstance 有语义理解能力。

## 面试官可能继续追问

> "Python 的 Protocol 和 Java 的 interface 本质区别是什么？你的评估管道怎么保证新的检索实现接进来不用改代码？"

回答骨架：

1. **先给结论**：interface 是名义类型——显式声明、编译强制；Protocol 是结构类型——按方法形状匹配、只有静态工具强制。
2. **再讲原因**：Python 把宽松留给运行时，把契约检查外包给 CI 里的 mypy；代价是运行时不兜底，所以还需要构造器校验和行为测试补第二、三层。
3. **落到场景**：在 Neural Garden 加 BM25 时，我会让两路检索都满足同一个 Retriever Protocol，返回 frozen dataclass 的 SearchResult；配一个 golden query 集的行为测试（例如固定 50 条查询、断言 Recall@10 落在预期区间），实现替换的成本和回归风险都变成可度量的事。

## 合上文档前的 60 秒

**今天最重要的三句话：**

- type hint 不会执行，执行它的是 mypy，而 mypy 看不穿 Any 的边界；
- Protocol 按行为结契、不按血统——这是 Python 解决多实现组件的可替换方式；
- dataclass 给了数据形状，不给你类型安全；运行时底线永远是显式校验加测试。

**60 秒口头题：** 现在要求 `VectorRetriever` 和 `KeywordRetriever` 通过同一份 mypy 检查——它们最少需要各自拥有哪个方法、什么签名？在脑子里把 Protocol 定义写一遍，再想想返回类型选 `list` 还是 `Iterator`，理由是什么。

## 明天的钩子

contract 定义了一次"正确的返回"长什么样。但线上系统更多处理的是"不正确的失败"：Embedding API 三十秒不吐字，是连接超时还是读取超时？该不该重试？重试三次会不会写出三份重复数据？明天拆 Timeout、Retry 与异常设计——以及为什么 `except Exception: return None` 是诊断性的头号杀手。

## 资料来源

1. Python 官方文档 · `dataclasses`（生成 `__init__`/`__repr__`/`__eq__` 的机制与 `frozen` 等参数，Python 3.7 引入）：https://docs.python.org/3/library/dataclasses.html
2. PEP 544 – *Protocols: Structural subtyping (static duck typing)*（Python 3.8 起，Final）：https://peps.python.org/pep-0544/
3. mypy 官方文档 · Protocols and structural subtyping（`runtime_checkable` 只检查属性存在性的限制）：https://mypy.readthedocs.io/en/stable/protocols.html

*本篇的 dataclass 与 PEP 544 事实已于今日对照官方文档在线核对。*
