# 项目评估

整理日期：2026-10-11。这里只整理已有证据，没有重新调用模型、检索服务或修改业务数据库。

| 评估类型 | 现状总结 | 方法论（指标） | 数据集 | 实际测试情况与说明 |
|---|---|---|---|---|
| 检索与 RAG | [现状总结](retrieval-rag/current-status/README.md) | [指标与核验口径](<retrieval-rag/methodology/README.md>) | [问题与原始标注](<retrieval-rag/datasets/README.md>) | [结果与结论](<retrieval-rag/results/README.md>) |
| 知识检索 Agent | [现状总结](knowledge-retrieval-agent/current-status/README.md) | [工具与证据核验](<knowledge-retrieval-agent/methodology/README.md>) | [历史场景整理](<knowledge-retrieval-agent/datasets/README.md>) | [运行记录与失败案例](<knowledge-retrieval-agent/results/README.md>) |
| 指标 Agent | [现状总结](usage-metrics-agent/current-status/README.md) | [选参与统计核验](<usage-metrics-agent/methodology/README.md>) | [样本数据与场景](<usage-metrics-agent/datasets/README.md>) | [历史调用与当前工具测试](<usage-metrics-agent/results/README.md>) |

每类采用相同四个子目录。current-status 汇总当前能力、已有证据、问题与待验证项；方法论说明如何评价；数据集保存输入和预期；实际测试情况保存输出、人工判断和限制。自动化功能测试仍在 `tests/`。

原始报告、JSON、日志和历史脚本按字节保留，旧命令、路径和工具名称代表运行当时的状态。新说明负责解释它们与当前实现的区别，不能把历史结果当成当前版本已重新验证。Agent 场景是从结果记录中整理出来的，没有事前冻结证据，不属于未见测试集。

原始资料已迁入对应评估目录，临时迁移清单已移除。日志原路径对应的测试数据库已删除，合成数据保存在指标评估数据集中。真实业务数据库未复制到评估目录。
