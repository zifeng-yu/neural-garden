# 指标 Agent 已有结果

[record.md](record.md) 保留原始自述；[archived-logs/](archived-logs/) 保存三次运行；[historical-run-results.jsonl](historical-run-results.jsonl) 从日志提取问题、工具参数、工具结果和回答，便于查看。

| 场景 | 实际证据 | 结论与限制 |
|---|---|---|
| 热门词及零结果追问 | 合成库返回 RAG 3、ES 2 等；追问沿用 [10-01,10-08)，返回 RAG 2、Agent 1、ES 1 | 数值与日期继承在该历史运行中符合预期 |
| “帮我看看统计” | 模型询问指标与日期，没有工具调用 | 该历史场景澄清符合预期 |
| 真实库文档点击 | 03-ng-data.log 明确记录真实库路径，工具 items=[]，模型说明无符合口径记录 | 有真实库接入记录；不证明非空数据的正确性 |

原 record.md 写“没接入ng的表”，与 03-ng-data.log 的真实库调用不一致。整理以日志补充说明，原文不改写。日志还说“符合口径的记录共3项”，实际是 3 个搜索词分组、共 4 次零结果搜索，表达有歧义，不能理解为只有 3 次搜索。

三份日志模型 qwen3.5-plus、temperature=0，使用旧版 query_usage_summary(metric, dates) 单工具。当前实现已重构为三个独立工具，历史模型调用不能作为新版 Agent 的选工具通过证据。

2026-10-11 已执行当前 tests/test_feedback_tools.py 与 tests/test_feedback_analytics.py，共 27 项通过；默认 tests/ 共 179 项通过。这是离线工具/SQL 和功能测试，没有重跑真实模型 Agent。后续需对当前 src.usage_metrics_agent_cli 重新执行三个场景，并保存新版日志。
