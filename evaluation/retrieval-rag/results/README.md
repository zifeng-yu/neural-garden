# 检索与 RAG 已有结果

批次：2026-10-06T01:17:11.598939+00:00，共 7 题 × 3 模式 = 21 条记录，生成模型为历史报告所记 qwen3.5-plus。

- [results.jsonl](results.jsonl)：两路候选、融合排名、sources、模型回答、耗时与原标注指标。
- [report.md](report.md)：逐题原始报告和人工核验。
- [审查报告](hybrid-search-review-2026-10-05.md)：首轮链路核验、标注订正和适用范围。
- [存储与代码指纹](hybrid-review-evidence-2026-10-05.json)：当时 ES mapping、数据一致性和指纹。
- [两条真实问答证据](hybrid-review-live-2026-10-05.json)：首轮真实调用结果。

有答案回归题 q01–q04：

| 模式 | Hit@3 | MRR@3 原标注 | MRR@3 事后补入等价证据 |
|---|---|---|---|
| vector | 3/4 | 0.5833 | 0.7500 |
| bm25 | 4/4 | 0.7500 | 0.8750 |
| hybrid | 4/4 | 0.5417 | 0.7083 |

上表沿用历史审查口径；原始数据未改写。q06 单独记录，三模式均命中；q05/q07 三模式均拒答。局部结果支持混合检索扩大 q02 的完整证据覆盖，但不支持宣称整体排序或准确率全面提升。

已记录问题：q03 三模式都遗漏“超过”；q04 已有足够 sources 却拒答；q02 vector 回答引用仅支持部分条件。默认保留 vector、提供 hybrid 可选是当时结论。

原始审查中的旧代码路径、notes/hybrid-eval 路径和已迁移测试命令属于历史记录。当前评估路径以本目录及 ../datasets/ 为准；功能测试使用 tests/test_hybrid_search.py、tests/test_rag_answer.py、tests/test_api_contract.py、tests/test_api_answers.py。

历史脚本保存在 [archived-scripts/](archived-scripts/)，仅作实验实现存档：硬编码旧本机路径、加载时有副作用，不能按新目录直接运行。此次整理没有重新执行脚本或模型调用。后续需先重构路径和 main 入口，并为每次运行保存单独结果。
