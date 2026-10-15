# 检索与 RAG 数据集

[cases.jsonl](cases.jsonl) 原样迁移：7 个问题，5 个 regression、2 个标记为 holdout；5 个有答案、2 个无答案。

每项包含 id、query、split、answerable、expected_answer、gold_chunks。gold_chunks 保存 document_id、chunk_id 和 content_hash；这些是当时数据库的标识，重建索引后不能只按旧 ID 判断命中，需要核对原文和 hash。

| ID | 检查内容 | 原有标注 |
|---|---|---|
| q01 | 数组重复检测的集合结构 | 有答案；原 gold 为 (1,3) |
| q02 | 根据情况调整判断、承认不确定性的专家类型 | 有答案；(4,20) |
| q03 | 专家与预测数量，保留“超过”限定词 | 有答案；(4,17) |
| q04 | 能否把整体统计推广为每位专家、所有类别 | 有答案；(4,19) |
| q05 | 未给出的精确总体百分比 | 无答案 |
| q06 | 材料对日本借钱问题的解释 | 有答案；原 gold 为 (5,25)，只按材料观点评估 |
| q07 | 资料没有提供的 NBA 冠军信息 | 无答案，不用模型外部知识回答 |

历史语料指纹和存储核验见 [证据 JSON](../results/hybrid-review-evidence-2026-10-05.json)。此处没有完整的 SQLite/Chroma/ES 语料快照；results.jsonl 保存了本批实际召回内容和入模原文，不能等同完整语料备份。
