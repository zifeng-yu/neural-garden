# 指标 Agent 数据集

[fixture.json](fixture.json) 从当前 tests/test_feedback_tools.py 的常量导出，包含 11 条合成搜索记录、13 条合成反馈事件和 [2026-10-01,2026-10-08) 的三个期望结果。包含区间前后、午夜边界、重复点击、非 click 事件和缺少文档 ID 等条件。

旧 notes/ng-usage-practice/fixture.sqlite 已删除；该合成数据与删除前的样本记录一致。自动化测试用 tmp_path 创建数据库并调用项目 create_tables，测试数据维护入口仍是 tests/test_feedback_tools.py；修改它时应同步此文档快照。

[cases.jsonl](cases.jsonl) 从三个历史日志整理出：热门词首问及零结果追问、缺参澄清、真实库点击查询。真实库用例 ma03 的历史期望只能依据保存的日志，未保存当时数据库快照；不能用今天的数据库重跑后要求相同结果。

严格日期、Top5、禁止额外参数、缺库/缺表等离线场景由 tests/test_feedback_tools.py 维护，不代表这些情况已通过真实模型端到端测试。
