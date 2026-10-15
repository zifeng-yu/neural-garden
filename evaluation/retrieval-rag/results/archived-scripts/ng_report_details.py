import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config.logging_config import setup_logging

logger = logging.getLogger(__name__)
setup_logging()
p = Path("/Users/zifeng/neural-garden/notes/hybrid-eval")
rs = [json.loads(l) for l in (p / "results.jsonl").read_text().splitlines()]
run = rs[-1]["run_id"]
rs = [r for r in rs if r["run_id"] == run]
reviews = {
    "q01": {
        "vector": "答案正确；S3明确写出HashSet用于检测重复，支持断言。",
        "bm25": "答案正确；S1的Python set实现与S2的HashSet说明共同支持答案，无关键条件遗漏。",
        "hybrid": "答案正确；S3明确支持HashSet检测重复。",
    },
    "q02": {
        "vector": "类型名称正确，但引用仅部分支持：S2说明狐狸会说“取决于很多因素”“我不确定”，支持承认不确定性；未明确写出根据具体情况调整判断。严格完整证据标准下不算完全支持，预填gold仍未命中。",
        "bm25": "答案正确，S1明确支持根据具体情况调整判断并承认不确定性，无关键遗漏。",
        "hybrid": "答案正确，S3明确支持两个条件，无关键遗漏。",
    },
    "q03": {
        m: "部分正确：引用S1支持284位专家及超过28000条预测；回答遗漏“超过”，把下限写成精确数量。严格标准下不算完全正确。"
        for m in ["vector", "bm25", "hybrid"]
    },
    "q04": {
        m: "回答未满足预期：sources含chunk19，原文足以解释“整体略好于随机，不能推到每位专家/所有类别”；模型却拒答。未产生错误断言，但遗漏应有的否定判断及理由。"
        for m in ["vector", "bm25", "hybrid"]
    },
    "q05": {
        m: "正确拒答；资料未提供精确总体百分比，claims为空，没有编造数字，source_ids不适用。"
        for m in ["vector", "bm25", "hybrid"]
    },
    "q06": {
        "vector": "按资料观点答案正确；S1提供通缩结论，S2支持物价下降与等待更划算。应理解为该文观点，不能推广为所有日本人都不借钱或唯一经济原因。",
        "bm25": "按资料观点答案正确；S1/S2支持工资与通缩，S3支持等待回报。答案把不同因素压缩在同一因果句中，表达略笼统；不视为外部经济事实验证。",
        "hybrid": "按资料观点答案正确；S2/S3支持通缩导致等待有回报的解释，未证明日本所有人都不借钱。",
    },
    "q07": {
        m: "正确拒答；所提供原文与NBA冠军问题无支持关系，claims为空，未借用模型外部知识作答。"
        for m in ["vector", "bm25", "hybrid"]
    },
}
lines = [
    "\n## 实际上下文、回答与人工核验",
    "\n依据同一批次 results.jsonl；人工核验按保存的原文进行，不修改模型原答或预填gold。耗时单位为秒。检索耗时是客户端采集候选时间；hybrid记录顺序采集两路耗时之和，不是并发耗时，也未单独计入融合耗时。回答耗时含原文回查、sources构建、模型调用与输出校验，不含已缓存的检索时间。单次测量不能代表稳定性能。\n",
]
for q in ["q02", "q01", "q03", "q04", "q05", "q06", "q07"]:
    lines.append(f"### {q}\n")
    for r in [x for x in rs if x["case_id"] == q]:
        m = r["mode"]
        a = r["answer"]
        ids = (
            "；".join(
                f"{s['source_id']}=({s['document_id']}, {s['chunk_id']})"
                for s in a["sources"]
            )
            or "无"
        )
        lines.extend(
            [
                f"**{m}**\n",
                f"- 实际 sources：{ids}",
                f"- status：`{a['status']}`",
                "- claims / source_ids：",
            ]
        )
        lines.extend(
            [
                "```json",
                json.dumps(a["claims"], ensure_ascii=False, indent=2),
                "```",
                f"- 人工判断：{reviews[q][m]}",
                f"- 检索耗时：{r['retrieval_seconds']:.4f}；回答耗时：{r['answer_seconds']:.4f}\n",
            ]
        )
report = p / "report.md"
text = report.read_text()
marker = "\n## 实际上下文、回答与人工核验"
text = text.split(marker)[0]
report.write_text(text.rstrip() + "\n" + "\n".join(lines) + "\n")
assert len(rs) == 21 and all(q in reviews for q in [r["case_id"] for r in rs])
logger.info("Added 21 mode records with sources, model output, manual review, timings")
