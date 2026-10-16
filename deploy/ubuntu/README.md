# Ubuntu 两阶段离线建库

文档外归一使用 `config.yaml` 的 `chroma.concept_similarity_threshold: 0.4`：只有 `1 - cosine_distance > 0.4` 的候选进入后续判断，最终是否合并仍由 LLM 判定。文档内归一阈值保持 `0.2`。`0.4` 是本轮评估起点，需结合日志中的候选数、匹配率和具体匹配结果评估漏合并情况；归一日志会记录该候选阈值。

新增业务入口 `src/indexer_two_stage.py`，原 `src/indexer.py` 保持原样。只建立 SQLite/Chroma，不启动或连接 ES。

- **第一阶段并发**：doc → chunk → knowledge → concept → 文档内归一。默认 2 个文档线程，单文档内部顺序执行；每个线程同时最多一个 SDK 请求。输出逐文档 JSON 和评估包，**不创建 SQLite/Chroma，不自动启动第二阶段**。
- **第二阶段串行**：按文章文件名排序，文档外归一 → SQLite 保存 → Knowledge/Concept 向量写入 → 关系提取及证据保存。第一篇从空概念库开始；下一篇能查询前一篇已入库的标准概念。直接消费第一阶段结果，不重跑知识提取、概念提取或文档内归一。

## 上传并运行第一阶段

先将新增业务代码和部署改动 commit/push 到 Git。默认拉取 `https://github.com/zifeng-yu/neural-garden.git` 的 `master`。部署包只带启动脚本和私有 `.env`，不含业务代码。

文章不随 Git 提交。请先创建 `/home/admin/pilot/`，将要处理的 UTF-8 `.md` 文章直接上传到该目录（不递归扫描子目录），上传完成后再启动。默认读取此路径，可通过 `NG_PILOT_DIR` 指定其他绝对路径。

第一阶段准备时将文章复制到本轮 `runs/运行编号/project/data/pilot/` 并记录 hash；旧 Git 快照中的 pilot 内容不会混入。第二阶段和结果导出均使用这份快照，之后修改上传目录不会影响本轮。

重新开始时，保留 `/home/admin/ng_run/tools/`、`.env` 以及外部 `/home/admin/pilot/`。上传本机 `dist/ng-ubuntu-run.tar.gz` 到 `/home/admin/ng_run/` 后执行：

```bash
cd /home/admin/ng_run
# 保留已有 .env，只解压脚本和说明。
tar -xzf ng-ubuntu-run.tar.gz run.sh runner.py status.sh README.md
bash run.sh phase1 --workers 2
tail -f /home/admin/ng_run/latest.log
```

默认后台运行，SSH 断开不影响任务。Ctrl+C 只退出 tail 查看。下载超时默认 300 秒，依赖下载并发默认 4，可通过 UV_HTTP_TIMEOUT/UV_CONCURRENT_DOWNLOADS 覆盖。workers 允许 1–8；并发会增加瞬时 API 请求量，默认用 2，出现限流再调整。复用 tools 的下载缓存，固定使用受管理的 Python 3.12.10；首次可能需要下载该补丁版本。无 sudo 权限时先安装 git/curl/ca-certificates/tar/util-linux/coreutils。

监测命令：

```bash
bash /home/admin/ng_run/status.sh
cat /home/admin/ng_run/latest-run/status.json
```

阶段状态：prepared、probe_passed → phase1_running → phase1_ready_for_review → **awaiting_phase1_review**。安装依赖时 status.json 尚未生成；看 latest.log。

第一阶段完成后：

```bash
cat /home/admin/ng_run/latest-run/PHASE1_READY.txt
```

该文件写明 `exports/phase1-review-运行编号.tar.gz` 的下载路径，旁边有 `.sha256`。**PID 退出在此时属于正常行为，不是失败。**任务不会自动续跑。将评估包下载发给我，检查日志、失败/跳过率、概念和耗时后，再手动执行第二阶段。

评估包包含：logs/app.log、logs/error.log、phase1-report.json、skipped-chunks.json、每篇中间结果 phase1/*.json、pilot 原文、输入 hash、模型探测、代码版本及配置快照。不含 API Key/.env/.venv。phase1-report.json 有每篇 chunk 数、knowledge 数、归一概念数、跳过数、耗时和按固定顺序排列的文件清单。

**不要删除 runs/、tools/ 或 .env，也不要修改本轮 project/config.yaml 和中间结果。**第二阶段需要这些内容。

## 评估确认后，手动运行第二阶段

```bash
cd /home/admin/ng_run
bash run.sh phase2
tail -f /home/admin/ng_run/latest.log
```

默认继续 latest-run 指向的第一阶段。若存在多轮结果，显式指定已评估的目录更稳妥：

```bash
bash run.sh phase2 --run /home/admin/ng_run/runs/具体运行编号
```

第二阶段不拉新代码、不覆盖配置，沿用第一阶段保存的代码、Python 环境和 runner 副本。开始前核对原文 hash、中间文件 hash、跳过记录、配置和依赖锁文件。只有第一阶段完整且评估包已生成，第二阶段才允许启动。人工确认通过“执行 phase2 命令”表达；脚本不会自行判断模型语义质量。

完成后生成 SUCCESS.txt，status.json 为 complete：

```bash
cat /home/admin/ng_run/latest-run/SUCCESS.txt
```

下载 `exports/neural-garden-运行编号[-with-skips].tar.gz` 及 `.sha256`。最终包包含：data/sqlite/neural.db、完整 data/chroma/、data/pilot/、两阶段日志、跳过清单、最终数据校验及第一阶段评估报告。SQLite 使用 backup API 导出，Chroma 在工作进程退出后复制；逐文件校验和写在 SHA256SUMS 中。

## 失败及重跑边界

知识提取失败（包含缺少 keywords、JSON 解析错误、模型拒绝和请求异常）记录日志并保留 chunk，继续运行。没有 knowledge 的 chunk 不提取概念。跳过记录用于最终验证，结果包带 -with-skips，需人工评估其影响。

概念提取、文档内归一等其他第一阶段错误会将该文档标为失败，其他已提交的文档任务继续完成；最终第一阶段失败，第二阶段被阻止，仍尽量生成 `phase1-review-运行编号-failed.tar.gz` 供诊断。准备环境或探测失败时尚无评估包，直接看 latest.log/FAILED.txt/failure.json。

第二阶段 embedding 或关系提取异常仍中止；失败数据保留，不伪装成成功。**目前不自动续跑半成品、不自动清理旧结果**：第一阶段重跑创建新轮次；第二阶段发现已有 data/ 会拒绝追加，避免重复或不完整入库。若第二阶段失败，先保留本轮中间结果和日志，再决定如何从第一阶段结果重建，无需先重新提取全文。

已有 OpenClaw 监控规则需要同步：status=awaiting_phase1_review 且存在 PHASE1_READY.txt、没有活动 PID，是“第一阶段完成、等待评估”；只通知一次评估包路径，不应报“进程消失失败”。phase2_running、indexed、verified 等仍表示后续处理中，只有 SUCCESS.txt 表示最终建库完成。

## 路径和日志

```text
/home/admin/ng_run/
├── .env                       # 私有 API_KEY，权限 600
├── run.sh / runner.py / status.sh
├── tools/                     # Python、uv 和依赖缓存
├── repository.git/            # Git 源码缓存
├── logs/                      # 启动日志
├── runs/<运行编号>/
│   ├── project/               # 固定 commit 的代码、配置、.venv
│   ├── runner.py              # 本轮部署驱动快照
│   ├── phase1/*.json          # 文档内归一后、外部归一前的中间数据
│   ├── phase1-report.json / PHASE1_READY.txt
│   ├── status.json / skipped-chunks.json / logs/
│   ├── data/sqlite/neural.db  # 仅第二阶段创建
│   ├── data/chroma/           # 仅第二阶段创建
│   └── SUCCESS.txt            # 仅最终校验打包成功后创建
└── exports/                   # 第一阶段评估包 + 最终索引包
```

项目配置中的数据路径都改成本轮绝对路径，不污染其他轮次。日志使用 DEBUG，禁用按大小轮转，保留整轮过程；会增加磁盘占用。SDK 调用设置 180 秒超时，embedding 失败最多尝试 3 次。日志写入时对当前 API Key 脱敏。网络、额度和服务端故障无法预先保证消除。

`INDEX {JSON}` 日志记录阶段起止、source、适用时的 chunk_no/chunk_id、seconds、outcome。LLM 请求记录模型、HTTP 状态、request_id 和接口提供的 token 数。文档外归一记录候选数、是否调用 LLM、是否匹配和是否改名。父阶段耗时包含子阶段，评估时按 stage 分组，不要跨层级相加。并发阶段的总墙钟时间以 phase1-report.json 的 seconds 为准，不能累加各文档耗时当作总时间。

日志按两类用途分析，可共用 `logs/app.log`：

- **运行与数据健康（runtime）**：阶段起止、异常类型、HTTP 错误与重试；结合 `status.json`、`failure.json`、`skipped-chunks.json`、`validation.json` 判断整轮是否完成、哪些 chunk 被跳过、SQLite/Chroma 是否一致。`error.log` 是错误级别的副本，不应和 app.log 重复计数。`awaiting_phase1_review` 是正常等待状态；某次 LLM 请求失败或 knowledge 被跳过，不等于整轮失败。
- **系统评估（evaluation）**：文档/chunk 提取数量、归一前后概念数量、外部候选阈值和命中数、LLM 匹配结果、关系数量及无效端点数、请求耗时和 token 用量。结合第一阶段中间 JSON、原文和最终数据库检查提取与合并质量。结构完整性校验通过不代表语义质量已合格。

`INDEX` 结构化事件带 `log_categories: ["runtime", "evaluation"]`，同一条事件可服务两类分析；按 `stage`、`event`、`outcome` 和具体指标筛选。原有第三方/文本日志仍保留。下载第一阶段评估包或最终结果包即可，不必手动拆分日志。

建议至少 4GB 内存、10GB 可用磁盘；prepare 要求 5GiB 剩余空间。Git/uv/PyPI/DashScope 需可访问；uv 使用[官方安装器](https://docs.astral.sh/uv/getting-started/installation/)。支持 Ubuntu 22.04/24.04 的 x86_64/ARM64。可通过 NG_BASE_DIR、NG_REPO_URL、NG_BRANCH 更改默认值，私有仓库请先配置凭据，不在 URL 中放 token。

## 本地使用最终数据

下载包后：

```bash
shasum -a 256 -c 具体包名.tar.gz.sha256
mkdir ng-import
tar -xzf 具体包名.tar.gz -C ng-import
cd ng-import
shasum -a 256 -c SHA256SUMS
```

停止本机数据库使用进程并备份旧数据，使用包内相同 commit/uv.lock，整体替换 SQLite 和 Chroma（不要合并新旧向量目录）。server-config.yaml 是服务器路径快照，不覆盖本机配置。跨平台 Chroma 文件需在目标机验证读取和搜索。新库 ID 会重新分配，本机 ES 使用 `uv run python -m src.es --mode reset` 全量重建，不能用 upsert 保留旧记录。
