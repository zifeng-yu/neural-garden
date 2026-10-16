# Ubuntu 离线建库

在 Ubuntu 22.04/24.04 的 x86_64 或 ARM64 服务器上，从 Git 拉取项目，测试 DashScope，再建立 SQLite/Chroma。**不启动、不连接、不同步 Elasticsearch。**

默认仓库 `https://github.com/zifeng-yu/neural-garden.git`，分支 `master`。每次抓取分支最新提交，并记录确切 commit。本文中的成功表示技术性校验通过，不保证模型抽取的语义质量。

## 上传与运行

部署包内包含 `run.sh`、`runner.py`、本说明和仅有 `API_KEY` 的私有 `.env`。私有部署包位于本地被 Git 忽略的 `dist/`，不要加入 Git。

服务器先创建工作目录。普通用户运行时：

```bash
sudo mkdir -p /home/admin/ng_run
sudo chown "$(id -u):$(id -g)" /home/admin/ng_run
chmod 700 /home/admin/ng_run
```

将本地 `dist/ng-ubuntu-run.tar.gz` 上传到服务器，例如：

```bash
scp dist/ng-ubuntu-run.tar.gz admin@服务器IP:/home/admin/ng_run/
```

服务器执行：

```bash
cd /home/admin/ng_run
tar -xzf ng-ubuntu-run.tar.gz
bash run.sh
```

默认后台运行，断开 SSH 后继续。启动成功不等于建库成功。查看：

```bash
tail -f /home/admin/ng_run/latest.log
cat /home/admin/ng_run/latest-run/status.json
cat /home/admin/ng_run/latest-run/SUCCESS.txt
```

`SUCCESS.txt` 只在校验及打包完成后出现。错误会使脚本非零退出，`FAILED.txt` 标明失败；Python 阶段另有 `failure.json`。失败的工作目录保留，不生成完成的数据包。

若只想先验证环境和网络：

```bash
bash run.sh --foreground --check-only
```

测试会真实调用一次 qwen3.5-plus 和一次 embedding（失败可能重试），使用现有 API Key 的额度。检查成功后再 `bash run.sh` 会新建正式运行目录。锁禁止同一目录内同时启动两个任务。脚本若需要安装系统基础工具，可能在启动时请求 sudo；无 sudo 权限时请先安装 `git curl ca-certificates tar util-linux`。

## 路径与运行方式

```text
/home/admin/ng_run/
├── .env                        # 私有 API_KEY，权限 600；不作为 shell 执行
├── run.sh / runner.py
├── repository.git/             # 只缓存 Git 代码
├── tools/                      # uv、受管理的 Python、依赖下载缓存
├── logs/                       # 启动日志
├── runs/<UTC时间-PID>/
│   ├── project/                # 指定 commit 的独立副本、.venv、文章
│   ├── data/sqlite/neural.db
│   ├── data/chroma/            # knowledge / concept / insight
│   ├── logs/                   # 项目日志，包括 embedding 耗时
│   ├── status.json / probe.json / validation.json
│   ├── skipped-chunks.json     # 模型判为无效的 chunk 清单
│   └── SUCCESS.txt             # 完成标志及下载路径
└── exports/                    # 最终 .tar.gz 与 .sha256
```

配置只修改本次 project 副本，所有数据库、日志路径均指向本次 runs 目录。锁定依赖通过 `uv sync --locked --no-dev --python 3.12` 安装。uv 使用[官方安装器](https://docs.astral.sh/uv/getting-started/installation/)，不会复制 Mac 的虚拟环境。

这是**每次独立全量建立**，不复用现有本地库、不删除旧运行目录、不自动续跑半成品。重跑会重新调用模型并计费；这样避免当前 indexer 的同 hash 跳过机制掩盖失败。默认沿用仓库的概念阈值和分块策略，不做性能调参。

技术性知识提取失败直接中止；embedding 最多尝试 3 次，关系返回结构不合法也中止。只有 `ContentNotExtrator`（模型明确判定无效文章）允许跳过：保留原文，记录 hash 和块号，生成包名称带 `-with-skips`。这些跳过不代表原文确实无效，应下载后复核。

SDK HTTP 调用设置 180 秒超时。日志记录失败和 embedding 耗时，不打印 API Key。网络、额度或第三方服务故障不能由脚本保证消除；脚本保证检测失败，不用完成标志掩盖错误。

建库后检查文章集合和原文 hash/分块、SQLite integrity/外键/关联、缺失知识单元与明确跳过记录对应、关系证据端点，以及 SQLite/Chroma 的 ID、文本、metadata、向量。校验进程退出后才复制整个 Chroma 目录，SQLite 使用 backup API 导出，避免遗漏 WAL 数据。建议至少有 4GB 内存和 10GB 可用磁盘，实际资源需求随文章增长；prepare 阶段会要求至少 5GiB 剩余空间。

如使用私有仓库，先在服务器配置 Git 凭据；不要把 Git token 拼入仓库 URL。可以用 `NG_REPO_URL`、`NG_BRANCH`、`NG_BASE_DIR` 覆盖默认值；改 origin 时请使用新的工作根目录。

## 下载及本地使用

先从 `SUCCESS.txt` 取得具体文件名，在本机执行：

```bash
scp admin@服务器IP:/home/admin/ng_run/exports/neural-garden-具体时间.tar.gz ./
scp admin@服务器IP:/home/admin/ng_run/exports/neural-garden-具体时间.tar.gz.sha256 ./
shasum -a 256 -c neural-garden-具体时间.tar.gz.sha256
mkdir ng-import
tar -xzf neural-garden-具体时间.tar.gz -C ng-import
cd ng-import
shasum -a 256 -c SHA256SUMS
```

包中有 `data/sqlite/`、完整 `data/chroma/`、语料 `data/pilot/`、commit、依赖版本、校验报告和跳过清单；**不含 API Key、.env、.venv**。`server-config.yaml` 仅供溯源，里面是服务器绝对路径，不要覆盖本机配置。

在本机停止所有使用数据库的进程，备份旧数据，按包内 commit/uv.lock 安装相同版本依赖，然后**整体替换** SQLite 和 Chroma 两套数据（不要将 Chroma 合并到旧目录）。先在本地验证读取与搜索；服务器与本机架构不同，Chroma 目录跨平台可用性仍须在目标机实际验证，并保留原包用于恢复。

这是新建语料，ID 会重新分配，本地 ES 必须全量重建：

```bash
uv run python -m src.es --mode reset
```

不要用 `upsert` 保留旧 ES 记录。完成本地验证前保留旧数据库备份。
