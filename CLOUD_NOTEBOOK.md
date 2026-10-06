# 云端（2× L20）实验记录

计划见 `CLOUD_PLAN.md`。这里只记云端的事，台式机负责合并到 `NOTEBOOK.md` / `RESULTS.md`。分支 `cloud-l20`。

## 环境

- 2× NVIDIA L20（各 48 GB），驱动 580.126.09，CUDA 13.0；32 核，247 GB 内存。
- 盘：系统盘 100 GB（剩 67 GB）；500 GB 的 `/dev/nvme1n1` 经用户同意格式化为 ext4，挂在 `/data`（已写入 `/etc/fstab`）。`HF_HOME=/data/hf_cache`，`COT_DATA=/data/datasets`。
- Python 3.12.3，uv 建 `.venv`，版本按 `CLOUD_PLAN.md` 第 3 节和笔记本分支的 `laptop_requirements.txt`（台式机环境清单）。
- 租期：48 小时（10-06 起）。

## 代码移植（P0 第 1、2 步）

- `paths.py`：`COT_DATA`（默认 `D:\datasets`）、`HF_HOME`（默认 `D:\hf_cache`）。替换了 `run_app_memory.py`、`analyze_apps.py`、`explore_extmem.py`（3 处）、`explore_longconv.py`、`explore_probe.py` 的写死路径。台式机不设变量时路径不变。
- `probe.load`：`device_map` 读 `COT_DEVICE_MAP`（默认 `cuda`）。只有 32B 的任务在队列里设 `auto`；4B 和 14B 判分器仍是单卡，和台式机一致。
- `app_common.MODELS` 加 `Qwen3-32B`（bf16）。
- 下载脚本 `cloud/fetch_cloud.py`：台式机钉过的 revision 照用（来自 `laptop_fetch.py`），其余取当前 main 并打印 commit。

## 队列

- 脚本在 `cloud/`：`queue_cloud.sh`（`queue15.ps1` 的 bash 版）、`resume_queue.sh p0|p1|p2`、`pause_queue.sh`。
- 任务清单：`cloud/jobs_p0.txt`（移植验证）、`cloud/jobs_p1.txt`（32B 原模型）、`cloud/jobs_p2.txt`（长测试集，代码还没写）。
- 日志：`logs/queue_cloud.log`，每个任务一个 `logs/<name>.log`。

## 预注册

**统一门槛**（照抄 4B 的复现规则，开跑前写定）：在一个测试集上，倒序（或检索序）比正序低 ≥10 个点，并且配对 bootstrap 95% CI 不含 0，就算"在 32B 上复现"。留出集上同时报告整群 bootstrap：ConvoMem 按人设，PersonaMem 按共享上下文，MemConflict 按用户，LoCoMo 按对话。没复现时写清原因（题目不需要历史 / 正序本身做不好的地板效应 / 真反例），附上支撑判断的数字。

**移植验证（P0）**：Qwen3-4B 上 `run_app_logs.py --n 100` 和 `explore_order_rule.py run`，用台式机的 `compare_reference.py` 比对 `reference/desktop/`：逐条答案一致率 ≥95%，各条件正确率差 ≤2 个点。不过就先查原因，不往下跑。

## 移植验证结果

（未开始）

## 实验结果

（未开始）

## 留出集接触记录

（无）

## 问题和更正

- 2026-10-06：台式机参照输出已推到 `main` 的 `reference/desktop/`（commit 0933b9f），已合并进 `cloud-l20`。两个文件的 sha256 和 README 不同，原因是 git 把 CRLF 换成了 LF；换回 CRLF 后哈希一致（8f3b353f…、96803347…），内容没变。
