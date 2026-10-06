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

10-06 10:13–10:27，Qwen3-4B bf16 单卡（GPU0），`cloud/jobs_p0.txt`。

| 测试 | 逐条一致率（门槛 ≥95%） | 各条件正确率最大差（门槛 ≤2） | 判定 |
|---|---|---|---|
| 倒序日志 `run_app_logs.py --n 100`（1800 条） | 97.6% | 1.7（git 倒序 70.7 → 69.0） | **通过** |
| E15 `explore_order_rule.py`（900 条） | **94.8%**（853/900，差 2 条） | 1.3（earliest_first 倒序 16.0 → 17.3） | **按预注册规则：不通过** |

E15 各条件：current 正序 44.7/44.7、倒序 41.3/42.0；earliest 正序 72.7/73.3、倒序 23.3/22.7；earliest_first 正序 78.7/79.3、倒序 16.0/17.3（台式机/云端）。

**查原因**（`cloud/diag_p0_e15.py`）：
- 47 条不同的回答里，prompt 的 token 数全部一致（输入相同）；23 条只是措辞不同、抽出的答案位置相同，24 条答案位置不同；正确与否变化的 14 条，方向两边都有。
- 在两边回答第一次分叉的那一步，用云端模型算两个 token 的 logit 差：中位数 0.0，48% 的差 <0.5，最大 2.9。作为对照，回答一致的条目第一个生成 token 的 top1−top2 差中位数是 8.9（10% 分位 1.5）。也就是说分叉都发生在几乎打平的地方。
- 而且在分叉点做一次完整前向（不用 KV cache），云端自己生成时选的 token 也只有 54% 是 top-1：同一台机器换一种计算路径就会翻。这是 bf16 下贪心解码的数值噪声，不是移植错误。
- 环境：所有包版本和台式机清单一致；显卡从 RTX 5080（Blackwell）换成 L20（Ada），SDPA 内核不同。
- 结论：按规则 E15 没过（差 0.2 个点）；诊断显示原因是数值噪声。**门槛不改**，这里如实记为不通过。
- **10-06 用户决定：接受，进入 P1（32B）。**

## 硬件检查（10-06）

- 两张 L20 都是 PCIe Gen4 x16（空闲时降到 Gen1，有负载时恢复 Gen4），拓扑 PHB，没有 NVLink，P2P 不可用（虚拟机）。GPU0→GPU1 拷贝 15.6 GB/s，GPU→CPU 20 GB/s。
- bf16 矩阵乘每卡 117 TFLOPS（L20 标称约 119.5），显存拷贝 706 GB/s；功耗上限 350 W（满额），持久模式开，CPU 调频 performance。没有降频或配置问题。
- 32B 用 `device_map=auto` 按层拆到两张卡（流水线）：卡间每次前向只传一份隐藏状态（25k token 约 250 MB，约 16 ms），带宽不是瓶颈；RDMA 是跨机器用的，单机两卡用不上。代价是两张卡轮流算、不同时算。
- SDPA 的 flash / efficient / cuDNN 后端都可用；按计划保持台式机的 `LONG_SDPA` 顺序，不改。

## 判分器

- Qwen3-14B 原模型 4-bit nf4，现成模型，不是我们训练的。每条只做一次前向，比较 A/B/C 三个字母的概率。输入是旧证据、新证据、问题、标准答案和待判回答，不看原始长对话。
- 32B 的结果也用它判：判分器手里有标准答案，任务是对照，不需要比被判的模型强；和台式机所有结果用同一个判分器，结果才可比。换判分器等于多改一个变量。
- 已知弱点（NOTEBOOK）：只答一个裸值的回答容易被判 C。32B 判完后报告每个模型的回答长度分布和判分器与字符串匹配的一致率，看 32B 的作答风格会不会让判分偏。

## 实验结果

（未开始）

## 留出集接触记录

（无）

## 问题和更正

- 2026-10-06：台式机参照输出已推到 `main` 的 `reference/desktop/`（commit 0933b9f），已合并进 `cloud-l20`。两个文件的 sha256 和 README 不同，原因是 git 把 CRLF 换成了 LF；换回 CRLF 后哈希一致（8f3b353f…、96803347…），内容没变。
