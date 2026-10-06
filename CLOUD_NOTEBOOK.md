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

**P2 · MemoryAgentBench-CR 32k 版**（10-06 写定，开跑前）：
- 数据：`factconsolidation_sh_32k`，100 题 × 正序 / 倒序 / 检索序（BM25），每条约 37.5k token（Qwen3 上限 40960）。代码：`explore_extmem.py --tasks mabcr32k`，构造和 6k 版完全相同，只换 source；默认任务列表不变。字符串判分，不用判分器。
- 主分析：答案事实恰有一条更早冲突事实的题（58/100，6k 版是 76/100），沿用 6k 版的规则。其余 42 题单独报告。
- 门槛：统一门槛（倒序或检索序比正序低 ≥10 个点，配对 bootstrap 95% CI 不含 0）。4B 和 32B 分别判定。
- 不是留出集。已知风险：先验冲突造成的地板效应（4B 在 6k 上正序 83% 选了旧值）。正序本身很低时，按"地板效应"报告，附正序正确率和选旧值比例。

**P2 · BEAM 知识更新**（10-06 用户同意下载并按此方案做；开跑前写定）：
- 数据：HF `Mohammadta/BEAM`（论文 2510.27246，ICLR 2026，CC-BY-SA-4.0），`100K` / `500K` / `1M` 三个 split，共 90 段对话；不用 `BEAM-10M`。**定为冻结留出集**：只看汇总；格式检查时看过的 1 个对话已记在接触记录里。
- 题目：每段对话的 `knowledge_update` 题（每段 2 道，最多 180 道）。旧信息 `source_chat_ids.original_info`、新信息 `updated_info`。旧、新在同一个 session（同一日期）的题剔除，报告剔除数。
- 构造（仿 MemConflict）：单位是一轮"用户消息 + 助手回复"；同一 session 里相邻的几轮合成一块，块头标该 session 的日期（`### Conversation (date: ...)`）。旧信息块 = 旧证据所在轮及前后各 1 轮；新信息块同理；再从新信息所在 session 往前，每个中间 session 取中间 3 轮作一块，直到总长约 20k token（超了就停）。旧块 + 新块本身超过 20k 的题剔除。prompt 措辞、"Current date"（新信息日期 + 1 天）、64 token 上限都照 MemConflict。
- 条件：正序 / 倒序 / 检索序（`explore_extmem.bm25_order`，问题作查询）。
- 判分：14B 4-bit 判分器，用 `run_app_memory.JUDGE`（A 新值 / B 旧值 / C 都不是），旧 / 新证据用题目自带的 `conversation_references`（两条时）或证据消息原文前 400 字。主指标 = A 的比例；同时报告 B（选旧值）。
- 门槛：统一门槛（倒序或检索序比正序低 ≥10 个点、配对 bootstrap 95% CI 不含 0），另报按对话的整群 bootstrap。各模型分别判定。没复现时按地板效应 / 题目不需要历史 / 反例分类，附数字。

**vLLM 推理引擎**（10-06 用户同意试，开跑前写定）：
- 做法：vLLM 装在单独的虚拟环境（`/data/vllm-venv`），作为服务运行（`cloud/vllm_ctl.sh`）。实验脚本仍在原环境，prompt、chat 模板、分词、输出解码都不变；`probe.greedy` 把同样的 token id、长度上限和停止集合发给服务（`vllm_client.py`）。解码规则照搬 HF 版：每步取最大、停止 token 保留、忽略模型自带的 generation_config。只用于原模型 bf16 生成；判分器（14B 4-bit）和 CoT 打分（`run_ext_eval --benches cot`，要逐个候选的对数概率）仍用 HF。
- 验证门槛（和 P0 同一把尺子）：vLLM 跑 Qwen3-4B（单卡）的倒序日志和 E15，用 `compare_reference.py` 和台式机比：逐条一致 ≥95%，各条件正确率差 ≤2 个点。
- 速度：同一批 10 条长 prompt（MemConflict），32B 上 HF（两卡流水线）和 vLLM（两卡张量并行）各跑一遍。
- 决定规则：两项验证都通过、并且实测快 ≥1.5 倍，32B 的所有生成任务都用 vLLM（每个测试集内所有条件同一个引擎，不混用）；否则照原计划用 HF。

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

## vLLM 验证结果（10-06）

vLLM 0.31.0（torch 2.13 + cu130，`/data/vllm-venv`），Qwen3-4B 单卡（GPU1），通过 `COT_ENGINE=vllm` 跑和 P0 相同的两项，`compare_reference.py` 与台式机比。输出存在 `results/vllm_check/`。

| 测试 | 逐条一致（≥95%） | 各条件最大差（≤2） | 判定 |
|---|---|---|---|
| 倒序日志（1800 条） | 97.2% | 2.0（git 正序 75.0 → 73.0） | **通过**（刚好在门槛上） |
| E15（900 条） | 95.3% | 0.7 | **通过** |

和云端 HF 的 P0（97.6% / 94.8%）相比，vLLM 和台式机的一致程度相当，E15 还略高。速度：倒序日志约 10 条/秒，HF 约 6 条/秒（4B 单卡、短 prompt）。
- 启动问题：flashinfer 启动时要编译内核，需要 venv 里的 ninja 和 CUDA 13.0 工具链；`cloud/vllm_ctl.sh` 已加 PATH / CUDA_HOME。
- **长 prompt 补充检查结果**（MAB-CR 32k，4B，vLLM 对本机 HF，300 条）：逐条一致 **94.0%**（门槛 ≥95，**不通过**）；各条件正确率 HF/vLLM：正序 39.0/39.0，倒序 39.0/38.0，检索序 54.0/52.0，最大差 2.0（门槛 ≤2，通过）。按预注册规则：长 prompt 的测试集不用 vLLM。输出在 `results/vllm_check/extmem_Qwen3-4B.jsonl`。
- **10-06 用户决定（推翻上面的预注册规则）：所有生成任务都用 vLLM，包括长 prompt。** 理由（用户）：效率优先；每个模型自己的条件之间是同一个引擎，自己和自己比；之后可能还要跑 70B。为了让模型之间也可比，**云端用 vLLM 重测 4B 和 14B**（14B 当答题模型用 bf16，`Qwen3-14B-bf16`；判分器仍是 HF 的 14B 4-bit）。论文里 32B 与台式机 4B/14B 的结果比较时要注明引擎不同；云端内部的模型比较都是 vLLM。
  - 结果文件名带 `~vllm`（`app_common.tag`），不覆盖 HF 结果。
  - 例外：CoT 打分（`--benches cot`）要候选的对数概率，仍用 HF；它在每个模型内部的条件之间也是同一个引擎。
  - 队列：`cloud/jobs_v32.txt`（32B，两卡张量并行）→ `cloud/jobs_v4.txt`（GPU0 :8000）和 `cloud/jobs_v14.txt`（GPU1 :8001）并行 → `cloud/jobs_judge.txt`。`jobs_p1.txt` 作废。
  - 4B 的 MAB-CR 32k vLLM 结果（上面的长 prompt 检查）直接作为 `extmem_Qwen3-4B~vllm.jsonl` 的一部分，不重跑。
- 还差：32B 的速度对比（等 32B 下完）。脚本 `cloud/speedtest_32b.py`：10 条 MemConflict 正序 prompt（按长度均匀取，约 8k–25k token，每条上下文不同，vLLM 的前缀缓存帮不上忙），HF 和 vLLM 各跑一遍，不写 results/。
- 判分器冒烟测试（10-06）：14B 4-bit 加载 6 s、显存 10.6 GB，三条合成样例（新值 / 旧值 / 不知道）分别判 A / B / C，每条约 0.1 s。
- **长 prompt 补充检查**（10-06 11:35 写定，开跑前）：上面两项的 prompt 都短（E15 中位 1k token）。用 vLLM 把 4B 的 MAB-CR 32k（300 条，每条约 37.5k token）再跑一遍，和本机 HF 的结果（`results/extmem_Qwen3-4B.jsonl`）逐条比。标准同 P0：逐条答案一致 ≥95%，各条件（正序 / 倒序 / 检索序，各 100 条）正确率差 ≤2 个点。注意每个条件只有 100 条，1 条 = 1 个点。不过就不用 vLLM 跑长 prompt 的测试集。

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

### P2 · MAB-CR 32k，Qwen3-4B（10-06 10:43–11:27，HF 单卡，字符串判分）

| 子集 | n | 正序 | 倒序 | 检索序 | 倒序−正序 [95% CI] | 检索序−正序 [95% CI] | 选旧值（正/倒/检） |
|---|---|---|---|---|---|---|---|
| 有一条更早冲突事实（主分析） | 58 | 19.0 | 15.5 | 39.7 | −3.4 [−13.8, 6.9] | **+20.7 [8.6, 32.8]** | 53.4 / 58.6 / 41.4 |
| 无冲突 | 42 | 66.7 | 71.4 | 73.8 | +4.8 [−9.5, 19.0] | +7.1 [−7.1, 21.4] | 0 / 0 / 0 |

- **按预注册门槛：4B 上没有复现。** 倒序比正序只低 3.4 个点、CI 含 0；检索序反而高 20.7 个点（方向相反）。
- 原因按预注册的分类：**地板效应**。冲突题正序只答对 19.0%，53.4% 选了旧值（6k 版台式机记录正序 83% 选旧值，同一类先验冲突）。正序本身就做不好，倒序没有往下掉的空间。无冲突题正序 66.7%，说明模型能读这么长的上下文，问题出在冲突题上。
- 检索序高出 20.7 个点是意外结果，没有预注册假设，这里只记录不解释；32B 跑完再一起看。

## 留出集接触记录

- 10-06 约 12:10：为确认 BEAM 的数据格式，通过 HF datasets-server 看了 BEAM `100K` split 第 0 个对话的原文：knowledge_update、contradiction_resolution、temporal_reasoning 各 1 道题（含答案和 source_chat_ids），以及第 1 条用户消息。BEAM 当时还没定为留出集；若之后用作留出集，这个对话要在报告里标注或剔除。

## 问题和更正

- 2026-10-06：台式机参照输出已推到 `main` 的 `reference/desktop/`（commit 0933b9f），已合并进 `cloud-l20`。两个文件的 sha256 和 README 不同，原因是 git 把 CRLF 换成了 LF；换回 CRLF 后哈希一致（8f3b353f…、96803347…），内容没变。
