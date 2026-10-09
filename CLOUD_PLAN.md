# 云端（2× L20）测试计划

写于 2026-10-05 晚，台式机会话。读者：在租用机器上通过 SSH 运行的新 Claude Code 会话。

## 0. 先读

1. `CLAUDE.md`：用户的规则，不可更改。
2. `NOTEBOOK.md`：当前状态。至少读 §3、§4b、§6、§9、§10。
3. 本文件。
4. 上级目录规则（台式机 `pro/CLAUDE.md`）已抄在下面第 1 节最后一条，云端没有这个文件。

用中文回复用户，如实报告，包括负面结果。

> **10-08 晚更新**：租期最后一晚的排程见第 5 节 **P8**（14B / 32B 开思考的 MemConflict，06:30 停 GPU，留一小时上传）。

## 1. 规则（摘自 CLAUDE.md，云端同样适用）

- **不在测试数据上调东西。** 门槛先写进 `EXPLORE_PLAN.md`（云端写进 `CLOUD_NOTEBOOK.md` 的"预注册"一节），再开跑。对照组只差一个变量。
- **诊断工具不是修复。** 探针、oracle 干预只用于解释。
- **留出集**：ConvoMem 人设 50–99、ConvoMem 长版、PersonaMem、MemConflict、LoCoMo 都冻结。LongMemEval 已看过，作为修复证据时要标注。MemoryAgentBench-CR **不是**留出集。
- **永远不打印 LongMemEval 的对话原文，只打印汇总。** 留出集同样只看汇总；看过原文要记进 `CLOUD_NOTEBOOK.md` 的"留出集接触记录"。
- **每个实验结束后更新记录**，然后 commit 并 push（见第 6 节，推到 `cloud-l20` 分支，不推 main）。
- **修复的评测照搬主会论文的做法**，必须包括"不伤原模型"的检查：MMLU / ARC-C / HellaSwag、GSM8K、IFEval；长上下文训练还要加 LongBench。门槛要写进预注册。训练数据的作答格式必须多样化，**并且"要不要写推理"也要听提示**。

## 2. 这次在云端做什么，不做什么

**做**（都不需要训练）：

1. 移植代码，并在 4B 上复现几个台式机结果，确认移植没出错（P0）。
2. 在 **Qwen3-32B 原模型（bf16）** 上确认现象：顺序效应、检索序、E15（P1）。这是论文规模部分的证据。
3. 跑台式机 16 GB 显存装不下的长测试集：MemoryAgentBench-CR 32k 版、BEAM（P2）。
4. 时间够的话，换一个模型家族：OLMo-2-32B（P3，先问用户）。

**不做：训练 32B。** 原因：

- 目前的训练配方有已知缺陷。E17 训练目标里推理清单是必写的，模型不听"只答字母"。PersonaMem 因此比原模型低约 14 个点，其中约 6 个点是清单写不完、约 8 个点是方法用错，详见 NOTEBOOK「PersonaMem 下降的诊断」。
- 新配方（E18）由台式机会话在 4B 上设计和验证，**过了所有门槛后**才会通知云端训 32B（P4）。
- 没有用户明确指示，不要开始 P4。

## 3. 环境

- **仓库**：`git clone` 私有仓库 `kirkz7/cot-overwrite`。GitHub 认证由用户自己完成，不要替用户输入 token。然后 `git checkout -b cloud-l20`。
- **Python**：用 uv 建 `.venv`。版本按台式机固定：

  | 包 | 版本 |
  |---|---|
  | torch | 2.11.0（CUDA 12.8 版） |
  | transformers | 5.17.0 |
  | peft | 0.21.2 |
  | datasets | 5.0.1 |
  | lm-eval | 0.4.13，装 `lm-eval[ifeval,longbench]` |

  其余依赖缺什么补什么：bitsandbytes、accelerate、pandas、numpy、scipy、tqdm、scikit-learn 等。
- **HF 缓存**：设 `HF_HOME` 到一块大盘，**至少需要约 200 GB**：32B 约 66 GB，14B 约 30 GB，4B 约 8 GB，数据集若干。下载完后跑队列时设 `HF_HUB_OFFLINE=1`、`HF_DATASETS_OFFLINE=1`。
- **模型**：

  | 模型 | 用途 |
  |---|---|
  | `Qwen/Qwen3-32B` | 主角，bf16，双卡 |
  | `Qwen/Qwen3-14B` | 判分器。**必须保持 4-bit nf4**，和台式机所有已有结果一致。卡够用也不要改成 bf16，否则判分器变了，结果不可比 |
  | `Qwen/Qwen3-4B` | P0 复现用 |
  | `allenai/OLMo-2-0325-32B-Instruct` | P3，先问用户 |

- **数据**：路径目前写死在 `D:\`，见第 4 节。

  | 数据 | 来源 |
  |---|---|
  | LongMemEval | HF `xiaowu0162/longmemeval-cleaned`，**固定 revision `98d7416c24c778c2fee6e6f3006e7a073259d48f`**，文件 `longmemeval_oracle.json` |
  | ConvoMem | HF `Salesforce/ConvoMem`，用 `core_benchmark/evidence_questions/changing_evidence/2_evidence/*.json` |
  | PersonaMem | HF `bowen-upenn/PersonaMem-v1` |
  | MemoryAgentBench | HF `ai-hyz/MemoryAgentBench`，用 `data/Conflict_Resolution-*.parquet` |
  | MemConflict | 论文 2605.20926，作者仓库 TaoZhen1110/MemConflict（先查 HF，没有再用 GitHub），文件 `Data/Step4_4.jsonl`。台式机放在 `D:\datasets\memconflict\Step4_4.jsonl` |
  | LoCoMo | GitHub `snap-research/locomo` 的 `data/locomo10.json` |
  | WikiText | HF `Salesforce/wikitext` 的 `wikitext-2-raw-v1/train/0000.parquet`，revision `refs/convert/parquet`。合成数据生成器要用 |
  | lm-eval 用的数据集 | 跑一次 `python explore_general.py fetch` 即可缓存 |

- **下载前**先列出文件名、来源和大小给用户看，用户同意后再下。

## 4. 代码移植（P0，第一个 commit）

台式机的代码是为 Windows 和单张 16 GB 卡写的。要改的地方，**都必须向后兼容**：台式机不设环境变量时，行为不变。

1. **路径。** 新增 `paths.py`，从环境变量 `COT_DATA`（默认 `D:\datasets`）和 `HF_HOME`（默认 `D:\hf_cache`）拼出路径，替换下面 7 处写死的路径。HF 缓存里的路径最好改用 `huggingface_hub` 的 `snapshot_download(..., local_files_only=True)` 或 `hf_hub_download` 来取。
   - `run_app_memory.py:31`、`analyze_apps.py:68`：LongMemEval
   - `explore_extmem.py:40-42`：MemConflict、LoCoMo、MAB
   - `explore_longconv.py:31`：PersonaMem
   - `explore_probe.py:123`：ConvoMem
2. **双卡加载。** `probe.py:38` 是 `device_map="cuda"`，32B 放不下单卡。新增环境变量 `COT_DEVICE_MAP`（默认 `cuda`），云端设为 `auto`。在 `app_common.MODELS` 里加 `"Qwen3-32B": ("Qwen/Qwen3-32B", False)`，P3 时再加 OLMo-2-32B。
   - 注意：有些脚本用 `.cuda()` 或 `input_ids.cuda()` 把输入放到 `cuda:0`。`device_map=auto` 时输入要放到第一层所在的设备，通常就是 `cuda:0`，一般没问题，但要检查。
   - `torch.cuda.max_memory_allocated` 之类的统计只看一张卡。
3. **队列。** `queue15.ps1` 是 PowerShell 写的。写一个 bash 版的 `queue_cloud.sh`，语义相同：日志里已有 `end <name> exit=0` 的任务跳过；每个脚本自己也会跳过已完成的条目，所以都可以续跑。用 `nohup` 或 `tmux` 后台运行，日志放 `logs/`。不需要 `keepawake.py`。
4. **注意力核。** `explore_probe.LONG_SDPA` 指定了 SDPA 后端顺序，Linux 上应该能用。如果 flash 后端可用、更快，**不要改**，保持和台式机一致，只是速度慢一点。
5. **移植验证（门槛，开跑 32B 前必须过）**：
   - 台式机的参照输出在 `reference/desktop/`，说明、哈希和台式机环境见那里的 README。`results/` 被 `.gitignore` 忽略，所以参照文件单独放在这个目录。
   - **不要把参照文件复制进 `results/`**，否则脚本会以为已经跑完，全部跳过。
   - 在 Qwen3-4B 上重跑，输出写进 `results/`：
     - `run_app_logs.py --n 100 --models Qwen3-4B`
     - `explore_order_rule.py run --models Qwen3-4B`
   - 然后运行 `python compare_reference.py logs` 和 `python compare_reference.py e15`，逐条比对。门槛已写在脚本里：逐条答案一致率 ≥95%，各条件正确率差 ≤2 个点。贪心解码，换卡后的浮点差异可能让少数条目变化，所以不要求 100%。
   - 不过就先查原因，例如注意力后端、精度、chat template、transformers 版本，不要往下跑。

## 5. 实验（按优先级）

**统一的门槛**（照抄 4B 的复现规则，开跑前写进 `CLOUD_NOTEBOOK.md`）：在一个测试集上，倒序（或检索序）比正序低 ≥10 个点，并且配对 bootstrap 95% CI 不含 0，就算"在 32B 上复现"。留出集上同时报告整群 bootstrap：ConvoMem 按人设，PersonaMem 按共享上下文，MemConflict 按用户，LoCoMo 按对话。没复现时必须写清原因，例如题目不需要历史、正序本身做不好（地板效应），或者真是反例，并附上支撑这个判断的数字。

### P8 · 租期最后一晚：14B / 32B 开思考的 MemConflict（台式机 10-08 复盘写定，北京时间 10-09，租期约 07:30 结束）

- **为什么做这一项**：
  - 新的干净留出集上，只开思考就基本去掉了顺序效应：4B −21 → −7，8B −16 → −3（CLOUD_NOTEBOOK 10-08）。
  - 但 MemConflict 上原模型 8B 开思考还是 80.4 / 63.7 / 72.5（倒序 − 正序 −16.7），E18.1 + 惩罚比只开思考高 +17.1。
  - 所以"开思考够不够、训练还有没有必要"到 14B、32B 会怎样，决定论文的说法。这件事只能在 L20 上做：bf16 的 14B、32B 放不进 5080，HF 4-bit 开思考要十几个小时。
  - 没选 32B 跑新干净留出集（云端 10-08 10:05 列的候选）：那里 4B、8B 只开思考已基本修好，32B 的结果可以预料；MemConflict 是只开思考还修不好的集，结果预料不到。
- **排程**（北京时间；时间都是估计。8B 开思考实测：思考平均约 360 token，没写完 2.8%）：

  | 时间 | GPU0 | GPU1 |
  |---|---|---|
  | 现在 → 约 02:30 | `jobs_c`：原模型 8B 思考开通用能力（不动） | `jobs_lme8`：LongMemEval 8B（用户 23:20 排的，不动；估计 04:30–05:30 结束） |
  | 约 02:30 起 | `jobs_fin0`：14B 思考开，正序 + 倒序 480 条（约 1.5–2 小时）→ 判分 | 同上 |
  | 14B 跑完，GPU1 还忙，还没到 05:00 | `jobs_fin0x`：14B 检索序 240 条 → 判分 | 同上 |
  | GPU1 在 05:00 前空出 | `jobs_fin2`（两卡 TP2）：32B 思考开，正序 + 倒序 480 条（约 1.5–2.5 小时，可能在 06:05 被截断） | （同左） |
  | 32B 在 05:20 前跑完 | `jobs_fin2x`：32B 检索序 | （同左） |
  | GPU1 到 05:00 还忙 | 不跑 32B；GPU0 续跑 14B | 不动 |
  | 06:05 | 所有生成停止 → `jobs_finj` 统一判分（parse v2） | |
  | 06:30 | GPU 工作全部结束；`jobs_lme8` 如果还在跑也停（已存的行都保留，没判分的回台式机判） | |
  | 06:30 → 07:30 | 上传（见下面的清单） | |

  - 规则（写在 `cloud/orchestrate_fin.sh` 里）：GPU1 空出时，GPU0 上还在跑的队列最多再跑 15 分钟（不超过 05:00），然后两卡都给 32B。
  - 06:05 前还剩生成时间，就续跑 14B（先补完正序 + 倒序，再补检索序）。
  - 编排不改动用户的队列，只有 06:30 收尾时会停 `jobs_lme8`。要保证 32B 能跑，就要让 GPU1 在 05:00 前空出来，这由用户决定，例如把 `jobs_lme8` 还没开始的组挪到后面。
- **设置**：
  - 和 8B 思考开（`b8_mc_t`）完全一样：同样的题和提示、Qwen3 思考采样（T 0.6、top-p 0.95、top-k 20）、每题固定种子、上限 1024、判分器 Qwen3-14B 4-bit、parse v2。
  - **不加 presence_penalty**：测的是原模型；惩罚是方案 C 给 E18.1 的部署配置。
  - 只有两处不同：
    1. 并发请求（`cloud/run_extmem_conc.py`，同时发 6 条；原脚本一次一条）。vLLM 成批计算，数值和逐条跑有微小差别，要和"引擎不同"一样注明。
    2. 题目按固定的伪随机顺序跑（题号的 crc32），同一题的各个顺序一起提交。到点停下时，跑完的是随机子样本，每题的顺序都齐全。
  - 对照是同一模型、同一引擎（vLLM bf16）的思考关结果，已有：14B 84.6 / 48.8 / 69.2，32B 80.0 / 58.8 / 70.0。两者只差"开不开思考"这一个变量。
- **判定：沿用 EXPLORE_PLAN「10-06 复盘新增」C**。C 在 10-06 写定，那时还没有任何 4B 以上原模型开思考的 MemConflict 输出。判定标准不改，只补两条（都在任何 14B / 32B 开思考的输出之前写定）：14B 也按 C 判定；到点截断时，配对题少于 120 道只作描述。`diag_think_scale.py` 照此计算。
  - 只用两次运行（开 / 关）都有正序和该顺序的题：
    - E_think = 倒序（或检索序）− 正序（思考开），配对 bootstrap 95% CI，另报按用户整群的 CI。
    - E_off = 同一批题上思考关的倒序 − 正序。
    - D = E_think − E_off。
  - 统一门槛（思考开本身）：E_think ≤ −10 且 CI 不含 0 → **复现**；否则未复现。
  - 思考的作用：
    - E_think > −5 且 CI 含 0 → **思考消除**；
    - 否则，D ≥ +10 且 CI 不含 0 → **思考减弱**；
    - 其余 → **没有明显作用**。
  - 标注：
    - 配对题少于 120 道（到点截断）→ 只作描述，不下判定。
    - 思考没写完超过 5% → 结果加标注。C 原计划为这种情况补跑上限 2048，这次租期内没有时间，如实写明。
  - 检索序（如果跑到）按同样规则另报。
  - 论文怎么写（C 已定，这里补全）：
    - 32B **思考消除**：必须写"32B 开思考时没有这个问题"，"思考模式无效"改为小模型的结论。
    - 32B 开思考仍**复现**：写"到 32B，只开思考仍不够，训练仍有必要"。
    - 其余情况：只报数字和区间。
    - 14B 作为规模曲线的中间点。8B 已有：E_think −16.7（80.4 / 63.7），思考关 −37.1。
  - C 里"同时用 vLLM 跑 4B 思考开作同引擎对照"一项，由同引擎的 8B 思考开代替。用户 10-07 定了云端只跑大显存模型。4B 改在 5080 上用 HF 跑（`base_mc_think`），和 HF 的思考关结果配对。
- **执行步骤（云端会话）**：
  1. 取文件（不合并分支；本分支的 `CLOUD_PLAN.md` 只比 `cloud-l20` 多了 main 上的 P7 第 7 步和本节）：
     ```
     git fetch origin claude/blissful-volta-87xkbv
     git checkout FETCH_HEAD -- CLOUD_PLAN.md diag_think_scale.py cloud/run_extmem_conc.py cloud/orchestrate_fin.sh cloud/jobs_fin0.txt cloud/jobs_fin0x.txt cloud/jobs_fin2.txt cloud/jobs_fin2x.txt cloud/jobs_finj.txt
     ```
  2. 冒烟测试（02:30 前做）：借 GPU1 上 `jobs_lme8` 正在用的 :8001 服务，写到临时文件，不碰正式结果。
     - 先用 `curl -s http://127.0.0.1:8001/v1/models` 看它在服务哪个模型：`Qwen/Qwen3-8B` 对应 `--models Qwen3-8B-bf16`，`.../e181-q8/merged` 对应 `--models Qwen3-8B-e181`。
     - 然后运行：
       ```
       HF_HOME=/data/hf_cache COT_DATA=/data/datasets HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 COT_ENGINE=vllm COT_VLLM_URL=http://127.0.0.1:8001 \
         .venv/bin/python cloud/run_extmem_conc.py --models Qwen3-8B-bf16 --tasks memconf --conds chrono,rev --think --budget 1024 --conc 4 --limit 4 --out /tmp/smoke_mc.jsonl
       ```
     - 通过的标准：4 行；字段和 `results/extmem_Qwen3-8B-bf16~vllm+think.jsonl` 的行一样；`full` 里有 `</think>`；`response` 不空。只打印这些判断，不打印题目和回答，应输出 `4 True True`：
       ```
       .venv/bin/python -c "import json; s=[json.loads(l) for l in open('/tmp/smoke_mc.jsonl')]; r=json.loads(open('results/extmem_Qwen3-8B-bf16~vllm+think.jsonl').readline()); print(len(s), all(sorted(x)==sorted(r) for x in s), all('</think>' in x['full'] and bool(x['response']) for x in s))"
       ```
     - 测完删掉临时文件。
  3. 启动编排（只启动一次，它会先等 `jobs_c` 结束）。启动前先确认没有别的编排脚本在等着接队列：`pgrep -af '[o]rchestrate|[c]hain_after'` 应该没有输出。
     ```
     setsid nohup cloud/orchestrate_fin.sh >> logs/queue_cloud.out 2>&1 < /dev/null &
     ```
     进度看 `logs/queue_cloud.log` 里 `orchestrate_fin:` 开头的行。
     - 要中止：先 `pkill -f '[o]rchestrate_fin.sh'`，再停对应的队列（把 `fin0` 换成要停的队列名）：
       ```
       pid=$(pgrep -f '[q]ueue_cloud.sh cloud/jobs_fin0.txt' | head -1); [ -n "$pid" ] && pkill -TERM -s "$(ps -o sid= -p "$pid" | tr -d ' ')"
       ```
       不要用 `cloud/pause_queue.sh`：它不分队列，会停掉找到的第一个队列，两条队列同时在跑时可能停错。
  4. 06:30 后上传：
     - 确认日志里已有 `orchestrate_fin: GPU work over`。
     - 统计（只看汇总）：`.venv/bin/python diag_think_scale.py`（8B / 14B / 32B 三对）和 `.venv/bin/python explore_extmem.py stats --parse v2`。
     - 在 `CLOUD_NOTEBOOK.md` 新开一节"P8 · 开思考的规模曲线"，照预注册逐条写判定和数字，写明配对题数、没写完的比例、是否被截断。
     - `git add -f` 所有还没提交的结果，包括 10-08 以来的这些：LongMemEval 官方 4B / 8B、LoCoMo 官方、新留出集的判分、方案 C、general_pp、E18.1-8B 通用能力、`extmem_Qwen3-8B-bf16~vllm+think_judged2.jsonl`、P8 的新结果。每个文件都要小于 100 MB。
     - commit，push 到 `cloud-l20`，然后用 `git log origin/cloud-l20 -1` 核对确实推上去了。
     - 用户撤销 HF token 和 GitHub PAT，关机。
- **这次不排的**：
  - 任何 4B 测试：5080 能跑。原模型 4B 开思考的 MemConflict 已在台式机队列里（`base_mc_think`），建议提前，和台式机 HF 的思考关结果（75.4 / 36.2 / 56.2）配对，给规模曲线补上 4B 这个点。
  - 14B / 32B 的新干净留出集：4B、8B 只开思考已基本修好，结果可以预料；时间也不够。
  - 32B 的 LongMemEval 官方端到端：太长。
  - Gemma 的干净留出集、14B / 32B 的训练：时间不够，也没有预注册。

### P0 · 移植并验证（见第 4 节）

### P1 · Qwen3-32B 原模型：现象是否在大模型上仍然存在

先各跑 10 条量一下速度，算出整份清单的耗时，报给用户，再决定全跑还是先跑哪些。

| 顺序 | 测什么 | 命令（`--models Qwen3-32B`） | 判分 |
|---|---|---|---|
| 1 | MemConflict / LoCoMo / MAB-CR 三种顺序（含检索序） | `explore_extmem.py run`，再 `explore_extmem.py judge` | 14B 4-bit 判分，MAB 用字符串 |
| 2 | ConvoMem 长版、PersonaMem | `explore_longconv.py run`，再 `explore_longconv.py judge` | 判分 / 选项字母 |
| 3 | LongMemEval + ConvoMem（应用测试） | `run_app_memory.py read`，再 `run_app_memory.py judge` | 14B 判分（LongMemEval 已看过，要标注） |
| 4 | 日志、agent 轨迹 | `run_app_logs.py --n 100`、`run_app_agent.py --n 150` | 字符串 |
| 5 | CoT / MAB / ToT / TempReason / GSM8K / BABILong | `run_ext_eval.py` | 字符串 |
| 6 | E15：问"最早的值"，区分语义规则还是机械近因 | `explore_order_rule.py run` | 字符串，预注册判定在脚本的 `stats` 里 |
| 7 | 思考模式（可选，很慢） | `run_app_thinking.py` 那一套 | — |

- 判分器只在需要判分的数据都生成完后跑一次，省得反复加载。
- 32B 在 2 万 token 的 prompt 上 prefill 很慢，MemConflict 可能每条十几到几十秒，先测再说。

### P1b · Qwen3-14B 长对话（10-06 从台式机队列移来）

- 台式机原计划跑 `explore_longconv.py run --models Qwen3-14B`，然后 `explore_longconv.py judge`，测 ConvoMem 长版和 PersonaMem。在 16 GB 上只能用 4-bit，3 万 token 时可能爆显存，所以移到云端。
- **同时补测 14B 在 10-04 之后新加的测试集**，用 `--models Qwen3-14B`：
  - `explore_extmem.py run`：MemConflict、LoCoMo、MAB-CR，三种顺序；
  - `explore_order_rule.py run`：E15。

  14B 的其他测试（CoT 总表、日志 / 邮件 / git、agent、LongMemEval、E1b、E7 探针）台式机都已做过（4-bit），不用重跑。已知模式：短的结构化记录上 14B 几乎不受顺序影响（日志正序 98.8 / 倒序 94.9），长会话里和 4B 一样掉（LongMemEval 带日期 78.2 → 53.8）。32B 要检验的就是这个模式。
- **用 `MODELS` 里现有的 4-bit 配置跑**，和台式机已有的 14B 结果保持一致。时间够的话，再另跑一份 bf16 作对照（加一个新的 `MODELS` 键，例如 `Qwen3-14B-bf16`）。

### P2 · 台式机装不下的长测试集（4B 和 32B 都跑）

- **MemoryAgentBench-CR `factconsolidation_sh_32k`**（约 3.75 万 token）：`explore_extmem.py` 的 `mabcr()` 只读了 6k 版，加一个参数选 32k。照原来的规则，主要看"答案事实恰有一条更早冲突事实"的题。注意它**不是留出集**，而且有先验冲突导致的地板效应：4B 在 6k 上正序 83% 选了旧值。
- **BEAM**：还没写读取代码。先读论文和数据格式，找出"同一事实被更新"的题，写好构造方法和门槛（预注册）再跑。

### P3 · 换家族（先问用户）：OLMo-2-32B-Instruct

只跑 P1 的第 1、2、4、5 项。OLMo-2 原生上下文只有 4k，长测试集可能不适用；先查它支持的长度，不行就只跑短任务。

### P6 · E18 的通用能力检查（10-06 用户要求放到云端；在 P1 之外的空闲卡上做，优先级高于 P2 / P3）

- **背景**：E18 是台式机上训的 4B 修复候选（清单放进 Qwen3 思考模式）。修复的主要测试在台式机跑；"不伤原模型"的通用能力检查改到云端跑，台式机队列里已删除。门槛见 `EXPLORE_PLAN.md`「通用能力检查」和「E18」第 5 条：相对原模型，MMLU / ARC-C / HellaSwag / GSM8K 各降 ≤2 个点，IFEval 降 ≤3，LongBench 四项平均降 ≤2。**测的是思考关**（`explore_general.py` 默认 `enable_thinking=False`）。
- **取得权重**：权重在单独的分支 `weights-e18`（不在 main），按那里 `weights/e18-dec/RESTORE.md` 的步骤还原到 `runs/e18-dec/final/`，并**核对 SHA256 = `45f05d3d41fbd9fc3a4428e5e6d0eafa91a5fff3345e27af2712a348901e91e2`**。
  ```
  git fetch origin weights-e18
  git checkout origin/weights-e18 -- weights/e18-dec
  ```
  还原后不要把 `weights/` 提交到 `cloud-l20`。
- **原模型和 E18 都在云端跑**，在同一台机器、同一套设置下比较。台式机的原模型结果只做参照：MMLU、ARC、HellaSwag、IFEval 已有，LongBench 没跑完。
  ```
  python explore_general.py fetch                                    # 一次，缓存数据集（联网）
  python explore_general.py run --models Qwen3-4B
  python explore_general.py run --models Qwen3-4B@runs/e18-dec/final
  python run_ext_eval.py --models Qwen3-4B --benches gsm8k
  python run_ext_eval.py --models Qwen3-4B@runs/e18-dec/final --benches gsm8k
  python explore_general.py stats --models Qwen3-4B,Qwen3-4B@runs/e18-dec/final
  ```
- **不要改 `explore_general.py` 的设置**，包括题数、batch、截断长度：MMLU 每个学科前 50 题、HellaSwag 前 2000、IFEval 全部 541、LongBench-E 四项各前 50、MMLU 和 LongBench 用 batch 1。这些是预注册时定的，台式机也用这套。L20 显存大，也不要为了更快而改 batch，batch 会轻微影响对数似然的数值。
- 两张卡可以各跑一个模型并行（`CUDA_VISIBLE_DEVICES=0` / `1`）。
- **报告**：每项给原模型、E18、差值（附 lm-eval 报出的标准误），逐项按门槛写"达到 / 没达到"。写进 `CLOUD_NOTEBOOK.md`，`results/general_*.json` 和 `results/ext_gsm8k_*.jsonl` 推到 `cloud-l20`。
- **E17 和短对话 LoRA 不测**：已被 E18 取代，不进论文。

### P7 · E18.1：训练和测试都在云端（10-06 17:30，用户要求；优先级高于 P2 / P3）

- **是什么**：E18 的改进版，只改训练数据。为什么改、改了什么、门槛，见 `EXPLORE_PLAN.md`「E18.1」，**照那里的门槛判定，不要改**。
  - E18 的问题：思考开时 PersonaMem 掉了 7–17 个点，思考被"记录清单"模板劫持。
  - E18.1 的改动：加入原因题、整句选项的多选题，以及原版 Qwen3-4B 自己生成的通用回答（自蒸馏）。
- **0. 先合并 main**：台式机在 main 上改了代码，你的 `cloud-l20` 需要这些改动：`app_common.strip_think / to_think / out_tag`、各测试脚本的 `--think`、`run_app_fix.generate_think`（思考模式用官方采样）、`train_lora` 的思考行、`gen_bind_data_v3/v4`、`gen_selfdistill.py`、`explore_format_eval --data bind4`。执行 `git merge origin/main`，冲突时两边都保留。
- **1. 还原输入数据**：E12 的 1000 行在分支 `weights-e18` 的 `data/`，按那里 README 的命令还原为 `data_train/reason_decoupled_train.jsonl`，**核对 SHA256 = `21de579406e02d0dcb1c1a45a154a30d71ef2e2cec0fb4c68e5b93fb944252bf`**。可选的自检：`python gen_bind_data_v3.py` 应得到 `bind3_decoupled_train.jsonl`，哈希前缀 `4BAFEB760E58B565`。如果不一致，多半是 WikiText 的 parquet 版本变了，先停下来报告。
- **2. 自蒸馏数据**：`python gen_selfdistill.py --n 600 --engine vllm`（用 vLLM 的 venv；缺 pandas / datasets / transformers 就在那个 venv 里补装，或者用 `--engine hf` 在主环境跑，会慢）。
  - **要下载**：`allenai/tulu-3-sft-mixture` 的 6 个 parquet（约 1.4 GB，固定 revision `b14afda6…`），以及 `google/IFEval`（很小，可能已缓存）。**先把下载清单给用户确认**。
  - 记下保留行数、丢弃数（写不完 / 太长）、思考开的行数、哈希。
- **3. 生成 E18.1 训练数据**：`python gen_bind_data_v4.py`。它会打印 `bind4_decoupled_*` 三个文件的哈希和各题型数量，自检不过会直接报错。
- **4. 训练（4B，HF，单卡）**：
  ```
  python train_lora.py --model Qwen3-4B --out runs/e181-dec --data data_train/bind4_decoupled_train.jsonl --val data_train/bind4_decoupled_val.jsonl --dev data_train/bind4_decoupled_dev.jsonl --rank 16 --alpha 32 --lr 1e-4 --epochs 1 --accum 8 --max_len 8192 --eval_every 300 --save_every 50 --val_n 60 --dev_n 60 --seed 0
  ```
- **5. 测试：全部用 HF（不要设 `COT_ENGINE=vllm`）**，和台式机的 E18 用同一套代码路径，结果才能直接比较。按下面的顺序跑，先跑能决定成败的：
  ```
  M=Qwen3-4B@runs/e181-dec/final
  python explore_format_eval.py run --data bind4 --modes think,direct --models $M
  python explore_longconv.py run --think --budget 1024 --models $M      # ConvoMem 长版 + PersonaMem，思考开（主要门槛）
  python explore_longconv.py judge
  python explore_extmem.py run --tasks memconf --think --budget 1024 --models $M
  python explore_extmem.py judge --parse v2
  python explore_general.py run --models $M ; python run_ext_eval.py --models $M --benches gsm8k
  python explore_longconv.py run --models $M                            # 思考关
  python explore_reason_eval.py run --think --models $M ; python explore_reason_eval.py run --models $M ; python explore_reason_eval.py judge
  python explore_extmem.py run --tasks memconf --models $M ; python explore_extmem.py judge --parse v2
  ```
  看结果用各脚本的 `stats`，带 `+think` 的结果文件是思考开。原模型的通用能力基线直接用 P6 的结果（同一台机器）。
- **6. 报告和交付**：
  - 在 `CLOUD_NOTEBOOK.md` 逐条写：数据哈希、训练日志的验证分数，以及每条门槛"达到 / 没达到"，附数字和置信区间。和台式机的 E18 并排比较：PersonaMem 思考开 63.9 / 56.7，思考关 72.7 / 72.3；ConvoMem 长版思考开 97.6 / 96.8；MemConflict 思考开 72.9 / 71.7 / 70.0。
  - PersonaMem 没过门槛时，只按汇总诊断：思考开头是不是还是清单、写不完的比例、原因题和演变题分别的正确率。可参照台式机的 `diag_pm_harm.py` 和 NOTEBOOK 里的诊断写法。
  - 结果文件推到 `cloud-l20`。**权重**按 `weights-e18` 的做法（权重文件超过 100 MB 就切成两段，并给出 SHA256）推到新分支 `weights-e181`，台式机要用。
- **7. 交给台式机（10-06 21:00 加；现在就做，不用等训练）**：台式机要用同样的数据训 E18.1 的种子 1、2。把 `data_train/selfdistill_train.jsonl` 推到分支 `weights-e18` 的 `data/selfdistill_train.jsonl`，在 `data/README.md` 里补上它的 SHA256（LF）和行数，再推一次。
- **不要做**：改门槛、改训练超参、改数据比例、用测试结果回头改训练数据；也不要训 32B。

### P5 · MemConflict 端到端（**10-06 用户决定降级、暂缓：先不要做，包括读论文和查仓库**；保留设计备查）

- **为什么做**：前面的 MemConflict 测试是我们截取会话、人为排顺序做出来的受控测试，只能证明机制。端到端测试要用原作者的全部题、全部会话，加一个标准检索记忆，顺序由检索器决定，检索失败也计入成绩。它回答的是"现实部署里修复有没有用"。
- **原论文（2605.20926）的协议**（台式机 10-06 已读原文核实）：
  - 评测 6 个记忆系统（A-Mem、LangMem、Letta、MemOS、Mem0、Memobase），后端 LLM 都是 gpt-5.0-mini，**没有 BM25 / embedding 加读者的基线**；
  - 默认检索 K=3（敏感性分析用 K=2、5）；
  - 判分：用 LLM 比对答案再人工复核，判分提示见原文附录 Fig. A5；
  - 指标：AA 按动态 / 静态 / 条件分别报告，总分宏平均，动态冲突另报 UOCS、静态冲突另报 CRS；
  - 主实验用了 12 个用户，我们的数据文件有 30 个。
- **我们的设置**（读者研究，不换记忆系统）：
  - **检索器**：照 LongMemEval 论文里检索加读者实验的默认设置（检索器、按会话还是按轮切块、K）。**云端先读论文核实，写进预注册**。另加 BM25 一组，它不依赖额外模型。
  - **K**：按会话切块时 K=3 为主（约 1.2 万 token），K=5 作敏感性分析。
  - **两种呈现顺序都报**：检索器的相关度顺序（最相关在前），以及按会话日期重排（很多记忆系统这样做）。两种都是部署里会出现的，都不是我们排的。
  - **题目**：全部 30 个用户、全部三类冲突，原题原答案，不做任何筛选。
  - **判分**：原文 Fig. A5 的提示，判分模型换成 Qwen3-14B 4-bit（偏离官方协议，论文里写明）。抽约 100 题由用户人工核对，报一致率。是否另用 gpt-5-mini 判一个子集，由用户决定（需要 API 和费用）。
  - **比较**：原模型和修复模型都跑，同一检索结果、同一提示。4B 原模型、E17 / E18 在台式机跑；14B、32B 原模型在云端跑；32B 修复模型等 P4。
- **预注册要点**（开跑前由台式机写进 EXPLORE_PLAN，云端照用）：
  - 修复模型总分（宏平均 AA）不低于原模型；
  - 动态冲突 AA 高于原模型，置信区间不含 0（按用户整群 bootstrap）；
  - **静态冲突不能明显下降**：静态冲突的正确答案是原本稳定的值，不是最新的值，"永远选最新"的矫枉过正会在这里暴露。
- **云端现在能先做的**：读 LongMemEval 论文的检索加读者设置并写下来；查 MemConflict 的 GitHub 仓库有没有公开评测代码和判分提示的原文，有就记下路径和 commit。**脚本等台式机推送后再跑。**

### P4 · 32B 训练（必须等台式机通知，并经用户同意）

- 用 E18 的最终配方和数据文件，由台式机推送。训练超参照 4B：LoRA rank 16、alpha 32、学习率 1e-4、1 轮、max_len 8192。32B 用 bf16 双卡，开梯度检查点；显存不够再考虑 4-bit 底座（QLoRA），但那是变量变化，要预注册。
- 训完必须跑：修复测试（ConvoMem 长版、MemConflict、格式测试）+ 通用能力检查（`explore_general.py run --models Qwen3-32B@runs/...`）+ 32B 原模型的通用能力检查作为基线。

## 6. 记录和同步

- **云端只写 `CLOUD_NOTEBOOK.md`**，不直接改 `NOTEBOOK.md` / `RESULTS.md`，以免和台式机冲突，台式机会合并。内容：
  - 环境
  - 移植验证结果
  - 每个实验的预注册门槛和结果（附数字和置信区间）
  - 留出集接触记录
  - 问题和更正
- **commit 并 push 到 `cloud-l20` 分支**。`results/` 下的 jsonl 和 json 都提交；大文件（权重、`runs/`）不提交。
- 每个实验结束都推一次，这样台式机随时能拉下来看。

## 7. 需要先问用户的事

- 租多久、预算多少，决定 P1 跑全还是挑着跑。
- 盘有多大，至少约 200 GB。
- P3 要不要做。
- 下载清单是否同意。

## 8. 背景速查：现在知道什么

- **现象**：显式时间（日期、编号）在场时，模型仍按呈现位置判断哪个值更新。14 个模型上都有，k=8 时 89.5–99.5% 选最后一个出现的值。
  - MemConflict（别人的数据）：4B 倒序 −39.2、检索序 −19.2。
  - "语义规则还是机械近因"仍未区分开（E15 结果是 MIXED），论文的说法是"位置压过显式时间"。
- **修复（4B）**：E17 绑定训练（多题型、按要求格式作答、块顺序解耦），格式测试和 MemConflict 的门槛都达到：
  - MemConflict：72.5 / 72.5 / 73.3，原模型 75.4 / 36.2 / 56.2；
  - ConvoMem 长版：96.8 / 96.8，原模型 95.2 / 77.4；
  - 但 PersonaMem 低约 14 个点，通用能力检查在台式机上进行中。
- 修复按两阶段讲：选择（解耦）+ 绑定（事实与时间）。
