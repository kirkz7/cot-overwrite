# 笔记本（RTX 4060 Laptop，8 GB）上的队列

## 跑什么，为什么只跑这些
- 8 GB 能装下的只有 **Qwen3-1.7B**（bf16 约 3.4 GB）。Qwen3-4B、Phi-4-mini 的 bf16 权重约 8 GB，放不下；改成 4-bit 等于多改了一个变量，结果不能和台式机上的直接比，所以不做。Qwen3-14B 判分器（4-bit 约 9 GB）和所有 LoRA 训练（max_len 8192）也放不下。
- 所以笔记本只跑确认阶段里**还没做的 Qwen3-1.7B 部分**（预注册，queue14 第 4、7 步）：原模型 + 种子 0 的解耦组 / 对照组，在外部测试、LongMemEval 记忆、倒序日志、agent 轨迹上**只生成、不判分**。命令和 queue14 完全一样，输出文件按模型命名，拷回台式机直接用，不会和台式机的文件冲突。
- LongMemEval 是"已看过"的数据，这里是预注册评测的一部分，不涉及修复设计；不碰任何留出集。

## 一、台式机这边已经准备好（10-05，不需要 U 盘）
本分支里已经提交了笔记本需要、但只有台式机才有的东西：
- `runs\q17-dec-s0\final`、`runs\q17-chr-s0\final`（两组 LoRA，各约 70 MB）；
- `results\` 里所有 `Qwen3-1.7B` 的已有结果（`ext_*_Qwen3-1.7B.jsonl` 已经齐了，笔记本会逐条跳过）；
- `laptop_requirements.txt`（台式机环境的完整版本清单）。
公开的模型和数据集（约 4 GB）由 `laptop_fetch.py` 在笔记本上直接从 HuggingFace 下载，**版本号钉死为台式机用的那一版**，只下台式机实际用到的文件，不需要登录。

## 二、在笔记本上
1. `git clone https://github.com/kirkz7/cot-overwrite.git`，进目录，`git checkout claude/elegant-cerf-ywisw5`。
2. 建环境（显卡驱动要支持 CUDA 12.8）：
   ```
   uv venv --python 3.12 .venv
   uv pip install --python .venv\Scripts\python.exe -r laptop_requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
   ```
3. 准备 `D:\hf_cache\hub\`：`run_app_memory.py` 写死了这个路径，而且它是冻结的评测脚本，不能改。如果笔记本没有 D 盘：先建一个文件夹，再 `subst D: C:\那个文件夹`。
4. 下载模型和数据集（约 4 GB）：`.\.venv\Scripts\python.exe laptop_fetch.py`，最后一行会打印 `all files in D:\hf_cache\hub`。
5. 插上电源，运行 `.\resume_queue.ps1 -Queue queue_laptop`，进度在 `logs\queue_laptop.log`。如果缺文件，日志第一行会写出缺哪个。暂停用 `.\pause_queue.ps1`，再用同一条 resume 命令续跑，已经完成的部分不会丢。
6. 跑完后，所有结果会自动收集到 **`laptop_out\`**。结果文件很小（几 MB），可以 `git add -f laptop_out` 提交到本分支推回去，也可以用 U 盘。
7. **笔记本在跑这些任务时，台式机不要恢复 queue14**（里面有同样的 1.7B 任务，会重复）。

## 三、拷回台式机
1. `laptop_out\results\*` → 台式机 `results\`（文件名不同，不会覆盖别的结果）。
2. `laptop_out\logs\*` → 台式机 `logs\laptop\`（**不要**直接放进 `logs\`，同名日志会覆盖）。
3. 判分：`.\.venv\Scripts\python.exe run_app_memory.py judge`。它只判还没判的行，用 14B，要等台式机队列空闲或先暂停。外部测试、日志、agent 在生成时已经判好。
4. 以后台式机恢复 queue14 时，这些任务会发现结果已齐，几秒内结束。
