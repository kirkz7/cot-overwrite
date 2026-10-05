"""No-harm suite (EXPLORE_PLAN.md "通用能力检查", 10-05): does a fix hurt the base model's general abilities?
Protocol copied from published fine-tuning fixes (FILM-7B / IN2 2404.16811; Xiong et al. 2406.19292; Biderman et al.
TMLR 2024), plus IFEval because their multiple-choice benchmarks cannot see format collapse. lm-evaluation-harness 0.4.13,
the reader loaded exactly as in every other experiment (app_common.load_reader: LoRA merged into bf16 Qwen3-4B).
  loglikelihood, no chat template : MMLU 5-shot (as FILM), ARC-Challenge 0-shot, HellaSwag 0-shot (first 2000)
  generation, chat template, no thinking : IFEval (all 541), LongBench-E qasper / multifieldqa_en / hotpotqa / 2wikimqa
                                            (first 100 each, context cut to 16k tokens by the harness)
GSM8K is already measured for every reader in run_ext_eval.py (reported alongside).
usage: python explore_general.py fetch | run --models Qwen3-4B,Qwen3-4B@runs/e17-dec/final | stats
"""
import argparse
import json
import os

OUT = "results/general_{}.json"
LL_TASKS = {"mmlu": 5, "arc_challenge": 0, "hellaswag": 0}
LL_LIMIT = {"hellaswag": 2000}
GEN_TASKS = ["ifeval", "longbench_qasper_e", "longbench_multifieldqa_en_e", "longbench_hotpotqa_e", "longbench_2wikimqa_e"]
GEN_LIMIT = {t: 100 for t in GEN_TASKS if t.startswith("longbench")}
MAX_LEN = 16384
# headline metric per task
METRIC = {"mmlu": "acc,none", "arc_challenge": "acc_norm,none", "hellaswag": "acc_norm,none",
          "ifeval": "prompt_level_strict_acc,none", "longbench_qasper_e": "qa_f1_score,none",
          "longbench_multifieldqa_en_e": "qa_f1_score,none", "longbench_hotpotqa_e": "qa_f1_score,none",
          "longbench_2wikimqa_e": "qa_f1_score,none"}


def fetch(args):
    """online once: datasets into HF_HOME and the nltk tokenizer IFEval needs, so the queue can run offline"""
    import nltk
    from lm_eval.tasks import TaskManager, get_task_dict
    nltk.download("punkt_tab")
    tm = TaskManager()
    d = get_task_dict(list(LL_TASKS) + GEN_TASKS, tm)
    print("tasks loaded:", len(d))


def evaluate(lm, tasks, fewshot, limit, chat):
    from lm_eval import simple_evaluate
    res = {}
    for t in tasks:
        r = simple_evaluate(model=lm, tasks=[t], num_fewshot=fewshot.get(t, None) if fewshot else None,
                            limit=limit.get(t), apply_chat_template=chat, log_samples=False, bootstrap_iters=1000)
        res[t] = {k: v for k, v in r["results"].items()}
        print(t, {k: v for k, v in r["results"].get(t, {}).items() if not k.startswith("alias")}, flush=True)
    return res


def run(args):
    import torch
    from lm_eval.models.huggingface import HFLM
    from app_common import tag, load_reader, free_gpu
    for name in args.models.split(","):
        path = OUT.format(tag(name))
        done = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
        todo_ll = [t for t in LL_TASKS if t not in done]
        todo_gen = [t for t in GEN_TASKS if t not in done]
        print(name, "todo", todo_ll + todo_gen, flush=True)
        if not todo_ll + todo_gen:
            continue
        tok, model = load_reader(name)
        lm = HFLM(pretrained=model, tokenizer=tok, batch_size=args.batch, max_length=MAX_LEN, enable_thinking=False)
        for tasks, fs, lim, chat in ((todo_ll, LL_TASKS, LL_LIMIT, False), (todo_gen, None, GEN_LIMIT, True)):
            for t in tasks:
                lm.batch_size_per_gpu = 1 if t.startswith("longbench") else int(args.batch)   # 16k contexts: one at a time
                done.update(evaluate(lm, [t], fs, lim, chat))
                json.dump(done, open(path, "w", encoding="utf-8"), indent=1)    # resumable per task
                torch.cuda.empty_cache()
        del lm, tok, model
        free_gpu()


def stats(args):
    from app_common import tag
    rows = {}
    for name in args.models.split(","):
        path = OUT.format(tag(name))
        if os.path.exists(path):
            r = json.load(open(path, encoding="utf-8"))
            rows[tag(name)] = {t: (r.get(t, {}).get(t, {}).get(m), r.get(t, {}).get(t, {}).get(m.replace(",", "_stderr,")))
                               for t, m in METRIC.items()}
    if not rows:
        return
    base = rows.get("Qwen3-4B")
    print(f"{'task':30s}" + "".join(f"{n:>22s}" for n in rows))
    for t in METRIC:
        line = f"{t:30s}"
        for n, r in rows.items():
            v, se = r[t]
            if v is None:
                line += f"{'-':>22s}"
                continue
            d = "" if base is None or n == "Qwen3-4B" or base[t][0] is None else f" ({(v - base[t][0]) * 100:+.1f})"
            line += f"{v * 100:>12.1f}{d:>10s}"
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fetch", "run", "stats"])
    ap.add_argument("--models", default="Qwen3-4B")
    ap.add_argument("--batch", default="4")
    args = ap.parse_args()
    {"fetch": fetch, "run": run, "stats": stats}[args.stage](args)


if __name__ == "__main__":
    main()
