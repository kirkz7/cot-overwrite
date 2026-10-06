"""P0 E15 diagnosis: at the first token where the cloud and desktop responses diverge, how far apart are the two tokens'
logits on the cloud model? Tiny gaps = greedy near-ties flipped by float differences; large gaps = a real port bug."""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd, torch
from app_common import load_jsonl, load_reader, chat_prompt
import explore_order_rule as e15
key = ["id", "order", "qtype"]
r = pd.DataFrame(load_jsonl("reference/desktop/e15_rule_Qwen3-4B.jsonl")).set_index(key)
n = pd.DataFrame(load_jsonl("results/e15_rule_Qwen3-4B.jsonl")).set_index(key).loc[r.index]
its = {(x["id"], x["order"], x["qtype"]): x for x in e15.items()}
tok, model = load_reader("Qwen3-4B")
gaps, ctrl = [], []
for k in r.index:
    a, b = str(r.loc[k, "response"]), str(n.loc[k, "response"])
    p = chat_prompt(tok, its[k]["prompt"])
    pid = tok(p, add_special_tokens=False).input_ids
    ta, tb = tok(a, add_special_tokens=False).input_ids, tok(b, add_special_tokens=False).input_ids
    if a.strip() == b.strip():
        if len(ctrl) < 60:   # control: top-1 minus top-2 gap at the first generated token of agreeing rows
            with torch.no_grad():
                lg = model(torch.tensor([pid], device="cuda")).logits[0, -1].float()
            t2 = lg.topk(2).values; ctrl.append((t2[0] - t2[1]).item())
        continue
    j = next((i for i in range(min(len(ta), len(tb))) if ta[i] != tb[i]), None)
    if j is None:
        continue
    with torch.no_grad():
        lg = model(torch.tensor([pid + tb[:j]], device="cuda")).logits[0, -1].float()
    gaps.append(dict(id=k, step=j, cloud_minus_desktop=(lg[tb[j]] - lg[ta[j]]).item(),
                     cloud_is_top1=bool(lg.argmax().item() == tb[j])))
g = pd.DataFrame(gaps)
print("diverging rows", len(g), "| cloud token is top-1 here:", g.cloud_is_top1.mean())
print("logit gap cloud - desktop token at the divergence:", g.cloud_minus_desktop.describe().round(3).to_dict())
print("share with |gap| < 0.25:", (g.cloud_minus_desktop.abs() < 0.25).mean().round(3), "| < 0.5:", (g.cloud_minus_desktop.abs() < 0.5).mean().round(3))
print("control: top1-top2 gap at first token of agreeing rows, median", np.median(ctrl).round(3), "p10", np.percentile(ctrl, 10).round(3))
