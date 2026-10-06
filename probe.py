"""Score every integer candidate in [LO, HI] by log p(tokens | prefix).

Works for any tokenizer: each candidate's continuation tokens are computed from
tok(prefix + str(c)); the prefix is run once, then all distinct proper prefixes of
the continuations are run as one right-padded batch on top of the branched KV cache.
"""
import os

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from tasks import LO, HI

# The Windows torch wheel has no flash-attention kernel, and mem-efficient SDPA does not
# support GQA, so long prefills fall back to the O(n^2)-memory math kernel and spill into
# shared system memory. cuDNN attention supports GQA; use it for prefill. For 1-token decode
# steps cuDNN re-plans as the KV length changes, so decode with the (cheap at q_len=1) math kernel.
PREFILL = [SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


@torch.no_grad()
def greedy(model, ids, max_new, stop_ids=()):
    with sdpa_kernel(PREFILL, set_priority=True):
        out = model(ids, use_cache=True, logits_to_keep=1)
    cache, gen = out.past_key_values, []
    nxt = out.logits[0, -1].argmax().item()
    with sdpa_kernel(SDPBackend.MATH):
        for _ in range(max_new):
            gen.append(nxt)
            if nxt in stop_ids:
                break
            out = model(torch.tensor([[nxt]], device=ids.device), past_key_values=cache, use_cache=True)
            cache, nxt = out.past_key_values, out.logits[0, -1].argmax().item()
    return gen


def load(name, four_bit=False):
    tok = AutoTokenizer.from_pretrained(name)
    # COT_DEVICE_MAP=auto spreads a model over both cloud cards (Qwen3-32B); unset on the desktop -> one card as before
    kw = dict(dtype=torch.bfloat16, device_map=os.environ.get("COT_DEVICE_MAP", "cuda"))
    if four_bit:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(name, **kw)
    model.eval()
    return tok, model


class Scorer:
    def __init__(self, tok, model):
        self.tok, self.model = tok, model
        self.cands = list(range(LO, HI + 1))
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        self._conts = {}

    def _enc(self, s):
        return self.tok(s, add_special_tokens=False).input_ids

    @torch.no_grad()
    def score(self, prefix):
        """Returns dict cand -> logprob."""
        p_ids = self._enc(prefix)
        # continuations depend only on the prompt's ending, so tokenize candidates once per ending
        key = prefix[-40:]
        if key not in self._conts:
            k_ids = self._enc(key)
            conts = {}
            for c in self.cands:
                full = self._enc(key + str(c))
                assert full[:len(k_ids)] == k_ids, "prefix tokenization changed; adjust prompt ending"
                conts[c] = tuple(full[len(k_ids):])
            self._conts[key] = conts
        conts = self._conts[key]
        paths = sorted({cont[:j] for cont in conts.values() for j in range(1, len(cont))})

        out = self.model(torch.tensor([p_ids], device="cuda"), use_cache=True, logits_to_keep=1)
        lp = {(): torch.log_softmax(out.logits[0, -1].float(), -1)}
        if paths:
            cache = out.past_key_values
            cache.batch_repeat_interleave(len(paths))
            m = max(map(len, paths))
            inp = torch.tensor([list(p) + [self.pad] * (m - len(p)) for p in paths], device="cuda")
            logits = self.model(inp, past_key_values=cache, use_cache=False).logits.float()
            for i, p in enumerate(paths):
                lp[p] = torch.log_softmax(logits[i, len(p) - 1], -1)
        return {c: sum(lp[cont[:j]][cont[j]].item() for j in range(len(cont))) for c, cont in conts.items()}
