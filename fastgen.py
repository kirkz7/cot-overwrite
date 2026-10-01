"""Batched generation with a static KV cache and a CUDA-graph decode step (no torch.compile / triton needed).

HF generate() with the dynamic cache fragments the allocator on Windows (reserved memory grows past the
16 GB card and spills into shared memory). Here the cache is preallocated once, prompts are left-padded and
prefilled in chunks with an explicit 4D mask, and the 1-token decode step is captured as a CUDA graph.
Works for bf16 HF decoder models whose cache layers are transformers' StaticLayer (Qwen3 tested).
"""
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AttentionInterface, StaticCache
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, sdpa_mask


def gqa_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    """Grouped-query attention without materializing repeated K/V (HF's sdpa path copies the whole static
    cache n_rep times per layer, which overflows 16 GB at long cache lengths). attention_mask: bool [B,1,q,L]."""
    B, H, q, D = query.shape
    kvh = key.shape[1]
    g = H // kvh
    qg = query.reshape(B, kvh, g * q, D)                         # head h uses kv head h // g (as repeat_kv)
    scores = torch.matmul(qg, key.transpose(-1, -2)).float() * (scaling if scaling is not None else D ** -0.5)
    if attention_mask is not None:
        m = attention_mask[:, :, None].expand(B, 1, g, q, key.shape[2]).reshape(B, 1, g * q, key.shape[2])
        scores = scores.masked_fill(~m, float("-inf"))
    out = torch.matmul(scores.softmax(-1).to(value.dtype), value)  # [B, kvh, g*q, D]
    return out.reshape(B, H, q, D).transpose(1, 2).contiguous(), None


def gqa_mask(*args, **kwargs):
    kwargs["allow_is_causal_skip"] = False      # gqa_attention has no is_causal path; always build the bool mask
    return sdpa_mask(*args, **kwargs)


AttentionInterface.register("gqa_static", gqa_attention)
ALL_MASK_ATTENTION_FUNCTIONS.register("gqa_static", gqa_mask)


def auto_batch(model, max_len, cap=8, margin_gb=1.5):
    """Largest batch whose static KV cache fits in the memory that is free right now (other programs,
    e.g. the desktop compositor or a game, also hold VRAM; overflow would spill into system memory)."""
    c = model.config
    hd = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    per_seq = 2 * c.num_hidden_layers * c.num_key_value_heads * hd * max_len * 2      # K and V, bf16
    free, _ = torch.cuda.mem_get_info()
    return max(1, min(cap, int((free - margin_gb * 2 ** 30) // per_seq)))


class GraphGen:
    def __init__(self, model, batch, max_len, chunk=128):
        self.model, self.B, self.L, self.chunk = model, batch, max_len, chunk
        model.set_attn_implementation("gqa_static")
        self.cache = StaticCache(config=model.config, max_cache_len=max_len)
        dev = model.device
        self.ids = torch.zeros(batch, 1, dtype=torch.long, device=dev)
        self.pos = torch.zeros(batch, 1, dtype=torch.long, device=dev)
        self.mask = torch.zeros(batch, 1, 1, max_len, dtype=torch.bool, device=dev)
        self.mask[:, :, :, 0] = True
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self._step()
            torch.cuda.current_stream().wait_stream(s)
            self._reset()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.logits = self._step()
        self._reset()

    def _step(self):
        return self.model(input_ids=self.ids, attention_mask=self.mask, position_ids=self.pos,
                          past_key_values=self.cache, use_cache=True, logits_to_keep=1).logits[:, -1]

    def _reset(self):
        for layer in self.cache.layers:
            if getattr(layer, "is_initialized", False):
                layer.cumulative_length.zero_()
                layer.keys.zero_()
                layer.values.zero_()

    @torch.no_grad()
    def generate(self, prompts, max_new, stop_ids, pad_id, sample=True, temperature=0.6, top_p=0.95, top_k=20,
                 generator=None):
        """prompts: list of token-id lists (len <= batch).
        Returns [(generated ids without the stop token, stop token id or None if max_new was reached)]."""
        n = len(prompts)
        prompts = prompts + [prompts[0]] * (self.B - n)
        P = max(map(len, prompts))
        assert P + max_new <= self.L, (P, max_new, self.L)
        dev = self.ids.device
        self._reset()
        ids = torch.full((self.B, P), pad_id, dtype=torch.long, device=dev)
        valid = torch.zeros(self.B, P, dtype=torch.bool, device=dev)
        for b, p in enumerate(prompts):
            ids[b, P - len(p):] = torch.tensor(p, device=dev)
            valid[b, P - len(p):] = True
        pos = (valid.long().cumsum(-1) - 1).clamp(min=0)
        causal = torch.ones(P, P, dtype=torch.bool, device=dev).tril()
        eye = torch.eye(P, dtype=torch.bool, device=dev)
        full = torch.zeros(self.B, 1, P, self.L, dtype=torch.bool, device=dev)
        full[:, 0, :, :P] = (causal[None] & valid[:, None, :]) | eye[None]   # pads attend to themselves (no NaN rows)
        with sdpa_kernel(SDPBackend.MATH):
            for c in range(0, P, self.chunk):
                out = self.model(input_ids=ids[:, c:c + self.chunk], attention_mask=full[:, :, c:c + self.chunk],
                                 position_ids=pos[:, c:c + self.chunk], past_key_values=self.cache, use_cache=True,
                                 logits_to_keep=1)
        del full
        logits = out.logits[:, -1]
        self.mask.zero_()
        self.mask[:, 0, 0, :P] = valid
        next_pos = pos[:, -1:] + 1
        stop = torch.tensor(sorted(stop_ids), device=dev)
        done = torch.zeros(self.B, dtype=torch.bool, device=dev)
        done[n:] = True
        out_ids = torch.full((self.B, max_new), pad_id, dtype=torch.long, device=dev)
        for t in range(max_new):
            tok = self._pick(logits, sample, temperature, top_p, top_k, generator)
            tok = torch.where(done, torch.full_like(tok, pad_id), tok)
            out_ids[:, t] = tok
            done |= torch.isin(tok, stop)
            if t % 32 == 31 and bool(done.all()):
                break
            self.ids.copy_(tok[:, None])
            self.pos.copy_(next_pos)
            self.mask[:, 0, 0, P + t] = True
            self.graph.replay()
            logits = self.logits
            next_pos += 1
        res = []
        for row in out_ids[:n].tolist():
            cut = next((i for i, x in enumerate(row) if x in stop_ids or x == pad_id), len(row))
            res.append((row[:cut], row[cut] if cut < len(row) else None))
        return res

    @staticmethod
    def _pick(logits, sample, temperature, top_p, top_k, generator):
        if not sample:
            return logits.argmax(-1)
        v, i = (logits.float() / temperature).topk(top_k, -1)
        p = v.softmax(-1)
        keep = (p.cumsum(-1) - p) < top_p
        p = (p * keep).clamp(min=0)
        choice = torch.multinomial(p / p.sum(-1, keepdim=True), 1, generator=generator)
        return i.gather(-1, choice)[:, 0]
