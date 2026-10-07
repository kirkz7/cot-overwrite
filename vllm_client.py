"""Optional vLLM engine (cloud only; CLOUD_NOTEBOOK.md "vLLM"). With COT_ENGINE=vllm, app_common.load_reader returns a
VLLMReader for a running `vllm serve` (own venv, started by cloud/vllm_ctl.sh) instead of an HF model, and probe.greedy
sends it the same token ids, budget and stop set. Prompts, chat templates and decoding of the output stay in this venv
(desktop versions); only the forward pass moves to vLLM.
Semantics copied from probe.greedy: argmax at every step, at most max_new tokens, stop right after the first token in
stop_ids and keep that token, no other stop (ignore_eos; the model's generation_config is not applied).
"""
import json
import os
import urllib.request

from transformers import AutoConfig


class VLLMReader:
    is_vllm = True

    def __init__(self, hf_id):
        self.url = os.environ.get("COT_VLLM_URL", "http://127.0.0.1:8000").rstrip("/")
        self.name = hf_id
        models = self._call("/v1/models")["data"]
        served = [m["id"] for m in models]
        self.max_model_len = next((m.get("max_model_len") for m in models if m["id"] == hf_id), None) or 10 ** 9
        assert hf_id in served, f"the vLLM server serves {served}, not {hf_id} (cloud/vllm_ctl.sh up {hf_id})"
        self.config = AutoConfig.from_pretrained(hf_id)   # scripts read max_position_embeddings / model_type
        text = getattr(self.config, "text_config", None)   # Gemma 3 keeps the text settings one level down
        if not hasattr(self.config, "max_position_embeddings") and text is not None:
            self.config.max_position_embeddings = text.max_position_embeddings

    def _call(self, path, body=None):
        req = urllib.request.Request(self.url + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as r:
            return json.load(r)

    def sample(self, ids, max_new, stop_ids, seed, temperature, top_p, top_k):
        """thinking mode (run_app_fix.generate_think on vLLM): Qwen3 sampling, one fixed seed per item"""
        stop_ids = sorted({int(i) for i in stop_ids})
        body = dict(model=self.name, prompt=[int(i) for i in ids], max_tokens=max_new, temperature=temperature, top_p=top_p,
                    top_k=top_k, seed=int(seed), ignore_eos=True, stop_token_ids=stop_ids, skip_special_tokens=False,
                    logprobs=1, return_tokens_as_token_ids=True)
        ch = self._call("/v1/completions", body)["choices"][0]
        return [int(t.split(":", 1)[1]) for t in ch["logprobs"]["tokens"]]

    def greedy(self, ids, max_new, stop_ids=()):
        stop_ids = sorted({int(i) for i in stop_ids})
        body = dict(model=self.name, prompt=[int(i) for i in ids], max_tokens=max_new, temperature=0.0, ignore_eos=True,
                    stop_token_ids=stop_ids, skip_special_tokens=False, logprobs=1, return_tokens_as_token_ids=True)
        ch = self._call("/v1/completions", body)["choices"][0]
        gen = [int(t.split(":", 1)[1]) for t in ch["logprobs"]["tokens"]]
        stop = ch.get("stop_reason")
        if isinstance(stop, int) and stop in stop_ids and (not gen or gen[-1] != stop):
            gen.append(stop)   # probe.greedy keeps the stop token; restore it if the server left it out
        return gen
