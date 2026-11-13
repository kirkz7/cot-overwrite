"""Paired synthetic checkpoint diagnostic with an executable compatibility oracle."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path


FAMILIES = {
    "alias": ("Add support for the new configuration key '{new}'. Decide how to handle '{old}'.",
              "accept '{old}' as an alias", "reject '{old}' as unknown"),
    "case": ("Add the new visible label '{new}'. Decide how to handle uppercase forms.",
             "accept uppercase '{old}' as equivalent", "reject uppercase '{old}'"),
    "default": ("Add an explicit '{new}' setting. Preserve the documented fallback when it is omitted.",
                "use '{old}' when omitted", "use '{new}' when omitted"),
    "serialization": ("Add the long identifier '{new}' to the exposed record. Decide what the on-disk ID contains.",
                      "store the short ID '{old}'", "store the long ID '{new}'"),
}

NAMES = {
    "alias": [("timeout", "request_timeout"), ("root", "workspace_root"),
              ("retries", "max_retries"), ("codec", "output_codec"),
              ("port", "listen_port"), ("zone", "region")],
    "case": [(word.upper(), word) for word in ("ready", "online", "open", "green", "active", "valid")],
    "default": [("warn", "strict"), ("local", "remote"), ("brief", "verbose"),
                ("legacy", "modern"), ("text", "json"), ("manual", "auto")],
    "serialization": [("s1", "service-one"), ("w2", "worker-two"), ("m3", "module-three"),
                      ("r4", "record-four"), ("b5", "bundle-five"), ("j6", "job-six")],
}


def cases():
    for group, (family, variants) in enumerate(NAMES.items()):
        for index, (old, new) in enumerate(variants):
            task, *options = [line.format(old=old, new=new) for line in FAMILIES[family]]
            yield {
                "id": f"{family}-{index}",
                "family": family,
                "task": task,
                "old": old,
                "new": new,
                "required": (index + group) % 2,
                "order": int(index % 3 == 0),
                "options": options,
            }


def behavior(case, variant):
    old, new = case["old"], case["new"]
    if case["family"] in {"alias", "case"}:
        accepted = {new, old} if variant == 0 else {new}
        return new in accepted, old in accepted
    if case["family"] == "default":
        def config(value=None):
            return value if value is not None else (old, new)[variant]
        return config(new) == new, config()

    def record():
        return {"visible": new, "on_disk": (old, new)[variant]}
    return record()["visible"] == new, record()["on_disk"]


def passes(case, variant):
    works, compatibility = behavior(case, variant)
    return works and compatibility == behavior(case, case["required"])[1]


def prompts(case, keep):
    options = case["options"][::(-1 if case["order"] else 1)]
    requirement = f"Compatibility requirement: {case['options'][case['required']]}."
    filler = "Repository metadata: " + ".".join(["index"] * 20)
    filler = filler[:len(requirement)].ljust(len(requirement), ".")
    user = (
        f"Task: {case['task']}\n"
        "Continue from the checkpoint below and implement the task.\n"
        f"Option A: {options[0]}. Option B: {options[1]}.\n"
        "Tools: read_requirements reopens the source file; apply_a and apply_b apply the named "
        "implementation; stop terminates. Respond with a single JSON object such as "
        '{"tool":"read_requirements"}. Apply actions finish the run.\n'
        "Checkpoint, chronological tool history:\n"
        f"read docs/requirements.txt -> {requirement if keep else filler}\n"
        "list source directory -> configuration handler found\n"
        "inspect handler -> both implementation options supported\n"
        "Choose your next tool."
    )
    return user, requirement


def episode(generate, case, keep):
    prompt, requirement = prompts(case, keep)
    trace = []
    passed = False
    for _ in range(3):
        reply = generate(prompt).strip()
        try:
            tool = json.loads(reply)["tool"]
        except (ValueError, TypeError, KeyError):
            tool = "invalid"
        if tool not in ("read_requirements", "apply_a", "apply_b", "stop"):
            tool = "invalid"
        trace.append(tool)
        if tool in ("apply_a", "apply_b"):
            passed = passes(case, int(tool == "apply_b") ^ case["order"])
            break
        if tool == "stop":
            break
        result = requirement if tool == "read_requirements" else "Invalid tool response. Use the specified JSON."
        prompt += f"\nTool action: {tool}\nTool return: {result}\nChoose your next tool."
    return {"passed": passed, "trace": trace}


def load_generator():
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

    name = "Qwen/Qwen3.5-9B"
    tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        name, local_files_only=True, device_map={"": 0}, dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.float16),
    ).eval()

    def generate(prompt):
        chat = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                             tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(chat, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        reply = tokenizer.decode(output[0, inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        return re.sub(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", r"\1", reply, flags=re.S)

    return generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("results/agent_resume_qwen35.json"))
    args = parser.parse_args()
    suite = list(cases())
    for case in suite:
        assert sum(passes(case, variant) for variant in range(2)) == 1
        assert len(prompts(case, True)[0]) == len(prompts(case, False)[0])
        correct = "apply_b" if case["required"] ^ case["order"] else "apply_a"
        replies = iter((json.dumps({"tool": "read_requirements"}), json.dumps({"tool": correct})))
        assert episode(lambda _: next(replies), case, False)["passed"]
    if args.selftest:
        print(f"{len(suite)} paired checkpoints validated")
        return
    generate = load_generator()
    runs = []
    for index, case in enumerate(suite):
        row = {"case": case["id"]}
        arms = ("keep", "drop") if index % 2 == 0 else ("drop", "keep")
        for arm in arms:
            row[arm] = episode(generate, case, arm == "keep")
        runs.append(row)
        print(f"{len(runs)}/{len(suite)}", flush=True)
    args.out.write_text(json.dumps(runs, indent=2) + "\n", encoding="utf-8")
    print("paired outcomes (keep, drop):", dict(Counter((r["keep"]["passed"], r["drop"]["passed"]) for r in runs)))


if __name__ == "__main__":
    main()
