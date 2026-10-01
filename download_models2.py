from huggingface_hub import snapshot_download

# post-training lineage (OLMo-2), R1-distill vs its base, and scale (Qwen3-14B)
REPOS = ["allenai/OLMo-2-1124-7B", "allenai/OLMo-2-1124-7B-SFT", "allenai/OLMo-2-1124-7B-DPO",
         "allenai/OLMo-2-1124-7B-Instruct", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "Qwen/Qwen2.5-Math-7B",
         "Qwen/Qwen3-14B"]
for repo in REPOS:
    print(repo, snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.bin", "*.txt", "*.model",
                                                         "*.py", "*.tiktoken", "*.jinja"]), flush=True)
