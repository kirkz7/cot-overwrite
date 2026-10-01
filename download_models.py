from huggingface_hub import snapshot_download

for repo in ["Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B-Base", "microsoft/Phi-4-mini-instruct", "Qwen/Qwen3-8B"]:
    print(repo, snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.py", "*.tiktoken"]), flush=True)
