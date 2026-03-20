from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-0.6B",
    local_dir="./model/Qwen3-0.6B",
    local_dir_use_symlinks=False,
)

# snapshot_download(
#     repo_id="HuggingFaceTB/SmolLM2-360M-Instruct",
#     local_dir="./model/SmolLM2-360M-Instruct",
#     local_dir_use_symlinks=False,
# )


if __name__ == "__main__":
    from transformers import pipeline

    # pipe = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-360M-Instruct")

    pipe = pipeline("text-generation", model="./model/Qwen3-0.6B")
    messages = [
        {"role": "user", "content": "Who are you?"},
    ]
    pipe(messages)
