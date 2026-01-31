from huggface_hub import snapshot_download
import os
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
model_name = "Qwen3-0.6B"
save_path = os.path.join(DIR_PATH, f"models/{model_name}")
snapshot_download(
    repo_id = f"Qwen/{model_name}",
    local_dir = save_path,
    local_dir_use_symlinks = False,
    resume_download = True,
)