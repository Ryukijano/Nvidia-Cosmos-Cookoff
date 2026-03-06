from huggingface_hub import snapshot_download
import os

# Download the BADAS-Open dataset (Nexar collision prediction)
repo_id = "nexar-ai/nexar_collision_prediction"
local_dir = "./nexar_data/badas_dataset"
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

print("Downloading BADAS-Open dataset from Hugging Face...")
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    token=token,
    resume_download=True
)
print("Download completed.")
