from huggingface_hub import snapshot_download
import os

# Download the Nexar dataset
repo_id = "nexar-ai/nexar_collision_prediction"
local_dir = "./nexar_data/hf_dataset"

print("Downloading Nexar dataset from Hugging Face...")
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    resume_download=True
)
print("Download completed.")
