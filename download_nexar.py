from huggingface_hub import hf_hub_download
import pandas as pd
import os
import json

# Download Nexar dataset metadata for positive training clips
print("Downloading Nexar collision prediction dataset metadata...")
positive_metadata = hf_hub_download(
    repo_id="nexar-ai/nexar_collision_prediction",
    filename="train/positive/metadata.csv",
    repo_type="dataset",
    local_dir="./nexar_data/"
)

# Load metadata
df = pd.read_csv(positive_metadata)
print(f"Dataset loaded: {len(df)} positive clips")

# Get first 10 clips for processing
positive_clips = df.head(10)
print(f"Selected {len(positive_clips)} positive clips for processing")

# Save selected clips info
os.makedirs("./nexar_data/selected", exist_ok=True)
positive_clips.to_csv("./nexar_data/selected/positive_clips.csv", index=False)

print("Positive clips saved to ./nexar_data/selected/positive_clips.csv")
print(positive_clips[['file_name', 'time_of_alert', 'time_to_accident']].head())
