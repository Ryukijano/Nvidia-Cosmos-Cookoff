import pandas as pd
import requests
import os
from pathlib import Path

# Create directories
os.makedirs("./nexar_data/test/positive", exist_ok=True)
os.makedirs("./nexar_data/test_clips", exist_ok=True)

# Download test positive metadata
metadata_url = "https://nexar-dataset.s3.amazonaws.com/v1.0/test/positive/metadata.csv"
metadata_path = "./nexar_data/test/positive/metadata.csv"

if not os.path.exists(metadata_path):
    print("Downloading test metadata...")
    response = requests.get(metadata_url)
    with open(metadata_path, 'wb') as f:
        f.write(response.content)
    print("Downloaded test metadata")

# Load test positive metadata
df = pd.read_csv(metadata_path)

# Select first 5 positive clips
selected_clips = df.head(5)

print(f"Selected {len(selected_clips)} test clips")

# Download clips
base_url = "https://nexar-dataset.s3.amazonaws.com/v1.0/test/positive/"

for idx, row in selected_clips.iterrows():
    clip_name = row['file_name']
    video_url = base_url + clip_name
    
    output_path = f"./nexar_data/test_clips/{clip_name}"
    
    if not os.path.exists(output_path):
        print(f"Downloading {clip_name}...")
        response = requests.get(video_url, stream=True)
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {clip_name}")
    else:
        print(f"Already exists: {clip_name}")

print("Test clips downloaded")
