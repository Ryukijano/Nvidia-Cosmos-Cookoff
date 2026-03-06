import os
from datasets import load_dataset
import shutil

# Load the UniDataPro real-time traffic video dataset (CCTV surveillance videos)
try:
    dataset = load_dataset("UniDataPro/real-time-traffic-video-dataset", split="train[:20]")
except Exception as e:
    print(f"Dataset loading failed: {e}")
    print("Attempting alternative download method...")
    # Fallback to manual download if needed
    dataset = None

# Ensure directory exists
os.makedirs("./nexar_data/sample_videos", exist_ok=True)

if dataset:
    # Download 20 CCTV traffic videos as samples
    for i in range(min(20, len(dataset))):
        try:
            # Get the video path
            video_entry = dataset[i]
            video_path = video_entry["video"]  # Local path after loading
            
            # Copy to our sample directory
            dest_path = f"./nexar_data/sample_videos/cctv_traffic_{i}.mp4"
            if os.path.exists(video_path):
                shutil.copy2(video_path, dest_path)
                print(f"Downloaded CCTV traffic video {i} to {dest_path}")
            else:
                print(f"Video path {video_path} not found for index {i}")
        except Exception as e:
            print(f"Failed to download video {i}: {e}")

    print("CCTV traffic video download complete!")
else:
    print("Please download videos manually from: https://huggingface.co/datasets/UniDataPro/real-time-traffic-video-dataset")
