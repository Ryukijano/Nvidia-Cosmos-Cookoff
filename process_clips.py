from huggingface_hub import hf_hub_download
import pandas as pd
import os
import cv2

# Load selected clips metadata
clips_df = pd.read_csv("./nexar_data/selected/positive_clips.csv")
print(f"Processing {len(clips_df)} clips")

# Create output directory
os.makedirs("./nexar_data/processed_clips", exist_ok=True)

# Download and process first 3 clips
for idx, row in clips_df.head(3).iterrows():
    clip_name = row['file_name']
    alert_time = row['time_of_alert']
    
    print(f"\nProcessing clip {idx+1}/3: {clip_name}")
    print(f"Alert time: {alert_time}s")
    
    # Download the video
    video_path = hf_hub_download(
        repo_id="nexar-ai/nexar_collision_prediction",
        filename=f"train/positive/{clip_name}",
        repo_type="dataset",
        local_dir="./nexar_data/videos/"
    )
    
    # Extract 10-second pre-alert window
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate start time (alert_time - 10 seconds)
    start_time = max(0, alert_time - 10)
    end_time = alert_time
    
    print(f"Extracting {end_time - start_time:.1f}s from {start_time:.1f}s to {end_time:.1f}s")
    
    # Create output video
    output_path = f"./nexar_data/processed_clips/{clip_name.replace('.mp4', '_prealert.mp4')}"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Extract frames
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
    current_time = start_time
    
    while current_time < end_time:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        current_time += 1/fps
    
    cap.release()
    out.release()
    print(f"Saved: {output_path}")

print("\n✅ Processing complete!")
