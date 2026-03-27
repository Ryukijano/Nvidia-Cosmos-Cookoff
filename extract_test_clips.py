import os
import cv2
import pandas as pd
from pathlib import Path

# Load test metadata
metadata_path = "./nexar_data/hf_dataset/time_to_accident_test_map.csv"
metadata = {}
with open(metadata_path, 'r') as f:
    lines = f.readlines()[1:]  # Skip first line
    for line in lines:
        parts = line.strip().split(',')
        video_id = parts[0]
        if len(parts) > 1 and parts[1]:
            time_of_alert = float(parts[1]) / 100  # Assuming times are in centiseconds or something, adjust if needed
            metadata[video_id] = time_of_alert

print(f"Loaded metadata for {len(metadata)} videos")

# Input directory
input_dir = "./nexar_data/hf_dataset/test-public/positive"

# Output directory
output_dir = "./nexar_data/test_processed_clips"
os.makedirs(output_dir, exist_ok=True)

# Process all videos
mp4_files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]
print(f"Found {len(mp4_files)} MP4 files")

# Process videos until 5 are saved
saved_count = 0
for mp4_file in mp4_files:
    if saved_count >= 5:
        break
    video_id = mp4_file.split('.')[0]  # Assuming filename is video_id.mp4
    if video_id in metadata:
        time_of_alert = metadata[video_id]
        video_path = os.path.join(input_dir, mp4_file)
        
        # Load video with OpenCV
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Extract 10s pre-alert
        start_time = max(0, time_of_alert - 10)
        end_time = time_of_alert
        
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        # Sample at 4fps
        frame_interval = int(fps / 4)
        
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for frame_idx in range(start_frame, end_frame, frame_interval):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        
        if frames:
            # Save as MP4
            output_path = os.path.join(output_dir, f"{video_id}_prealert.mp4")
            height, width, _ = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 4.0, (width, height))
            
            for frame in frames:
                out.write(frame)
            
            out.release()
            print(f"Saved {output_path}, alert at {time_of_alert}s")
            saved_count += 1
        else:
            print(f"No frames for {video_id}")
    else:
        print(f"No metadata for {video_id}")

print("Preprocessing completed.")
