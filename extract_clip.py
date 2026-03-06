import cv2
import os

def extract_pre_alert_clip(video_path, alert_time, output_path="./extracted_clip.mp4"):
    """Extract a context clip around the alert time at 4fps."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = float(total_frames / fps) if fps else 0.0
    
    start_time = max(0, alert_time - 8)
    end_time = min(duration_sec, alert_time + 3)
    
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    frame_interval = max(1, int(round(fps / 4)))
    
    frames = []
    for frame_idx in range(start_frame, end_frame, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    
    if frames:
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 4.0, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        print(f"Extracted clip saved to {output_path} ({start_time:.2f}s to {end_time:.2f}s)")
    else:
        print("No frames extracted")

if __name__ == "__main__":
    video_path = "./nexar_data/sample_videos/sample_dashcam_2.mp4"
    alert_time = 5.0  # assumed or from BADAS
    extract_pre_alert_clip(video_path, alert_time)
