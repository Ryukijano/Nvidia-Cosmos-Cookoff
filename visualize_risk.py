import cv2
import json
import re
from torchcodec import VideoDecoder
import numpy as np

def parse_bounding_boxes(text_output):
    """
    Parse bounding boxes from text output.
    Assumes format: Bounding boxes (normalized 0-1):
    Agent 1: [x1,y1,x2,y2]
    ...
    """
    bboxes = []
    bbox_section = re.search(r'Bounding boxes \(normalized 0-1\):(.*)', text_output, re.DOTALL)
    if bbox_section:
        lines = bbox_section.group(1).strip().split('\n')
        for line in lines:
            match = re.search(r'\[([\d\.]+),([\d\.]+),([\d\.]+),([\d\.]+)\]', line)
            if match:
                x1, y1, x2, y2 = map(float, match.groups())
                bboxes.append((x1, y1, x2, y2))
    return bboxes

def visualize_risk(video_path, text_output, output_path):
    """
    Visualize risk with bounding boxes and text overlay using torchcodec for decoding.
    """
    decoder = VideoDecoder(video_path)
    fps = decoder.metadata.average_fps
    height, width = decoder.metadata.height, decoder.metadata.width  # Assuming HWC order
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    bboxes = parse_bounding_boxes(text_output)
    
    num_frames = len(decoder)
    for frame_idx in range(num_frames):
        frame_tensor = decoder[frame_idx]
        frame_rgb = frame_tensor.permute(1, 2, 0).numpy()  # CHW to HWC RGB
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)  # To BGR for OpenCV
        
        # Draw bounding boxes (assuming they apply to the scene)
        for bbox in bboxes:
            x1 = int(bbox[0] * width)
            y1 = int(bbox[1] * height)
            x2 = int(bbox[2] * width)
            y2 = int(bbox[3] * height)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add text overlay on first few frames or all
        if frame_idx < int(fps * 5):  # First 5 seconds
            cv2.putText(frame, "RISK DETECTED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Visualization saved to {output_path}")

# Example usage
if __name__ == "__main__":
    video_path = "/teamspace/studios/this_studio/nexar_data/processed_clips/00822_prealert.mp4"
    text_output = """
    Critical risk detected at: 5.0 seconds
    Risk score: 4/5
    At-risk agent: White sedan
    Explanation: Vehicle approaching pedestrian at intersection.
    
    Bounding boxes (normalized 0-1):
    Agent 1: [0.1,0.2,0.3,0.4]
    Agent 2: [0.5,0.6,0.7,0.8]
    """
    output_path = "/teamspace/studios/this_studio/output_with_visualization.mp4"
    
    visualize_risk(video_path, text_output, output_path)
