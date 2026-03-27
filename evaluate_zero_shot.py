import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import cv2
from PIL import Image
import pandas as pd
import os
import json
from pathlib import Path

# Load Cosmos Reason 2 model
print("Loading Cosmos Reason 2-8B model...")
model_name = "nvidia/Cosmos-Reason2-8B"

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
)

processor = AutoProcessor.from_pretrained(model_name)
print("✅ Model loaded (FP16)")

# System and user prompts
SYSTEM_PROMPT = "You are Cosmos Risk Narrator, an AI safety agent analyzing dashcam footage from the ego vehicle's perspective. Your task is to identify developing risks before they become critical."

USER_PROMPT = """Analyze this dashcam video clip frame by frame from the ego vehicle's perspective.

Focus on:
1. Identify all agents (vehicles, pedestrians, cyclists) and their trajectories
2. Detect any agents entering collision corridors or unsafe situations
3. Predict when the risk becomes critical (provide timestamp in seconds)
4. Assess the physics of the situation (speeds, distances, reaction times)

Output format:
</think>
Frame-by-frame reasoning:
[Describe what you see and how the situation evolves]

Critical risk detected at: X.X seconds
Risk score: X/5
At-risk agent: [description]
Explanation: [clear sentence about why this is dangerous]
Time-to-impact if no action: X.X seconds

Bounding boxes (normalized 0-1):
Agent 1: [x1,y1,x2,y2]
Agent 2: [x1,y1,x2,y2]
..."""

def process_video(video_path, max_frames=40):
    """Process video at 4fps"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / 4)
    frames = []
    count = 0
    
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
        count += 1
    cap.release()
    
    print(f"Processed {len(frames)} frames")
    return frames

def run_risk_narrator(video_path):
    """Run Cosmos Reason 2 on video"""
    frames = process_video(video_path)
    
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "video", "video": frames, "fps": 4, "max_pixels": 768*768},
            {"type": "text", "text": USER_PROMPT}
        ]}
    ]
    
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)
    
    print("Generating risk assessment...")
    generated = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.2,
        do_sample=True,
        top_p=0.9
    )
    
    output = processor.batch_decode(generated, skip_special_tokens=True)[0]
    assistant_part = output.split("<|im_start|>assistant")[-1].strip()
    return assistant_part

def parse_output(output):
    """Parse the model output for metrics"""
    import re
    
    # Extract critical risk detected at
    risk_time_match = re.search(r'Critical risk detected at: (\d+\.?\d*) seconds', output)
    risk_time = float(risk_time_match.group(1)) if risk_time_match else None
    
    # Extract risk score
    risk_score_match = re.search(r'Risk score: (\d+)/5', output)
    risk_score = int(risk_score_match.group(1)) if risk_score_match else None
    
    return {
        'predicted_risk_time': risk_time,
        'predicted_risk_score': risk_score,
        'full_output': output
    }

# Load test data (assuming we have some test clips)
# For demo, use one test clip
test_clip = "/teamspace/studios/this_studio/nexar_data/processed_clips/00822_prealert.mp4"  # Placeholder
annotated_time = 5.0  # Placeholder from metadata

print(f"\nEvaluating on test clip: {test_clip}")
result = run_risk_narrator(test_clip)
parsed = parse_output(result)

print("\nParsed Results:")
print(f"Predicted Risk Time: {parsed['predicted_risk_time']}")
print(f"Predicted Risk Score: {parsed['predicted_risk_score']}")
print(f"Annotated Time: {annotated_time}")

# Compute metrics
if parsed['predicted_risk_time'] is not None:
    time_error = abs(parsed['predicted_risk_time'] - annotated_time)
    print(f"Time Prediction Error: {time_error} seconds")
    # Consider accurate if within 2 seconds
    accurate = time_error <= 2.0
    print(f"Accurate Prediction: {accurate}")
else:
    print("No risk time predicted")

print("\nFull Output:")
print(result)
