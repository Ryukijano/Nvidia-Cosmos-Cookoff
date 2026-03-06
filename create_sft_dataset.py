import pandas as pd
import json
import os

# Load the selected positive clips
clips_df = pd.read_csv("./nexar_data/selected/positive_clips.csv")

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
Time-to-impact if no action: X.X seconds"""

# Create SFT dataset in Llava format
sft_data = []

for idx, row in clips_df.iterrows():
    clip_name = row['file_name']
    alert_time = row['time_of_alert']
    
    # Video path
    video_path = f"/teamspace/studios/this_studio/nexar_data/processed_clips/{clip_name.replace('.mp4', '_prealert.mp4')}"
    
    # Generate synthetic answer (risk detected 3 seconds before alert)
    risk_time = max(0, alert_time - 3)
    synthetic_answer = f"""</think>
Frame-by-frame reasoning:
Analyzing the dashcam footage, I observe vehicles and potential hazards developing over time. The situation evolves with agents moving into risk positions.

Critical risk detected at: {risk_time:.1f} seconds
Risk score: 4/5
At-risk agent: approaching vehicle
Explanation: Vehicle trajectory indicates potential collision corridor violation
Time-to-impact if no action: 3.0 seconds"""
    
    # Create Llava entry
    entry = {
        "id": f"nexar_{idx}",
        "video": video_path,
        "conversations": [
            {
                "from": "human", 
                "value": f"<video>\n{USER_PROMPT}"
            },
            {
                "from": "gpt",
                "value": synthetic_answer
            }
        ]
    }
    
    sft_data.append(entry)

# Save to JSON
os.makedirs("./nexar_data/sft", exist_ok=True)
with open("./nexar_data/sft/train.json", "w") as f:
    json.dump(sft_data, f, indent=2)

print(f"Created SFT dataset with {len(sft_data)} entries")
print("Saved to ./nexar_data/sft/train.json")

# Show sample
print("\nSample entry:")
print(json.dumps(sft_data[0], indent=2))
