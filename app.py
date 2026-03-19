import gradio as gr
import os
import subprocess
import sys

def run_pipeline_gradio(video_file):
    # video_file is already the path from gr.Video
    # Run pipeline with the video path
    result = subprocess.run([sys.executable, "main_pipeline.py", video_file], capture_output=True, text=True, timeout=1200, cwd=".")
    output = result.stdout + "\n" + result.stderr

    # Check for generated files
    bboxes_img = "bboxes_visualization.png" if os.path.exists("bboxes_visualization.png") else None
    risk_img = "risk_visualization.png" if os.path.exists("risk_visualization.png") else None
    generated_video = "./cosmos-predict2.5/outputs/pipeline_simulation/cosmos_pipeline_simulation.mp4" if os.path.exists("./cosmos-predict2.5/outputs/pipeline_simulation/cosmos_pipeline_simulation.mp4") else None

    return {
        "pipeline_output": output,
        "bboxes_visualization": bboxes_img,
        "risk_visualization": risk_img,
        "simulation_video": generated_video
    }

with gr.Blocks() as iface:
    gr.Markdown("# Pure Cosmos Pipeline Demo")
    gr.Markdown("AI-powered collision detection, analysis, and counterfactual simulation with iterative feedback loop")
    
    with gr.Row():
        video_input = gr.File(label="Upload Dashcam Video (MP4)")
    
    with gr.Row():
        output_text = gr.Textbox(label="Pipeline Output", lines=20)
        bboxes_img = gr.Image(label="Bounding Boxes Visualization")
        risk_img = gr.Image(label="Risk Score Visualization")
        sim_video = gr.Video(label="Generated Simulation Video")
    
    submit_btn = gr.Button("Run Pipeline")
    
    def process_video(video_file):
        if video_file is None:
            return "Please upload a video file first.", None, None, None
        
        results = run_pipeline_gradio(video_file)
        return (
            results["pipeline_output"],
            results["bboxes_visualization"],
            results["risk_visualization"],
            results["simulation_video"]
        )
    
    submit_btn.click(
        process_video,
        inputs=[video_input],
        outputs=[output_text, bboxes_img, risk_img, sim_video]
    )

if __name__ == "__main__":
    iface.launch(share=True)
