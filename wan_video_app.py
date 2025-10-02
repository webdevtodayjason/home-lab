#!/usr/bin/env python3

"""
Wan2.2 Video Generation Web Interface
A Gradio-based web interface for text-to-video and image-to-video generation using Wan2.2 TI2V-5B model
"""

import gradio as gr
import torch
import os
import tempfile
import uuid
from pathlib import Path
from PIL import Image
import numpy as np
# Conditional import to handle version compatibility
try:
    from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
    DIFFSYNTH_AVAILABLE = True
except ImportError as e:
    print(f"DiffSynth import error: {e}")
    WanVideoPipeline = None
    ModelConfig = None
    DIFFSYNTH_AVAILABLE = False

# Configuration
MODEL_PATH = "/home/devops/models/Wan2.2-TI2V-5B"
OUTPUT_DIR = "/home/devops/admin-panel-project/outputs/videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global pipeline variable
pipeline = None

def load_model():
    """Load the Wan2.2 TI2V-5B model pipeline"""
    global pipeline
    
    if not DIFFSYNTH_AVAILABLE:
        return "❌ DiffSynth-Studio not available. Please check installation."
    
    try:
        print("Loading Wan2.2 TI2V-5B model...")
        
        # Configure model paths
        model_configs = [
            ModelConfig(
                model_id=MODEL_PATH,
                origin_file_pattern="diffusion_pytorch_model*.safetensors",
            ),
            ModelConfig(
                model_id=MODEL_PATH,
                origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
            ),
            ModelConfig(
                model_id=MODEL_PATH + "/google/umt5-xxl",
            )
        ]
        
        pipeline = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda" if torch.cuda.is_available() else "cpu",
            model_configs=model_configs
        )
        
        print("✅ Wan2.2 TI2V-5B model loaded successfully!")
        return "✅ Model loaded successfully!"
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return f"❌ Error loading model: {str(e)}"

def generate_video(
    prompt: str,
    image: Image.Image = None,
    num_frames: int = 24,
    height: int = 720,
    width: int = 1280,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    fps: int = 24,
    progress=gr.Progress()
):
    """Generate video from text prompt and optional image"""
    
    if pipeline is None:
        return "❌ Model not loaded. Please load the model first.", None
    
    if not prompt.strip():
        return "❌ Please enter a text prompt.", None
    
    try:
        progress(0, desc="Starting video generation...")
        
        # Generate unique filename
        video_id = str(uuid.uuid4())[:8]
        output_path = os.path.join(OUTPUT_DIR, f"wan_video_{video_id}.mp4")
        
        progress(0.1, desc="Preparing generation parameters...")
        
        # Prepare generation parameters
        generation_params = {
            "prompt": prompt,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
        }
        
        # Add image if provided (Image-to-Video)
        if image is not None:
            generation_params["image"] = image
            mode = "Image-to-Video"
        else:
            mode = "Text-to-Video"
        
        progress(0.2, desc=f"Generating {mode} with {num_frames} frames...")
        
        # Generate video
        video_frames = pipeline(**generation_params)
        
        progress(0.8, desc="Saving video...")
        
        # Save video using DiffSynth utilities
        try:
            from diffsynth import save_video
            save_video(video_frames, output_path, fps=fps)
        except ImportError:
            # Fallback video saving method
            import imageio
            imageio.mimsave(output_path, video_frames, fps=fps)
        
        progress(1.0, desc="Video generation complete!")
        
        # Generate result message
        result_msg = f"""✅ Video generated successfully!
        
**Generation Settings:**
- Mode: {mode}
- Frames: {num_frames}
- Resolution: {width}x{height}
- FPS: {fps}
- Steps: {num_inference_steps}
- Guidance Scale: {guidance_scale}
- Prompt: "{prompt}"

**Output:** {output_path}"""
        
        return result_msg, output_path
        
    except Exception as e:
        error_msg = f"❌ Error generating video: {str(e)}"
        print(error_msg)
        return error_msg, None

def get_example_prompts():
    """Get example prompts for video generation"""
    return [
        "A majestic dragon flying over a medieval castle at sunset",
        "Ocean waves crashing against rocky cliffs in slow motion",
        "A cat wearing sunglasses driving a convertible car",
        "Cherry blossoms falling in a peaceful Japanese garden",
        "Astronaut floating in space with Earth in the background",
        "Time-lapse of a flower blooming in a sunny meadow",
        "Northern lights dancing over a snowy mountain landscape",
        "Underwater scene with colorful fish swimming around coral reef"
    ]

def create_interface():
    """Create the Gradio interface"""
    
    # Custom CSS for dark theme
    css = """
    .gradio-container {
        background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
    }
    .gr-button-primary {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        border: none;
    }
    """
    
    with gr.Blocks(
        title="Wan2.2 Video Generation",
        theme=gr.themes.Soft(primary_hue="blue"),
        css=css
    ) as interface:
        
        gr.Markdown("""
        # 🎬 Wan2.2 Video Generation Studio
        
        Generate high-quality videos from text prompts or images using the state-of-the-art Wan2.2 TI2V-5B model.
        
        **Features:**
        - 🎯 Text-to-Video generation
        - 🖼️ Image-to-Video generation  
        - 📹 720P @ 24fps output
        - 🚀 Optimized for RTX 4090
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎮 Model Control")
                
                model_status = gr.Textbox(
                    label="Model Status",
                    value="❌ Model not loaded",
                    interactive=False
                )
                
                load_btn = gr.Button(
                    "🚀 Load Wan2.2 Model",
                    variant="primary",
                    size="lg"
                )
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 📝 Generation Settings")
                
                prompt = gr.Textbox(
                    label="Text Prompt",
                    placeholder="Describe the video you want to generate...",
                    lines=3,
                    max_lines=5
                )
                
                image = gr.Image(
                    label="Input Image (Optional - for Image-to-Video)",
                    type="pil",
                    height=200
                )
                
                with gr.Row():
                    num_frames = gr.Slider(
                        label="Number of Frames",
                        minimum=8,
                        maximum=48,
                        value=24,
                        step=8
                    )
                    fps = gr.Slider(
                        label="FPS",
                        minimum=8,
                        maximum=30,
                        value=24,
                        step=1
                    )
                
                with gr.Row():
                    width = gr.Slider(
                        label="Width",
                        minimum=512,
                        maximum=1280,
                        value=1280,
                        step=64
                    )
                    height = gr.Slider(
                        label="Height", 
                        minimum=512,
                        maximum=720,
                        value=720,
                        step=64
                    )
                
                with gr.Row():
                    num_inference_steps = gr.Slider(
                        label="Inference Steps (Quality)",
                        minimum=20,
                        maximum=100,
                        value=50,
                        step=10
                    )
                    guidance_scale = gr.Slider(
                        label="Guidance Scale",
                        minimum=1.0,
                        maximum=20.0,
                        value=7.5,
                        step=0.5
                    )
                
                generate_btn = gr.Button(
                    "🎬 Generate Video",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=2):
                gr.Markdown("### 🎥 Generated Video")
                
                result_text = gr.Textbox(
                    label="Generation Result",
                    lines=8,
                    max_lines=15,
                    interactive=False
                )
                
                output_video = gr.Video(
                    label="Generated Video",
                    height=400
                )
        
        # Example prompts section
        gr.Markdown("### 💡 Example Prompts")
        example_prompts = get_example_prompts()
        
        with gr.Row():
            for i in range(0, len(example_prompts), 2):
                with gr.Column():
                    if i < len(example_prompts):
                        gr.Button(
                            example_prompts[i],
                            size="sm"
                        ).click(
                            fn=lambda x=example_prompts[i]: x,
                            outputs=prompt
                        )
                    if i+1 < len(example_prompts):
                        gr.Button(
                            example_prompts[i+1],
                            size="sm"
                        ).click(
                            fn=lambda x=example_prompts[i+1]: x,
                            outputs=prompt
                        )
        
        # Model info
        gr.Markdown("""
        ---
        
        ### 📊 Model Information
        - **Model**: Wan2.2 TI2V-5B (720P High-Efficiency)
        - **Compression**: 16×16×4 VAE 
        - **Max Resolution**: 1280x720 @ 24fps
        - **GPU Memory**: ~12-16GB (RTX 4090 optimized)
        - **Generation Time**: ~2-5 minutes per video
        
        ### 💡 Tips for Best Results
        - Use descriptive, detailed prompts
        - For I2V: Use high-quality input images
        - Lower inference steps for faster generation
        - Higher guidance scale for more prompt adherence
        """)
        
        # Event handlers
        load_btn.click(
            fn=load_model,
            outputs=model_status
        )
        
        generate_btn.click(
            fn=generate_video,
            inputs=[
                prompt,
                image,
                num_frames,
                height,
                width,
                num_inference_steps,
                guidance_scale,
                fps
            ],
            outputs=[result_text, output_video],
            show_progress=True
        )
    
    return interface

if __name__ == "__main__":
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("🚀 Starting Wan2.2 Video Generation Server...")
    print(f"📁 Model Path: {MODEL_PATH}")
    print(f"📁 Output Directory: {OUTPUT_DIR}")
    
    # Create and launch interface
    interface = create_interface()
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7870,
        share=False,
        show_error=True,
        quiet=False
    )
