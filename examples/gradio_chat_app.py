#!/usr/bin/env python3

"""
Gradio Chat Interface for Local LLM Inference
Uses both vLLM and Ollama for model inference with GPU acceleration
"""

import gradio as gr
import requests
import json
import subprocess
import time
from typing import Generator, Tuple
import sys
import os

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class LLMInterface:
    """Interface for interacting with local LLM services."""
    
    def __init__(self):
        self.available_models = []
        self.check_services()
    
    def check_services(self):
        """Check which LLM services are available."""
        # Check Ollama
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                for line in lines:
                    if line.strip():
                        model_name = line.split()[0]
                        self.available_models.append(f"ollama:{model_name}")
        except:
            pass
    
    def chat_with_ollama(self, model: str, message: str, history: list) -> Generator[str, None, None]:
        """Stream chat response from Ollama."""
        model_name = model.replace("ollama:", "")
        
        try:
            # Build conversation context
            conversation = []
            for human, assistant in history:
                conversation.append({"role": "user", "content": human})
                conversation.append({"role": "assistant", "content": assistant})
            conversation.append({"role": "user", "content": message})
            
            # Use Ollama API for streaming
            payload = {
                "model": model_name,
                "messages": conversation,
                "stream": True
            }
            
            response = requests.post(
                "http://localhost:11434/api/chat",
                json=payload,
                stream=True
            )
            
            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        if 'message' in data and 'content' in data['message']:
                            chunk = data['message']['content']
                            full_response += chunk
                            yield full_response
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            yield f"Error: {str(e)}"
    
    def get_system_info(self) -> str:
        """Get system information for display."""
        try:
            # GPU info
            import torch
            gpu_info = ""
            if torch.cuda.is_available():
                gpu_info = f"""
**GPU Information:**
- Device: {torch.cuda.get_device_name(0)}
- Memory: {torch.cuda.get_device_properties(0).total_memory // (1024**3)} GB
- CUDA Version: {torch.version.cuda}
- PyTorch Version: {torch.__version__}
"""
            
            # Available models
            models_info = f"""
**Available Models:**
{chr(10).join(f"- {model}" for model in self.available_models)}
"""
            
            return f"{gpu_info}\n{models_info}"
        except:
            return "System info unavailable"

# Initialize LLM interface
llm_interface = LLMInterface()

def respond(message: str, history: list, model: str) -> Generator[Tuple[str, list], None, None]:
    """Generate streaming response."""
    if not message.strip():
        yield "", history
        return
    
    if not model:
        yield "", history + [(message, "Please select a model first.")]
        return
    
    # Add user message to history immediately
    new_history = history + [(message, "")]
    
    # Stream the response
    if model.startswith("ollama:"):
        for partial_response in llm_interface.chat_with_ollama(model, message, history):
            new_history[-1] = (message, partial_response)
            yield "", new_history
    else:
        # Fallback for unknown model types
        new_history[-1] = (message, "Unsupported model type.")
        yield "", new_history

def clear_history():
    """Clear chat history."""
    return []

def get_example_prompts():
    """Get example prompts for testing."""
    return [
        "Hello! How are you today?",
        "Explain machine learning in simple terms.",
        "What are the benefits of GPU acceleration?",
        "Write a Python function to calculate fibonacci numbers.",
        "What is the future of artificial intelligence?"
    ]

# Create Gradio interface
def create_interface():
    """Create the Gradio chat interface."""
    
    with gr.Blocks(
        title="Local LLM Chat Interface",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
        }
        .chat-message {
            padding: 10px;
            margin: 5px 0;
            border-radius: 10px;
        }
        """
    ) as demo:
        
        gr.Markdown("""
        # 🚀 Local LLM Chat Interface
        
        Chat with your local language models using GPU acceleration!
        Powered by vLLM, Ollama, and your RTX 4090.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Model selection
                model_dropdown = gr.Dropdown(
                    choices=llm_interface.available_models,
                    label="Select Model",
                    value=llm_interface.available_models[0] if llm_interface.available_models else None,
                    interactive=True
                )
                
                # System info
                with gr.Accordion("System Information", open=False):
                    system_info = gr.Markdown(llm_interface.get_system_info())
                
                # Example prompts
                with gr.Accordion("Example Prompts", open=False):
                    example_prompts = gr.Radio(
                        choices=get_example_prompts(),
                        label="Click to use:",
                        interactive=True
                    )
            
            with gr.Column(scale=3):
                # Chat interface
                chatbot = gr.Chatbot(
                    height=500,
                    label="Chat",
                    show_label=False,
                    avatar_images=(None, "🤖"),
                    bubble_full_width=False
                )
                
                with gr.Row():
                    msg_textbox = gr.Textbox(
                        placeholder="Type your message here...",
                        label="Message",
                        show_label=False,
                        scale=4
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                
                with gr.Row():
                    clear_btn = gr.Button("Clear History", variant="secondary")
                    refresh_models_btn = gr.Button("Refresh Models", variant="secondary")
        
        # Event handlers
        def send_message(message, history, model):
            return respond(message, history, model)
        
        def use_example_prompt(prompt):
            return prompt
        
        def refresh_models():
            llm_interface.check_services()
            return gr.Dropdown(choices=llm_interface.available_models)
        
        # Wire up events
        send_btn.click(
            send_message,
            inputs=[msg_textbox, chatbot, model_dropdown],
            outputs=[msg_textbox, chatbot]
        )
        
        msg_textbox.submit(
            send_message,
            inputs=[msg_textbox, chatbot, model_dropdown],
            outputs=[msg_textbox, chatbot]
        )
        
        clear_btn.click(clear_history, outputs=[chatbot])
        
        example_prompts.change(
            use_example_prompt,
            inputs=[example_prompts],
            outputs=[msg_textbox]
        )
        
        refresh_models_btn.click(
            refresh_models,
            outputs=[model_dropdown]
        )
        
        # Add footer
        gr.Markdown("""
        ---
        **Tips:**
        - Select a model from the dropdown to start chatting
        - Use example prompts to get started quickly
        - Clear history to start a fresh conversation
        - Check system information to verify GPU is detected
        
        **Note:** Make sure Ollama is running (`ollama serve`) for model access.
        """)
    
    return demo

if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()
    
    print("🚀 Starting Local LLM Chat Interface...")
    print(f"📊 Found {len(llm_interface.available_models)} available models")
    print("🌐 Open your browser to interact with the interface")
    
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Default Gradio port
        share=False,           # Set to True for public sharing via gradio.live
        debug=False
    )