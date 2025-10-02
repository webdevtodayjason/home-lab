#!/usr/bin/env python3

"""
Streamlit Dashboard for GPU Monitoring and LLM Management
Monitor your RTX 4090 performance and manage local LLM models
"""

import streamlit as st
import subprocess
import json
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import torch
import psutil
import requests
from typing import Dict, List, Optional
import threading
import queue

# Page configuration
st.set_page_config(
    page_title="LLM GPU Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .gpu-status {
        font-size: 1.2rem;
        font-weight: bold;
    }
    .model-card {
        border: 1px solid #ddd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        background-color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

class SystemMonitor:
    """System monitoring utilities."""
    
    @staticmethod
    def get_gpu_info() -> Dict:
        """Get GPU information using nvidia-smi."""
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, check=True)
            
            if result.stdout.strip():
                data = result.stdout.strip().split(', ')
                return {
                    'name': data[0],
                    'memory_total': int(data[1]),
                    'memory_used': int(data[2]),
                    'memory_free': int(data[3]),
                    'utilization': int(data[4]),
                    'temperature': int(data[5]),
                    'power_draw': float(data[6]) if data[6] != '[N/A]' else 0
                }
        except (subprocess.CalledProcessError, IndexError, ValueError):
            return {}
    
    @staticmethod
    def get_system_info() -> Dict:
        """Get system information."""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory': psutil.virtual_memory(),
                'disk': psutil.disk_usage('/'),
                'boot_time': datetime.fromtimestamp(psutil.boot_time())
            }
        except:
            return {}
    
    @staticmethod
    def get_torch_info() -> Dict:
        """Get PyTorch information."""
        try:
            return {
                'version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
                'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        except:
            return {}

class OllamaManager:
    """Manage Ollama models and services."""
    
    @staticmethod
    def get_models() -> List[Dict]:
        """Get list of installed Ollama models."""
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                models = []
                for line in lines:
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 3:
                            models.append({
                                'name': parts[0],
                                'id': parts[1],
                                'size': parts[2],
                                'modified': ' '.join(parts[3:]) if len(parts) > 3 else 'Unknown'
                            })
                return models
        except:
            pass
        return []
    
    @staticmethod
    def is_ollama_running() -> bool:
        """Check if Ollama service is running."""
        try:
            response = requests.get('http://localhost:11434/api/tags', timeout=2)
            return response.status_code == 200
        except:
            return False
    
    @staticmethod
    def pull_model(model_name: str) -> str:
        """Pull a new model from Ollama."""
        try:
            result = subprocess.run(['ollama', 'pull', model_name], 
                                  capture_output=True, text=True, timeout=300)
            return result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return "Model pull timed out"
        except Exception as e:
            return f"Error: {str(e)}"

def create_gpu_chart(gpu_data: Dict) -> go.Figure:
    """Create GPU utilization chart."""
    fig = go.Figure()
    
    # GPU Utilization
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=gpu_data.get('utilization', 0),
        domain={'x': [0, 0.5], 'y': [0, 1]},
        title={'text': "GPU Utilization (%)"},
        gauge={'axis': {'range': [None, 100]},
               'bar': {'color': "darkblue"},
               'steps': [
                   {'range': [0, 50], 'color': "lightgray"},
                   {'range': [50, 80], 'color': "yellow"},
                   {'range': [80, 100], 'color': "red"}],
               'threshold': {'line': {'color': "red", 'width': 4},
                           'thickness': 0.75, 'value': 90}}))
    
    # Memory Usage
    memory_used = gpu_data.get('memory_used', 0)
    memory_total = gpu_data.get('memory_total', 1)
    memory_percent = (memory_used / memory_total) * 100 if memory_total > 0 else 0
    
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=memory_percent,
        domain={'x': [0.5, 1], 'y': [0, 1]},
        title={'text': "Memory Usage (%)"},
        gauge={'axis': {'range': [None, 100]},
               'bar': {'color': "darkgreen"},
               'steps': [
                   {'range': [0, 50], 'color': "lightgray"},
                   {'range': [50, 80], 'color': "yellow"},
                   {'range': [80, 100], 'color': "red"}],
               'threshold': {'line': {'color': "red", 'width': 4},
                           'thickness': 0.75, 'value': 90}}))
    
    fig.update_layout(height=400, showlegend=False)
    return fig

def main():
    """Main dashboard function."""
    
    # Title and header
    st.title("🚀 LLM GPU Dashboard")
    st.markdown("Monitor your RTX 4090 and manage local language models")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Controls")
        
        # Refresh rate
        refresh_rate = st.selectbox(
            "Refresh Rate",
            options=[1, 2, 5, 10, 30],
            index=2,
            help="How often to refresh the dashboard (seconds)"
        )
        
        # Auto refresh toggle
        auto_refresh = st.checkbox("Auto Refresh", value=True)
        
        if st.button("🔄 Refresh Now"):
            st.experimental_rerun()
        
        st.divider()
        
        # System info
        st.subheader("💻 System")
        torch_info = SystemMonitor.get_torch_info()
        
        if torch_info.get('cuda_available'):
            st.success("✅ CUDA Available")
            st.info(f"PyTorch: {torch_info.get('version', 'Unknown')}")
            st.info(f"CUDA: {torch_info.get('cuda_version', 'Unknown')}")
        else:
            st.error("❌ CUDA Not Available")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 GPU Monitoring")
        
        # Get GPU data
        gpu_data = SystemMonitor.get_gpu_info()
        
        if gpu_data:
            # GPU metrics
            metric_cols = st.columns(4)
            
            with metric_cols[0]:
                st.metric(
                    label="GPU Name",
                    value=gpu_data.get('name', 'Unknown')
                )
            
            with metric_cols[1]:
                st.metric(
                    label="Temperature",
                    value=f"{gpu_data.get('temperature', 0)}°C",
                    delta=None
                )
            
            with metric_cols[2]:
                st.metric(
                    label="Power Draw",
                    value=f"{gpu_data.get('power_draw', 0):.1f}W",
                    delta=None
                )
            
            with metric_cols[3]:
                memory_used = gpu_data.get('memory_used', 0)
                memory_total = gpu_data.get('memory_total', 0)
                st.metric(
                    label="Memory",
                    value=f"{memory_used} MB",
                    delta=f"{memory_total - memory_used} MB free"
                )
            
            # GPU utilization charts
            st.plotly_chart(create_gpu_chart(gpu_data), use_container_width=True)
            
        else:
            st.error("❌ Unable to get GPU information. Make sure nvidia-smi is available.")
    
    with col2:
        st.header("🤖 Model Management")
        
        # Ollama status
        ollama_running = OllamaManager.is_ollama_running()
        
        if ollama_running:
            st.success("✅ Ollama Running")
        else:
            st.error("❌ Ollama Not Running")
            st.info("Start Ollama with: `ollama serve`")
        
        # Model list
        models = OllamaManager.get_models()
        
        if models:
            st.subheader(f"📚 Installed Models ({len(models)})")
            
            for model in models:
                with st.container():
                    st.markdown(f"""
                    <div class="model-card">
                        <strong>{model['name']}</strong><br>
                        <small>Size: {model['size']} | Modified: {model['modified']}</small>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No models found. Pull a model to get started.")
        
        # Model pulling
        st.subheader("⬇️ Pull New Model")
        
        popular_models = [
            "llama3.2:3b",
            "llama3.1:8b", 
            "gemma2:9b",
            "codellama:7b",
            "mistral:7b",
            "phi3:mini"
        ]
        
        model_to_pull = st.selectbox(
            "Select Model",
            options=["Custom..."] + popular_models,
            index=0
        )
        
        if model_to_pull == "Custom...":
            model_to_pull = st.text_input("Enter model name:")
        
        if st.button("Pull Model") and model_to_pull:
            with st.spinner(f"Pulling {model_to_pull}..."):
                result = OllamaManager.pull_model(model_to_pull)
                if "successfully" in result.lower():
                    st.success(f"Successfully pulled {model_to_pull}")
                else:
                    st.error(f"Failed to pull model: {result}")
    
    # System metrics row
    st.header("💻 System Metrics")
    
    system_info = SystemMonitor.get_system_info()
    if system_info:
        sys_cols = st.columns(4)
        
        with sys_cols[0]:
            st.metric(
                label="CPU Usage",
                value=f"{system_info.get('cpu_percent', 0):.1f}%"
            )
        
        with sys_cols[1]:
            memory = system_info.get('memory')
            if memory:
                st.metric(
                    label="RAM Usage",
                    value=f"{memory.percent:.1f}%",
                    delta=f"{memory.used // (1024**3):.1f}GB used"
                )
        
        with sys_cols[2]:
            disk = system_info.get('disk')
            if disk:
                st.metric(
                    label="Disk Usage", 
                    value=f"{(disk.used / disk.total * 100):.1f}%",
                    delta=f"{disk.free // (1024**3):.1f}GB free"
                )
        
        with sys_cols[3]:
            uptime = datetime.now() - system_info.get('boot_time', datetime.now())
            st.metric(
                label="Uptime",
                value=f"{uptime.days}d {uptime.seconds // 3600}h"
            )
    
    # Auto refresh
    if auto_refresh:
        time.sleep(refresh_rate)
        st.experimental_rerun()
    
    # Footer
    st.divider()
    st.markdown("""
    **💡 Tips:**
    - Monitor GPU utilization to optimize model performance
    - Keep GPU temperature below 80°C for optimal performance
    - Use smaller models for testing, larger models for production
    - Check memory usage before loading large models
    """)

if __name__ == "__main__":
    main()