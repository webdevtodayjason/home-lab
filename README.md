# 🚀 LLM Admin Panel & AI Development Environment

A comprehensive web-based admin panel for managing multiple LLM-related services with real-time status monitoring and AI development tools.

## 🏗️ System Architecture

This setup provides a complete AI development environment with:
- **Web-based Admin Panel**: Flask dashboard for service management
- **AI Development Tools**: JupyterLab, Streamlit, Gradio interfaces
- **Local LLM Services**: Ollama API with multiple models
- **Real-time Monitoring**: GPU metrics, service status, system health
- **External Access**: All services accessible remotely via network IP

## 📱 Services Overview

| Service | Port | Purpose | External URL | Status |
|---------|------|---------|--------------|--------|
| **Admin Panel** | 5000 | Main dashboard & control | http://10.11.1.105:5000 | ✅ |
| **JupyterLab** | 8888 | AI/ML notebooks & development | http://10.11.1.105:8888 | ✅ |
| **Gradio Chat** | 7860 | Interactive LLM chat interface | http://10.11.1.105:7860 | ✅ |
| **Streamlit** | 8501 | GPU monitoring & model management | http://10.11.1.105:8501 | ✅ |
| **Open WebUI** | 3000 | ChatGPT-like web interface | http://10.11.1.105:3000 | ✅ |
| **Wan2.2 Video Gen** | 7870 | AI video generation (T2V & I2V) | http://10.11.1.105:7870 | ✅ |
| **Ollama API** | 11434 | Local LLM inference service | http://10.11.1.105:11434 | ✅ |

## 🖥️ Server Configuration

### Hardware Environment
- **OS**: Ubuntu 22.04 LTS
- **GPU**: NVIDIA GeForce RTX 4090 (24GB VRAM)
- **CPU**: Multi-core processor
- **RAM**: 32GB+ recommended
- **Storage**: 500GB+ SSD space
- **Network**: Static IP: 10.11.1.105

### Software Stack
- **Python**: 3.12+ (Miniconda/Anaconda)
- **CUDA**: 13.0+ drivers
- **Docker**: Not used (native installation)
- **Web Server**: Flask development server + individual service servers
- **Process Manager**: Systemd (Ollama), bash scripts (others)

## 🛠️ Complete Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/webdevtodayjason/home-lab.git
cd home-lab
```

### 2. System Prerequisites

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y curl wget git build-essential python3-pip lsof

# Install NVIDIA drivers (if not already installed)
sudo ubuntu-drivers autoinstall
sudo reboot
```

### 3. Install Miniconda

```bash
# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

### 4. Install Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Enable and start Ollama service
sudo systemctl enable ollama
sudo systemctl start ollama

# Download models
ollama pull llama3.2:3b
ollama pull gemma2:27b
ollama pull qwen2.5vl:latest
ollama pull deepseek-r1:latest

# Download Wan2.2 models
mkdir -p /home/devops/models
cd /home/devops/models
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./Wan2.2-TI2V-5B
```

### 5. Python Environment Setup

```bash
# Create conda environment
conda create -n llm-env python=3.12 -y
conda activate llm-env

# Install core packages
pip install flask jupyter jupyterlab gradio streamlit
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate
pip install nvidia-ml-py3 psutil requests

# Install Open WebUI
pip install open-webui

# Install DiffSynth-Studio for Wan2.2 Video Generation
cd /home/devops
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

### 6. Configure Services

```bash
# Make scripts executable
chmod +x examples/launch_interfaces.sh
chmod +x examples/stop_interfaces.sh

# Update IP addresses in admin panel
# Edit app.py if your server IP is different from 10.11.1.105
```

## 🚀 Starting the Environment

### Method 1: Admin Panel (Recommended)
```bash
# Start admin panel
python app.py

# Access dashboard at: http://10.11.1.105:5000
# Login: admin / Dragon@123!@#
# Click "Start All" button
```

### Method 2: Manual Service Launch
```bash
# Start all services
cd examples
./launch_interfaces.sh

# Start admin panel separately
cd ../
python app.py
```

## 🔧 AI Development Environment

### JupyterLab Features
- **CUDA Support**: Full GPU acceleration for PyTorch/TensorFlow
- **Pre-installed Libraries**: 
  - PyTorch, Transformers, Datasets
  - Accelerate, PEFT, BitsAndBytes
  - Pandas, NumPy, Matplotlib, Seaborn
- **Model Development**: Fine-tuning, inference, evaluation
- **Data Science**: Full notebook environment with plotting
- **External Access**: No authentication, direct external connections

### Available Models (via Ollama)
```bash
# List installed models
ollama list

# Available models:
- llama3.2:3b (3.2B params) - Fast inference
- gemma2:27b (27.2B params) - High quality
- qwen2.5vl:latest (Vision + Language)
- deepseek-r1:latest (Code generation)
- llama3.1:8b (General purpose)
```

### Wan2.2 Video Generation Features
- **Text-to-Video**: Generate videos from text prompts
- **Image-to-Video**: Animate static images into videos
- **High Quality**: 720P @ 24fps output
- **Efficient Model**: TI2V-5B optimized for RTX 4090
- **Advanced VAE**: 16×16×4 compression ratio
- **Memory Optimized**: ~12-16GB VRAM usage
- **Generation Time**: 2-5 minutes per video

```bash
# Video generation examples:
# Text-to-Video: "A dragon flying over a castle"
# Image-to-Video: Upload image + "Make it come alive"
# Resolution: Up to 1280x720 @ 24fps
# Frames: 8-48 frames per video
```

### Development Workflow
1. **Access JupyterLab**: http://10.11.1.105:8888
2. **Create notebooks** for model experimentation
3. **Use Gradio** for quick UI prototyping
4. **Monitor GPU** via Streamlit dashboard
5. **Chat testing** through Open WebUI
6. **API access** via Ollama endpoint

## 📂 Project Structure

```
home-lab/
├── README.md                 # This documentation
├── .gitignore               # Git ignore rules
├── app.py                   # Flask admin panel server
├── templates/
│   ├── index.html          # Dashboard HTML template  
│   └── login.html          # Login page template
├── examples/
│   ├── README.md           # Services documentation
│   ├── launch_interfaces.sh # Start all services script
│   ├── stop_interfaces.sh  # Stop all services script
│   ├── gradio_chat_app.py  # Gradio chat interface
│   ├── streamlit_dashboard.py # GPU monitoring dashboard
│   ├── vllm_example.ipynb  # Sample Jupyter notebook
│   └── Untitled.ipynb     # Development notebook
└── nginx.conf              # Nginx configuration (optional)
```

## 🎯 Key Features & Fixes

### Admin Panel Features
- ✅ **Real-time Status Monitoring**: Live service health checks
- ✅ **Service Management**: Start/stop all services with one click
- ✅ **GPU Monitoring**: Temperature, utilization, memory usage
- ✅ **System Info**: Uptime, timestamps, service counts
- ✅ **Auto-refresh**: 30-second intervals with countdown
- ✅ **Responsive Design**: Dark theme, mobile-friendly

### Technical Improvements
- ✅ **Fixed Status Detection**: Handles both IPv4/IPv6 bindings
- ✅ **Color Coding**: Green (online) / Red (offline) indicators
- ✅ **JupyterLab Config**: Resolved theme errors, enabled external access  
- ✅ **Port Detection**: Reliable `ss -tlnp` parsing
- ✅ **External Access**: All services bind to 0.0.0.0

## 🔒 Security Configuration

### Current Setup (Development)
- **Admin Panel**: Password protected (admin / Dragon@123!@#)
- **JupyterLab**: No authentication (external access)
- **Other Services**: No authentication
- **Firewall**: Disabled (UFW inactive)
- **Network**: All services accessible externally

### Production Recommendations
```bash
# Enable firewall with specific ports
sudo ufw enable
sudo ufw allow 22    # SSH
sudo ufw allow 5000  # Admin Panel
sudo ufw allow 8888  # JupyterLab (restrict if needed)

# Change default passwords
# Edit app.py: ADMIN_PASSWORD = 'your-secure-password'
```

## 🐛 Troubleshooting

### Service Won't Start
```bash
# Check port usage
ss -tlnp | grep :PORT

# Kill existing processes
pkill -f jupyter
pkill -f gradio
pkill -f streamlit
pkill -f open-webui

# Check logs
tail -f app.log
```

### External Access Issues
```bash
# Test local access first
curl http://localhost:8888

# Check binding address
ss -tlnp | grep :8888
# Should show: 0.0.0.0:8888 (not 127.0.0.1:8888)

# Test external access
curl -I http://10.11.1.105:8888
```

### GPU Not Detected
```bash
# Verify NVIDIA drivers
nvidia-smi

# Check CUDA installation
nvcc --version

# Test PyTorch GPU access
python -c "import torch; print(torch.cuda.is_available())"
```

## 📊 Resource Requirements

### Minimum Requirements
- **CPU**: 8 cores, 3.0GHz+
- **RAM**: 16GB
- **GPU**: RTX 3080 (10GB VRAM)
- **Storage**: 100GB free space
- **Network**: 100Mbps

### Recommended (Current Setup)
- **CPU**: 16+ cores, 4.0GHz+
- **RAM**: 32GB+  
- **GPU**: RTX 4090 (24GB VRAM)
- **Storage**: 1TB NVMe SSD
- **Network**: Gigabit Ethernet

## 🚀 Advanced Usage

### Adding New Services
1. Edit `SERVICES_CONFIG` in `app.py`
2. Add service startup to `launch_interfaces.sh`
3. Add service cleanup to `stop_interfaces.sh`
4. Update this README

### Custom Models
```bash
# Add custom Ollama model
ollama create mymodel -f Modelfile
ollama run mymodel

# Add to Gradio interface
# Edit gradio_chat_app.py, update MODEL_NAME
```

### API Integration
```python
# Access Ollama API programmatically
import requests

response = requests.post('http://10.11.1.105:11434/api/generate', 
    json={
        'model': 'llama3.2:3b',
        'prompt': 'Hello, world!',
        'stream': False
    })
```

## 📝 Version History

### v1.0.0 (2025-10-02)
- ✅ Initial release with working admin panel
- ✅ Fixed service status detection (IPv4/IPv6 compatibility)
- ✅ JupyterLab external access configuration
- ✅ All services online and accessible
- ✅ Real-time monitoring with 30s auto-refresh
- ✅ Complete documentation and setup guide

---

**🎯 Status**: Production Ready ✅  
**👨‍💻 Author**: DevOps Team  
**📅 Last Updated**: October 2, 2025  
**🏠 Environment**: RTX 4090 Workstation (10.11.1.105)  
**🔗 Admin Panel**: http://10.11.1.105:5000 (admin / Dragon@123!@#)  
**📦 Repository**: https://github.com/webdevtodayjason/home-lab