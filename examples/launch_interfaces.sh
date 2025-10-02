#!/bin/bash

# Launch script for all web interfaces
# This script starts JupyterLab, Gradio, Streamlit, and Open WebUI

set -e

echo "🚀 Starting All LLM Web Interfaces..."
echo "========================================"

# Function to check if port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to start service in background
start_service() {
    local name=$1
    local command=$2
    local port=$3
    local url=$4
    
    echo "Starting $name on port $port..."
    
    if check_port $port; then
        echo "⚠️  Port $port is already in use. $name might already be running."
        echo "   Access at: $url"
    else
        echo "   Command: $command"
        eval "$command" > /dev/null 2>&1 &
        local pid=$!
        echo "   PID: $pid"
        echo "   URL: $url"
        
        # Wait a moment and check if process is still running
        sleep 2
        if kill -0 $pid 2>/dev/null; then
            echo "   ✅ $name started successfully"
        else
            echo "   ❌ $name failed to start"
        fi
    fi
    echo ""
}

# Check if Ollama is running
echo "Checking Ollama service..."
if ! pgrep -f "ollama serve" > /dev/null; then
    echo "Starting Ollama service..."
    ollama serve > /dev/null 2>&1 &
    echo "✅ Ollama started"
    sleep 3  # Wait for Ollama to initialize
else
    echo "✅ Ollama is already running"
fi
echo ""

# Change to examples directory
cd /home/devops/examples

# 1. JupyterLab (Port 8888)
start_service "JupyterLab" \
    "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.token='' --ServerApp.password='' --ServerApp.disable_check_xsrf=True --ServerApp.allow_remote_access=True" \
    8888 \
    "http://10.11.1.105:8888"

# 2. Gradio Chat Interface (Port 7860)
start_service "Gradio Chat" \
    "python gradio_chat_app.py" \
    7860 \
    "http://localhost:7860"

# 3. Streamlit Dashboard (Port 8501)
start_service "Streamlit Dashboard" \
    "streamlit run streamlit_dashboard.py --server.port 8501 --server.address 0.0.0.0 --server.headless true" \
    8501 \
    "http://localhost:8501"

# 4. Open WebUI (Port 3000)
start_service "Open WebUI" \
    "open-webui serve --port 3000 --host 0.0.0.0" \
    3000 \
    "http://localhost:3000"

echo "========================================"
echo "🎉 All interfaces started!"
echo ""
echo "📱 Access your interfaces:"
echo "  • JupyterLab:         http://localhost:8888"
echo "  • Gradio Chat:        http://localhost:7860"  
echo "  • Streamlit Dashboard: http://localhost:8501"
echo "  • Open WebUI:         http://localhost:3000"
echo "  • Ollama API:         http://localhost:11434"
echo ""
echo "💡 Tips:"
echo "  • All interfaces support remote access if firewall allows"
echo "  • Use 'pkill -f jupyter' to stop JupyterLab"
echo "  • Use 'pkill -f gradio' to stop Gradio"
echo "  • Use 'pkill -f streamlit' to stop Streamlit"
echo "  • Use 'pkill -f open-webui' to stop Open WebUI"
echo "  • Use 'pkill -f ollama' to stop Ollama"
echo ""
echo "🔍 Monitor processes with:"
echo "  ps aux | grep -E 'jupyter|gradio|streamlit|open-webui|ollama'"
echo ""
echo "To stop all services, run: ./stop_interfaces.sh"