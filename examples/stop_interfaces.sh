#!/bin/bash

# Stop script for all web interfaces
# This script stops JupyterLab, Gradio, Streamlit, and Open WebUI

echo "🛑 Stopping All LLM Web Interfaces..."
echo "========================================"

# Function to stop services by name
stop_service() {
    local service_name=$1
    local process_pattern=$2
    
    echo "Stopping $service_name..."
    
    # Find and kill processes
    local pids=$(pgrep -f "$process_pattern" 2>/dev/null || true)
    
    if [ -n "$pids" ]; then
        echo "  Found PIDs: $pids"
        echo "$pids" | xargs kill -TERM 2>/dev/null || true
        
        # Wait a moment for graceful shutdown
        sleep 2
        
        # Force kill if still running
        local remaining_pids=$(pgrep -f "$process_pattern" 2>/dev/null || true)
        if [ -n "$remaining_pids" ]; then
            echo "  Force killing remaining processes: $remaining_pids"
            echo "$remaining_pids" | xargs kill -9 2>/dev/null || true
        fi
        
        echo "  ✅ $service_name stopped"
    else
        echo "  ℹ️  $service_name was not running"
    fi
    echo ""
}

# Stop all services
stop_service "JupyterLab" "jupyter-lab"
stop_service "Gradio Chat" "gradio_chat_app.py"
stop_service "Streamlit Dashboard" "streamlit.*streamlit_dashboard.py"
stop_service "Open WebUI" "open-webui"
stop_service "Ollama" "ollama serve"

echo "========================================"
echo "✅ All interfaces stopped!"
echo ""
echo "🔍 Verify all processes are stopped:"
echo "  ps aux | grep -E 'jupyter|gradio|streamlit|open-webui|ollama'"
echo ""
echo "To start all services again, run: ./launch_interfaces.sh"