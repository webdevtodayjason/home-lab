#!/bin/bash
echo "🔍 LLM Admin Panel Status"
echo "========================"
echo ""

# Check system services
services=("nginx" "llm-admin-panel" "llm-services")
for service in "${services[@]}"; do
    if systemctl is-active --quiet "$service"; then
        echo "✅ $service: $(systemctl is-active $service)"
    else
        echo "❌ $service: $(systemctl is-active $service)"
    fi
done

echo ""
echo "🌐 Web Interface:"
echo "  Admin Panel: http://localhost/"
echo "  Credentials: admin / Dragon@123!@#"
echo ""

# Check individual service ports
echo "🔌 Service Ports:"
ports=(5000 8888 7860 8501 3000 11434)
port_names=("Admin Panel" "JupyterLab" "Gradio Chat" "Streamlit" "Open WebUI" "Ollama API")

for i in "${!ports[@]}"; do
    port=${ports[$i]}
    name=${port_names[$i]}
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "✅ $name (Port $port): Online"
    else
        echo "❌ $name (Port $port): Offline"
    fi
done
