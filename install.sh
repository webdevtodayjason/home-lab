#!/bin/bash

# Installation script for LLM Admin Panel
set -e

echo "🚀 Setting up LLM Admin Panel..."
echo "=================================="

# Check if running as root or with sudo
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run with sudo privileges"
   echo "Usage: sudo ./install.sh"
   exit 1
fi

# Get the actual user (not root when using sudo)
ACTUAL_USER=${SUDO_USER:-$USER}
USER_HOME=$(getent passwd "$ACTUAL_USER" | cut -d: -f6)

echo "Installing for user: $ACTUAL_USER"
echo "User home: $USER_HOME"

# Stop any existing Docker containers on port 80
echo "Checking for Docker containers on port 80..."
DOCKER_CONTAINERS=$(docker ps --filter "publish=80" -q 2>/dev/null || true)
if [[ ! -z "$DOCKER_CONTAINERS" ]]; then
    echo "Stopping Docker containers using port 80..."
    docker stop $DOCKER_CONTAINERS || true
fi

# Configure nginx
echo "Configuring nginx..."

# Remove default site
rm -f /etc/nginx/sites-enabled/default

# Copy our config
cp /home/devops/admin-panel/nginx.conf /etc/nginx/sites-available/llm-admin-panel
ln -sf /etc/nginx/sites-available/llm-admin-panel /etc/nginx/sites-enabled/

# Test nginx config
if nginx -t; then
    echo "✅ Nginx configuration is valid"
else
    echo "❌ Nginx configuration is invalid"
    exit 1
fi

# Install systemd services
echo "Installing systemd services..."

# Copy service files
cp /home/devops/admin-panel/llm-admin-panel.service /etc/systemd/system/
cp /home/devops/admin-panel/llm-services.service /etc/systemd/system/

# Set proper permissions
chown root:root /etc/systemd/system/llm-admin-panel.service
chown root:root /etc/systemd/system/llm-services.service
chmod 644 /etc/systemd/system/llm-admin-panel.service
chmod 644 /etc/systemd/system/llm-services.service

# Reload systemd
systemctl daemon-reload

# Enable services for auto-start
echo "Enabling services for auto-start..."
systemctl enable nginx
systemctl enable llm-admin-panel.service
systemctl enable llm-services.service

# Start nginx
echo "Starting nginx..."
systemctl restart nginx

# Start admin panel
echo "Starting admin panel..."
systemctl start llm-admin-panel.service

# Wait a moment for the admin panel to start
sleep 3

# Check service statuses
echo ""
echo "📊 Service Status:"
echo "=================="

services=("nginx" "llm-admin-panel")
for service in "${services[@]}"; do
    if systemctl is-active --quiet "$service"; then
        echo "✅ $service: Active"
    else
        echo "❌ $service: Inactive"
    fi
done

# Test the web interface
echo ""
echo "🔍 Testing web interface..."
if curl -s -o /dev/null -w "%{http_code}" http://localhost/ | grep -q "200\|302"; then
    echo "✅ Web interface is accessible"
else
    echo "⚠️  Web interface may not be fully ready yet"
fi

# Create a status check script
cat > /home/devops/admin-panel/check-status.sh << 'EOF'
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
EOF

chmod +x /home/devops/admin-panel/check-status.sh
chown $ACTUAL_USER:$ACTUAL_USER /home/devops/admin-panel/check-status.sh

# Final instructions
echo ""
echo "🎉 Installation Complete!"
echo "========================"
echo ""
echo "📱 Access your admin panel:"
echo "  URL: http://localhost/"
echo "  Username: admin"
echo "  Password: Dragon@123!@#"
echo ""
echo "📋 Management Commands:"
echo "  Status:  ./check-status.sh"
echo "  Restart: sudo systemctl restart llm-admin-panel nginx"
echo "  Logs:    sudo journalctl -f -u llm-admin-panel"
echo ""
echo "🔄 The services will auto-start on boot!"
echo ""
echo "⚠️  Note: LLM services may take a few minutes to fully start."
echo "   Use the admin panel to monitor and control them."