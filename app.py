#!/usr/bin/env python3

"""
Beautiful Admin Panel Homepage - Port 80
Password protected dark mode admin panel with service cards
"""

from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import subprocess
import requests
from datetime import datetime
import os
import hashlib

app = Flask(__name__)
app.secret_key = 'dragon_admin_panel_secret_key_2025'

# Admin credentials
ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = 'Dragon@123!@#'

# Base services configuration (without URLs - will be dynamically generated)
SERVICES_CONFIG = [
    {
        'id': 'jupyter',
        'name': 'JupyterLab',
        'description': 'Interactive Jupyter notebooks with GPU support',
        'icon': '📊',
        'port': 8888,
        'color': 'from-orange-500 to-red-500'
    },
    {
        'id': 'gradio',
        'name': 'Gradio Chat',
        'description': 'Interactive chat interface with local LLM models',
        'icon': '💬',
        'port': 7860,
        'color': 'from-blue-500 to-cyan-500'
    },
    {
        'id': 'streamlit',
        'name': 'Streamlit Dashboard',
        'description': 'GPU monitoring and model management dashboard',
        'icon': '📈',
        'port': 8501,
        'color': 'from-green-500 to-emerald-500'
    },
    {
        'id': 'openwebui',
        'name': 'Open WebUI',
        'description': 'Full-featured ChatGPT-like interface for Ollama',
        'icon': '🤖',
        'port': 3000,
        'color': 'from-purple-500 to-pink-500'
    },
    {
        'id': 'ollama',
        'name': 'Ollama API',
        'description': 'Local LLM API service for model inference',
        'icon': '🚀',
        'port': 11434,
        'color': 'from-yellow-500 to-orange-500'
    },
    {
        'id': 'wanvideo',
        'name': 'Wan2.2 Video Gen',
        'description': 'AI video generation with Wan2.2 TI2V-5B model',
        'icon': '🎬',
        'port': 7870,
        'color': 'from-purple-500 to-indigo-500'
    }
]

def get_services_with_dynamic_urls(host):
    """Generate services list with dynamic URLs based on the request host."""
    services = []
    for service_config in SERVICES_CONFIG:
        service = service_config.copy()
        service['url'] = f"http://{host}:{service['port']}"
        service['status'] = 'unknown'
        services.append(service)
    return services

def check_service_status(port):
    """Check if service is running on given port."""
    try:
        # First try with netstat/ss command which is more reliable
        result = subprocess.run(['ss', '-tlnp'], capture_output=True, text=True)
        if result.returncode == 0:
            # Check for both IPv4 and IPv6 formats
            # IPv4: 0.0.0.0:port or 127.0.0.1:port
            # IPv6: *:port or :::port or ::1:port
            lines = result.stdout.split('\n')
            for line in lines:
                if f':{port}' in line and 'LISTEN' in line:
                    return 'online'
        
        # Fallback to lsof if ss is not available
        result = subprocess.run(['lsof', '-Pi', f':{port}', '-sTCP:LISTEN', '-t'], 
                              capture_output=True, text=True)
        return 'online' if result.returncode == 0 and result.stdout.strip() else 'offline'
    except:
        return 'offline'

def get_system_info():
    """Get system information."""
    try:
        # GPU info
        gpu_result = subprocess.run(['nvidia-smi', '--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total',
                                   '--format=csv,noheader,nounits'], capture_output=True, text=True)
        gpu_info = {}
        if gpu_result.returncode == 0 and gpu_result.stdout.strip():
            data = gpu_result.stdout.strip().split(', ')
            gpu_info = {
                'name': data[0],
                'temperature': int(data[1]),
                'utilization': int(data[2]),
                'memory_used': int(data[3]),
                'memory_total': int(data[4])
            }
        
        # System uptime
        uptime_result = subprocess.run(['uptime', '-p'], capture_output=True, text=True)
        uptime = uptime_result.stdout.strip().replace('up ', '') if uptime_result.returncode == 0 else 'Unknown'
        
        return {
            'gpu': gpu_info,
            'uptime': uptime,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    except:
        return {
            'gpu': {},
            'uptime': 'Unknown',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

@app.route('/')
def index():
    """Main admin panel page."""
    if 'authenticated' not in session:
        return redirect(url_for('login'))
    
    # Get the host from the request (handles both hostname and IP)
    # Extract just the hostname/IP without the port
    host = request.host.split(':')[0]
    
    # Generate services with dynamic URLs based on the current host
    services = get_services_with_dynamic_urls(host)
    
    # Update service statuses
    for service in services:
        service['status'] = check_service_status(service['port'])
    
    system_info = get_system_info()
    
    return render_template('index.html', 
                         services=services, 
                         system_info=system_info,
                         current_host=host)

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page."""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['authenticated'] = True
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid credentials')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Logout and clear session."""
    session.pop('authenticated', None)
    return redirect(url_for('login'))

@app.route('/api/services/status')
def api_services_status():
    """API endpoint to get service statuses."""
    if 'authenticated' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    # Get the host from the request (extract hostname/IP without port)
    host = request.host.split(':')[0]
    services = get_services_with_dynamic_urls(host)
    
    statuses = {}
    for service in services:
        statuses[service['id']] = check_service_status(service['port'])
    
    return jsonify(statuses)

@app.route('/api/system/info')
def api_system_info():
    """API endpoint to get system information."""
    if 'authenticated' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    return jsonify(get_system_info())

@app.route('/api/services/start')
def start_services():
    """Start all services."""
    if 'authenticated' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        # Run the launch script
        result = subprocess.run(['/home/devops/examples/launch_interfaces.sh'], 
                              capture_output=True, text=True, cwd='/home/devops/examples')
        return jsonify({
            'success': True,
            'message': 'Services started successfully',
            'output': result.stdout
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Failed to start services: {str(e)}'
        }), 500

@app.route('/api/services/stop')
def stop_services():
    """Stop all services."""
    if 'authenticated' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        # Run the stop script
        result = subprocess.run(['/home/devops/examples/stop_interfaces.sh'], 
                              capture_output=True, text=True, cwd='/home/devops/examples')
        return jsonify({
            'success': True,
            'message': 'Services stopped successfully',
            'output': result.stdout
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Failed to stop services: {str(e)}'
        }), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('/home/devops/admin-panel/templates', exist_ok=True)
    
    # Bind to all network interfaces to allow access via hostname or IP
    app.run(host='0.0.0.0', port=5000, debug=False)
