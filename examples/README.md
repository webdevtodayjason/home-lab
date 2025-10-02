# Local Colab-like Dashboard Suite

This directory contains a set of web apps that together provide a Google Colab-like experience on your RTX 4090 workstation.

Components:
- JupyterLab (interactive notebooks)
- Gradio (LLM chat interface via Ollama)
- Streamlit (GPU monitoring + model manager)
- Open WebUI (full chat UI for Ollama)

## Quick start

1) Start all interfaces

```
./launch_interfaces.sh
```

Then open these URLs in your browser:
- JupyterLab: http://localhost:8888
- Gradio Chat: http://localhost:7860
- Streamlit Dashboard: http://localhost:8501
- Open WebUI: http://localhost:3000

2) Stop all interfaces

```
./stop_interfaces.sh
```

## Individual apps

- Jupyter notebook example: `examples/vllm_example.ipynb`
- Gradio chat app: `examples/gradio_chat_app.py`
- Streamlit dashboard: `examples/streamlit_dashboard.py`

Run individually if preferred:

```
# JupyterLab
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token='' --NotebookApp.password=''

# Gradio chat
python examples/gradio_chat_app.py

# Streamlit dashboard
streamlit run examples/streamlit_dashboard.py --server.port 8501 --server.address 0.0.0.0 --server.headless true

# Open WebUI
open-webui serve --port 3000 --host 0.0.0.0
```

## Notes

- The Gradio app uses Ollama models exposed via the local API. Ensure `ollama serve` is running.
- The Streamlit dashboard reads from `nvidia-smi`; ensure NVIDIA drivers are installed.
- JupyterLab is launched without a token for convenience on localhost. If exposing remotely, set a token or reverse proxy with auth.
- If ports are busy, edit the launch script to change ports.

## Remote access (optional)

If you want to access the dashboards from another machine, open the following ports in your firewall/router:
- 8888 (JupyterLab)
- 7860 (Gradio)
- 8501 (Streamlit)
- 3000 (Open WebUI)

For security, consider using an SSH tunnel instead of opening ports publically:

```
ssh -N -L 8888:localhost:8888 -L 7860:localhost:7860 -L 8501:localhost:8501 -L 3000:localhost:3000 user@your-server
```

## Troubleshooting

- If Jupyter fails to start, check logs: `ps aux | grep jupyter` and kill stale processes.
- If models don’t appear in Gradio, run `ollama list` and pull a model (e.g., `ollama pull llama3.2:3b`).
- If GPU metrics don’t show, verify `nvidia-smi` works.
- If Streamlit says `pyarrow` incompatible with datasets, upgrade pyarrow or pin datasets accordingly.
