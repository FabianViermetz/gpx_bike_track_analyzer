import webview
import subprocess
import time

subprocess.Popen([
    "streamlit", "run", "./gpx_power_calculator.py",
    "--server.headless", "true",
    "--server.address", "127.0.0.1",
    "--server.port", "8501",
])
time.sleep(2)  # wait for server

webview.create_window(
    title="GPX Analyzer",
    url="http://localhost:8501",
    width=1200,
    height=800
)
webview.start(gui="edgechromium")
