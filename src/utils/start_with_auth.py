import os
import sys
from pathlib import Path

def start_mlflow_server():
    Path("mlruns").mkdir(exist_ok=True)
    Path("mlartifacts").mkdir(exist_ok=True)
    
    backend_store_uri = os.getenv("MLFLOW_BACKEND_STORE_URI", "sqlite:///mlflow.db")
    artifact_root = os.getenv("MLFLOW_ARTIFACT_ROOT", "./mlruns")
    host = os.getenv("MLFLOW_HOST", "0.0.0.0")
    port = os.getenv("MLFLOW_PORT", "5000")
    
    cmd = f"mlflow server \
        --backend-store-uri {backend_store_uri} \
        --default-artifact-root {artifact_root} \
        --host {host} \
        --port {port}"
    os.system(cmd)

if __name__ == "__main__":
    start_mlflow_server()