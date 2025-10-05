#!/usr/bin/env python3
"""
Combined application runner that starts both FastAPI and Streamlit
"""
import subprocess
import threading
import time
import os
import sys

def run_fastapi():
    """Run the FastAPI server in a separate thread"""
    try:
        # Add the current directory to Python path
        sys.path.insert(0, os.getcwd())
        
        # Import and run the FastAPI app
        from api.deploy_api import app
        import uvicorn
        
        # Run on a different port (8001) so Streamlit can use the main port
        uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
    except Exception as e:
        print(f"Error starting FastAPI: {e}")

def run_streamlit():
    """Run the Streamlit app"""
    try:
        # Set the API_BASE environment variable to point to localhost
        os.environ["API_BASE"] = "http://localhost:8001"
        
        # Run Streamlit
        subprocess.run([
            "streamlit", "run", "src/demo/app.py", 
            "--server.port", os.environ.get("PORT", "10000"),
            "--server.address", "0.0.0.0"
        ])
    except Exception as e:
        print(f"Error starting Streamlit: {e}")

if __name__ == "__main__":
    # Start FastAPI in a separate thread
    api_thread = threading.Thread(target=run_fastapi, daemon=True)
    api_thread.start()
    
    # Give FastAPI a moment to start
    time.sleep(3)
    
    # Start Streamlit (this will block)
    run_streamlit()
