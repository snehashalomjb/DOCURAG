import subprocess
import sys
import os

def main():
    print("==================================================")
    print("          Starting DOCURAG Orchestrator           ")
    print("==================================================")
    
    python_bin = sys.executable
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "frontend", "app.py")
    
    print(f"Launching Streamlit application: {script_path}")
    print("The backend FastAPI service will auto-start in the background.")
    print("Open your browser and navigate to the Streamlit URL below.")
    print("Press Ctrl+C to terminate both servers.")
    print("==================================================")
    
    try:
        # Launch Streamlit process (runs synchronously)
        subprocess.run([python_bin, "-m", "streamlit", "run", script_path])
    except KeyboardInterrupt:
        print("\nStopping DOCURAG processes...")
    except Exception as e:
        print(f"\nError launching DOCURAG: {e}")

if __name__ == "__main__":
    main()
