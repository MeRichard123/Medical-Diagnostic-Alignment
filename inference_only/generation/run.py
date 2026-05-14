import subprocess
import time
import psutil
from datetime import datetime

def is_python_process_running():
    """Check if any Python process is currently running (excluding current process)."""
    current_pid = subprocess.os.getpid()
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if proc.info['name'] == 'python.exe' or proc.info['name'] == 'python':
                if proc.info['pid'] != current_pid:
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return False

def main():
    while True:
        print(f"[{datetime.now()}] Checking for running Python processes...")
        
        if is_python_process_running():
            print("Python process found. Skipping execution.")
        else:
            subprocess.run(["python", "-m", "inference_only.generation.vlm_inference"])
        
        # Wait 10 minutes before next check
        time.sleep(600)

if __name__ == "__main__":
    main()