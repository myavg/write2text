import os
import sys
import webbrowser
from pathlib import Path
import uvicorn

# Add backend directory to path
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from backend.main import app

if __name__ == "__main__":
    # Get frontend path
    frontend_path = BASE_DIR / "frontend" / "index.html"
    
    print("=" * 60)
    print("Starting Text Segmenter Service...")
    print("=" * 60)
    print(f"Backend API: http://localhost:8000")
    print(f"Frontend: {frontend_path}")
    print("=" * 60)
    print("\nPress Ctrl+C to stop the server\n")
    
    # Open frontend in browser after a short delay
    def open_browser():
        import time
        time.sleep(1.5)  # Wait for server to start
        if frontend_path.exists():
            webbrowser.open(f"file://{frontend_path.absolute()}")
        else:
            print(f"Warning: Frontend file not found at {frontend_path}")
    
    # Start browser in a separate thread
    import threading
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # Start the server
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    except KeyboardInterrupt:
        print("\n\nServer stopped.")
