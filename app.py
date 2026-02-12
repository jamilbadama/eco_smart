import sys
import os

# Ensure the root directory is in sys.path
sys.path.append(os.getcwd())

if __name__ == "__main__":
    import uvicorn
    # Import the app from the package
    from ecosmart.app import app
    
    # Run uvicorn
    print("Starting Eco-SMART Application from consolidated package...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
