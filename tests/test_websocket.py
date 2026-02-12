import asyncio
import websockets
import json
import base64
import numpy as np
import cv2

async def test_monitor():
    uri = "ws://localhost:8000/ws/monitor"
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to WebSocket")
            
            # 1 & 2. Simulate Video and Audio in a loop to trigger emotion detection (every 5 frames)
            print("Sending multiple frames to trigger periodic emotion detection...")
            for i in range(6):
                # Create a dummy image
                img = np.zeros((180, 320, 3), dtype=np.uint8)
                cv2.putText(img, f"Frame {i}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                _, buffer = cv2.imencode('.jpg', img)
                base64_frame = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
                
                # Sine wave segment
                t = np.linspace(0, 0.1, 441)
                audio_chunk = (np.sin(2 * np.pi * 440 * t) * 0.5).tolist()
                
                await websocket.send(json.dumps({"video": base64_frame, "audio": audio_chunk}))
                
                # Receive Response
                response = await websocket.recv()
                data = json.loads(response)
                
                if "analysis" in data and "emotion" in data["analysis"]:
                    print(f"Frame {i}: Received analysis with emotion: {data['analysis']['emotion']['label']}")
                else:
                    print(f"Frame {i}: Received analysis (no emotion update yet)")

            if "analysis" in data:
                print("\nFINAL STATE:")
                print(json.dumps(data, indent=2))
                print("\nSUCCESS: Multi-frame analysis received correctly!")
            else:
                print("\nFAILURE: No analysis in response.")
                
    except Exception as e:
        print(f"Connection failed: {e}")
        print("Make sure the app is running (python app.py)")

if __name__ == "__main__":
    asyncio.run(test_monitor())
