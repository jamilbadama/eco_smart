import asyncio
import websockets
import json
import base64
import numpy as np
import cv2

async def test_session_save():
    uri = "ws://localhost:8000/ws/monitor"
    async with websockets.connect(uri) as websocket:
        print("Connected to WebSocket")
        
        # 1. Send some frames
        for i in range(10):
            # Dummy frame
            img = np.zeros((180, 320, 3), dtype=np.uint8)
            cv2.putText(img, f"Frame {i}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            _, buffer = cv2.imencode('.jpg', img)
            base64_frame = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
            
            # Dummy audio
            audio_chunk = [0.1] * 441
            
            await websocket.send(json.dumps({"video": base64_frame, "audio": audio_chunk}))
            response = await websocket.recv()
            print(f"Received frame {i} response")

        # 2. Send save command
        print("Sending save command...")
        await websocket.send(json.dumps({
            "command": "save",
            "session_id": "TEST_LIVE_SESSION"
        }))
        
        # 3. Wait for save confirmation
        response = await websocket.recv()
        data = json.loads(response)
        print("Save Result:", data)
        
        if data.get("status") == "saved":
            print("SUCCESS: Session saved correctly!")
        else:
            print("ERROR: Session save failed.")

if __name__ == "__main__":
    asyncio.run(test_session_save())
