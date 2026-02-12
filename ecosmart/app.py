from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import sys
import json

# Ensure root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .ai_infra.orchestrator import get_supervisor
from .ai_infra.tools.detection import analyze_session
from .services.realtime_service import RealtimeMonitorService
from .data_layer.store import SessionStore

app = FastAPI(title="Eco-SMART API")

# Initialize agent once
print("DEBUG: Initializing supervisor...")
try:
    supervisor = get_supervisor()
    print("DEBUG: Supervisor initialized.")
except Exception as e:
    print(f"DEBUG: Supervisor initialization failed: {e}")
    supervisor = None

print("DEBUG: Initializing session store...")
store = SessionStore()
print("DEBUG: Session store initialized.")

class QueryRequest(BaseModel):
    query: str

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))

@app.websocket("/ws/monitor")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    service = RealtimeMonitorService()
    print("DEBUG: WebSocket connection accepted for real-time monitoring.")
    
    try:
        while True:
            data = await websocket.receive_json()
            
            response_payload = {}
            
            if "video" in data:
                v_markers = await service.process_video_frame(data["video"])
                if v_markers:
                    response_payload["video_markers"] = v_markers
            
            if "audio" in data:
                a_markers = await service.process_audio_chunk(data["audio"])
                if a_markers:
                    response_payload["audio_markers"] = a_markers
            
            # Handle Save Command
            if data.get("command") == "save":
                session_id = data.get("session_id", "LIVE_SESSION")
                summary = service.get_session_summary(session_id)
                if summary:
                    store.save_session_result(session_id, summary)
                    await websocket.send_json({
                        "status": "saved", 
                        "session_id": session_id,
                        "summary": summary
                    })
                    print(f"DEBUG: Session {session_id} saved via WebSocket.")
                else:
                    await websocket.send_json({"status": "error", "message": "No data to save"})

            # Send periodic analysis (e.g., every 10 frames/chunks)
            # For simplicity, send analysis if we have any markers
            if response_payload:
                analysis = service.get_rolling_analysis()
                response_payload["analysis"] = analysis
                await websocket.send_json(response_payload)
                
    except WebSocketDisconnect:
        print("DEBUG: WebSocket disconnected.")
    except Exception as e:
        print(f"DEBUG: WebSocket error: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass

@app.post("/api/session/save")
async def save_session(data: dict):
    """
    Finalizes and saves a real-time monitor session.
    """
    try:
        session_id = data.get("session_id", "LIVE_SESSION")
        summary = data.get("summary")
        
        if not summary:
            return {"status": "error", "message": "No session summary provided"}
            
        store.save_session_result(session_id, summary)
        print(f"DEBUG: Session {session_id} saved successfully.")
        return {"status": "success", "session_id": session_id}
    except Exception as e:
        print(f"DEBUG: Failed to save session: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/api/session/update")
async def update_session(data: dict):
    """
    Updates an existing session with ground truth clinical scores.
    """
    try:
        session_id = data.get("session_id")
        phq8_score = data.get("phq8_score")
        
        if session_id is None or phq8_score is None:
            return {"status": "error", "message": "Missing session_id or phq8_score"}
            
        existing = store.get_session_result(session_id)
        if not existing:
            return {"status": "error", "message": "Session not found"}
            
        # Update ground truth in result payload
        # Ensure deep nesting exists
        if "data" in existing and "analysis" in existing["data"] and "evidence" in existing["data"]["analysis"]:
            existing["data"]["analysis"]["evidence"]["clinical_grounding"] = {"phq8": phq8_score}
            
        store.save_session_result(session_id, existing["data"])
        print(f"DEBUG: Session {session_id} updated with PHQ-8 score: {phq8_score}")
        return {"status": "success", "session_id": session_id}
    except Exception as e:
        print(f"DEBUG: Failed to update session: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/api/analyze")
async def analyze(request: QueryRequest):
    try:
        # Invoke agent
        response = supervisor.invoke(
            {"messages": [{"role": "user", "content": request.query}]}
        )
        
        # Get final response
        final_message = response['messages'][-1].content
        
        # Look for clinical data in the final message using dashboard_data tags
        clinical_data = None
        if "<dashboard_data>" in final_message:
            try:
                raw_data = final_message.split("<dashboard_data>")[1].split("</dashboard_data>")[0].strip()
                clinical_data = json.loads(raw_data)
                print("DEBUG: Found clinical data in <dashboard_data> tags.")
                # Strip tags from the final report message
                final_message = final_message.split("<dashboard_data>")[0].strip()
            except Exception as e:
                print(f"DEBUG: Failed to parse <dashboard_data> content: {e}")
        
        # Comprehensive fallback search in message history
        if clinical_data is None:
            print("DEBUG: Falling back to history search...")
            for msg in reversed(response['messages']):
                msg_content = str(msg.content)
                if "<dashboard_data>" in msg_content:
                    try:
                        raw_data = msg_content.split("<dashboard_data>")[1].split("</dashboard_data>")[0].strip()
                        clinical_data = json.loads(raw_data)
                        print("DEBUG: Found clinical data in message history tags.")
                        break
                    except: pass
                
                # Check for raw JSON markers
                if clinical_data is None and '"session_id"' in msg_content:
                    try:
                        clean = msg_content.strip()
                        if "```json" in clean:
                            clean = clean.split("```json")[1].split("```")[0].strip()
                        data = json.loads(clean)
                        if "clinical_prediction" in data:
                            clinical_data = data; break
                    except: pass

        if clinical_data is None:
            print("DEBUG: No clinical data found in any message.")

        return {
            "status": "success",
            "report": final_message,
            "data": clinical_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files
static_dir = os.path.join(BASE_DIR, "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
