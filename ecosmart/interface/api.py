import json
import torch
from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional

from ..data_layer.ingest import DataIngestionService, DAICWOZAdapter
from ..data_layer.store import SessionStore
from ..ai_infra.marker_detection import MarkerDetector
from ..ai_infra.triangulation import TriangulationEngine
from ..ai_infra.interpretation import InterpretationEngine
from ..analytics.service import AnalyticsService
from ..services.auth import AuthService
from ..services.audit import AuditService

app = FastAPI(title="Eco-SMART Platform API")

# Initialize services
auth_service = AuthService()
audit_service = AuditService()
session_store = SessionStore()
data_service = DataIngestionService(DAICWOZAdapter())
marker_detector = MarkerDetector()
triangulation_engine = TriangulationEngine()
interpretation_engine = InterpretationEngine()
analytics_service = AnalyticsService(session_store)

class Feedback(BaseModel):
    session_id: str
    clinician_label: int
    rationale: Optional[str] = None

async def verify_auth(x_api_key: Optional[str] = Header(None)):
    if not auth_service.verify_key(x_api_key):
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return x_api_key

@app.get("/health")
def health():
    return {"status": "online", "version": "MDPI-iSMART-V3"}

@app.post("/sessions/{session_id}/analyze", dependencies=[Depends(verify_auth)])
async def analyze_session_endpoint(session_id: str):
    AuditService.log_access("clinician_1", session_id, "analyze")
    
    try:
        # 1. Data Layer
        data = data_service.get_session_data(session_id)
        
        # 2. AI Infrastructure - Marker Detection
        # (In a real implementation, we'd tokens/embeddings here)
        markers = {
            "text": marker_detector.extract_text_markers(data.get('raw_text', "")),
            "audio": marker_detector.extract_audio_markers(data.get('audio', torch.zeros(1))),
            "video": marker_detector.extract_video_markers(data.get('video', torch.zeros(1)))
        }
        
        # 3. AI Infrastructure - Model Fusion (Simulated / Placeholder for Model Run)
        # Note: In a full implementation, we'd load the model and call forward()
        # For the API prototype, we simulate outputs consistent with existing pipeline
        model_outputs = {
            "binary_prob": 0.72, # Placeholder
            "score": 12.0 # Placeholder
        }
        
        # 4. Triangulation
        evidence = triangulation_engine.triangulate(model_outputs, markers, {"phq8": 11}) # Simulated PHQ8
        
        # 5. Interpretation
        interpretation = interpretation_engine.interpret(evidence)
        
        # 6. Analytics
        analysis_result = {
            "markers": markers,
            "evidence": evidence,
            "interpretation": interpretation
        }
        payload = analytics_service.get_dashboard_payload(session_id, analysis_result)
        
        # Save to store
        session_store.save_session_result(session_id, payload)
        
        # Log decision
        AuditService.log_clinical_decision(session_id, interpretation["risk_level"], interpretation["confidence"])
        
        # Format as the expected <dashboard_data> block for the LLM
        return {
            "report": interpretation["summary"],
            "dashboard_data": f"<dashboard_data>{json.dumps(payload)}</dashboard_data>"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback", dependencies=[Depends(verify_auth)])
async def submit_feedback(fb: Feedback):
    AuditService.log_access("clinician_1", fb.session_id, "feedback")
    # For now, just log it. In a real system, this goes to a DB for retraining.
    print(f"FEEDBACK RECEIVED: {fb}")
    return {"status": "recorded"}
