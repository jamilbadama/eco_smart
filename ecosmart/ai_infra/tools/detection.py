import torch
import numpy as np
import os
import sys
import json

# Ensure root is in path
sys.path.append(os.getcwd())

from ecosmart.data_layer.ingest import DataIngestionService, DAICWOZAdapter
from ecosmart.data_layer.store import SessionStore
from ecosmart.ai_infra.marker_detection import MarkerDetector
from ecosmart.ai_infra.triangulation import TriangulationEngine
from ecosmart.ai_infra.interpretation import InterpretationEngine
from ecosmart.analytics.service import AnalyticsService

# Initialize Ecosystem components
_SESSION_STORE = SessionStore()
_DATA_SERVICE = DataIngestionService(DAICWOZAdapter())
_MARKER_DETECTOR = MarkerDetector()
_TRIANGULATION_ENGINE = TriangulationEngine()
_INTERPRETATION_ENGINE = InterpretationEngine()
_ANALYTICS_SERVICE = AnalyticsService(_SESSION_STORE)

def analyze_session(session_id: str):
    """
    Analyzes a clinical session using the Eco-SMART layered architecture.
    """
    # Normalize ID
    if not session_id.endswith('_P') and '_' not in session_id:
        session_id = f"{session_id}_P"
        
    try:
        # 1. Data Layer: Ingestion
        data = _DATA_SERVICE.get_session_data(session_id)
        
        # 2. AI Infrastructure: Marker Detection
        markers = {
            "text": _MARKER_DETECTOR.extract_text_markers(data.get('raw_text', "")),
            "audio": _MARKER_DETECTOR.extract_audio_markers(data.get('audio', torch.zeros(1))),
            "video": _MARKER_DETECTOR.extract_video_markers(data.get('video', torch.zeros(1)))
        }

        # 2b. High-Level Hub Fusion (MMFformer-style)
        fusion_result = _MARKER_DETECTOR.process_multimodal_fusion(markers)
        
        # 3. AI Infrastructure: Triangulation
        # (Simulating model outputs for this tool wrapper to avoid dependency bloat in CLI)
        # In a full run, we'd use the FusionModel weights.
        model_outputs = {
            "binary_prob": 0.72,
            "score": 12.5
        }
        
        # Simulated clinical context (PHQ-8 ground truth if available, else placeholder)
        self_reports = {"phq8": 11} 
        
        evidence = _TRIANGULATION_ENGINE.triangulate(model_outputs, markers, self_reports, fusion_result)
        
        # 4. AI Infrastructure: Interpretation
        interpretation = _INTERPRETATION_ENGINE.interpret(evidence)
        
        # 5. Analytics Layer
        analysis_result = {
            "markers": markers,
            "evidence": evidence,
            "interpretation": interpretation
        }
        payload = _ANALYTICS_SERVICE.get_dashboard_payload(session_id, analysis_result)
        
        # 6. Persistence
        _SESSION_STORE.save_session_result(session_id, payload)
        
        return f"<dashboard_data>{json.dumps(payload)}</dashboard_data>"
        
    except Exception as e:
        return {"error": f"Eco-SMART Analysis failed: {str(e)}"}

if __name__ == "__main__":
    # Test
    print(analyze_session("302"))
