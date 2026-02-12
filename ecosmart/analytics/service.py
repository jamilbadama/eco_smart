from datetime import datetime

class AnalyticsService:
    """
    Handles real-time + predictive analytics, trend tracking, and dashboard outputs.
    """
    def __init__(self, session_store):
        self.session_store = session_store

    def get_session_trends(self, session_id):
        # In a real app, this would look up historical sessions for the same user
        # For prototype, we simulate a trend if we have multiple sessions stored
        sessions = self.session_store._db.get("sessions", {})
        # Filter sessions for the same participant (simplified logic)
        participant_id = session_id.split('_')[0]
        historical = [v for k, v in sessions.items() if k.startswith(participant_id)]
        
        if len(historical) < 2:
            return {"status": "baseline", "message": "Baseline session established."}
            
        # Compare last two
        sorted_h = sorted(historical, key=lambda x: x['timestamp'])
        prev = sorted_h[-2]["data"]["interpretation"]["confidence"] # Just a placeholder comparison
        curr = sorted_h[-1]["data"]["interpretation"]["confidence"]
        
        diff = curr - prev
        trend = "improving" if diff > 0.05 else "declining" if diff < -0.05 else "stable"
        
        return {
            "trend": trend,
            "change_magnitude": round(float(diff), 2),
            "historical_count": len(historical)
        }

    def get_dashboard_payload(self, session_id, analysis_result):
        trends = self.get_session_trends(session_id)
        return {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "analysis": analysis_result,
            "trends": trends
        }
