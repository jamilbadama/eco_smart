class InterpretationEngine:
    """
    Maps outputs to risk bands, attaches confidence, and generates explanations.
    """
    def __init__(self, config=None):
        self.config = config

    def interpret(self, triangulation_result):
        score = triangulation_result["primary_signals"].get("score", 0)
        prob = triangulation_result["primary_signals"].get("binary_prob", 0)
        
        # Risk Band Mapping
        if score >= 15:
            risk_band = "Moderately Severe / Severe"
            clinical_priority = "High"
        elif score >= 10:
            risk_band = "Moderate"
            clinical_priority = "Medium"
        elif score >= 5:
            risk_band = "Mild"
            clinical_priority = "Low"
        else:
            risk_band = "Minimal/No Depression"
            clinical_priority = "Routine"
            
        # Uncertainty Estimate (Heuristic for MVP)
        # Higher confidence if model probability is extreme (0 or 1)
        confidence = 1.0 - abs(0.5 - prob) * 2  # This is actually inverse, fixed below
        confidence = 0.5 + abs(0.5 - prob) # Range 0.5 to 1.0
        
        uncertainty_label = "Low" if confidence > 0.8 else "Moderate" if confidence > 0.6 else "High"
        
        # Explanation generation
        markers = triangulation_result["behavioral_markers"]
        reasons = []
        if markers.get("text", {}).get("sentiment", 0.5) < 0.4:
            reasons.append("Negative linguistic sentiment detected")
        if markers.get("video", {}).get("movement_mean", 0.1) < 0.05:
            reasons.append("Signs of psychomotor retardation (reduced movement)")
        if markers.get("audio", {}).get("pitch_std", 1.0) < 0.5:
            reasons.append("Monotone speech patterns detected")
            
        return {
            "risk_level": risk_band,
            "clinical_priority": clinical_priority,
            "confidence": round(float(confidence), 2),
            "uncertainty_label": uncertainty_label,
            "explanations": reasons or ["No strong behavioral markers detected."],
            "summary": f"Assessment suggests {risk_band} risk with {uncertainty_label} uncertainty."
        }
