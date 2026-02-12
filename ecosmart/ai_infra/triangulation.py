import torch

class TriangulationEngine:
    """
    Converges multiple modes (text, audio, video) + sources (questionnaires, clinician notes).
    Produces a fused 'clinical evidence summary' with provenance.
    """
    def __init__(self, config=None):
        self.config = config

    def triangulate(self, model_outputs, marker_scores, self_reports=None, fusion_result=None):
        """
        model_outputs: dict with 'binary_prob' and 'score' (Traditional/Static models)
        marker_scores: dict with 'text', 'audio', 'video' markers
        self_reports: dict with questionnaire scores (e.g., PHQ-8)
        fusion_result: dict from MMFformer-style fusion hub
        """
        evidence = {
            "primary_signals": model_outputs,
            "behavioral_markers": marker_scores,
            "clinical_grounding": self_reports or {},
            "fusion_assessment": fusion_result or {},
            "provenance": {
                "models": ["MMFformer_V1", "FusionModel_V1"],
                "markers": ["MarkerDetector_V1"]
            }
        }
        
        # Advanced clinical agreement logic
        agreement = []
        
        # 1. Traditional/Static Model Check
        if model_outputs.get('binary_prob', 0) > 0.6:
            agreement.append("Static AI Fusion Model indicates risk")
        
        # 2. Self-report Check
        phq_score = self_reports.get('phq8', 0) if self_reports else 0
        if phq_score >= 10:
            agreement.append("Self-report (PHQ-8) indicates clinical significance")

        # 3. High-Level Fusion (MMFformer) Check
        if fusion_result and fusion_result.get('fusion_risk_score', 0) > 0.6:
            agreement.append(f"MMFformer fusion identifies high-level risk patterns: {', '.join(fusion_result.get('detected_patterns', []))}")

        # 4. Specialized Biomarker Agreement
        text_markers = marker_scores.get('text', {})
        if text_markers.get('depression_label') in ['moderate', 'severe']:
            agreement.append(f"Specialized Text AI identifies {text_markers['depression_label']} risk markers")

        audio_markers = marker_scores.get('audio', {})
        if audio_markers.get('audio_emotion_label') in ['sad', 'disappointed', 'anxious']:
            agreement.append(f"Neural Audio Analysis detects persistent {audio_markers['audio_emotion_label']} affect")

        if audio_markers.get('pitch_std', 1.0) < 0.3: # Neural proxy threshold
             agreement.append("Acoustic markers indicate potential psychomotor retardation (monotone speech)")
            
        # Convergence Logic
        if len(agreement) >= 4:
            evidence["convergence_summary"] = "Extremely strong convergence: All modalities and high-level fusion confirm significant clinical risk."
        elif len(agreement) >= 2:
            evidence["convergence_summary"] = "Moderate convergence: Evidence indicates consistent patterns of risk."
        else:
            evidence["convergence_summary"] = "Preliminary findings: Evidence is mixed or limited."

        evidence["agreement_points"] = agreement
        
        return evidence
