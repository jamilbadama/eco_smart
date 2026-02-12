import torch
import json
from ecosmart.ai_infra.marker_detection import MarkerDetector
from ecosmart.ai_infra.triangulation import TriangulationEngine

def test_mmfformer_convergence():
    print("Testing MMFformer Spatio-Temporal Fusion Convergence...")
    
    detector = MarkerDetector()
    triangulator = TriangulationEngine()
    
    # CASE 1: High Clinical Risk (Convergence)
    # - Text: Moderate depression
    # - Audio: Sad / Monotone (low pitch_std)
    # - Video: Low movement (Psychomotor retardation)
    markers_high = {
        "text": {"sentiment": 0.3, "depression_label": "moderate", "word_count": 10},
        "audio": {"audio_emotion_label": "sad", "pitch_std": 0.1, "energy_mean": 0.05},
        "video": {"movement_mean": 0.02}
    }
    
    fusion_high = detector.process_multimodal_fusion(markers_high)
    print(f"\n--- Case 1: High Convergence ---")
    print(f"Fusion Result: {fusion_high}")
    
    evidence_high = triangulator.triangulate(
        model_outputs={"binary_prob": 0.7, "score": 16},
        marker_scores=markers_high,
        self_reports={"phq8": 12},
        fusion_result=fusion_high
    )
    print(f"Convergence Summary: {evidence_high['convergence_summary']}")
    print(f"Agreement Points: {evidence_high['agreement_points']}")

    # CASE 2: Mixed Signals
    markers_mixed = {
        "text": {"sentiment": 0.9, "depression_label": "not depression"},
        "audio": {"audio_emotion_label": "sad", "pitch_std": 0.1, "energy_mean": 0.05}
    }
    fusion_mixed = detector.process_multimodal_fusion(markers_mixed)
    print(f"\n--- Case 2: Mixed Signals ---")
    print(f"Fusion Result: {fusion_mixed}")

if __name__ == "__main__":
    test_mmfformer_convergence()
