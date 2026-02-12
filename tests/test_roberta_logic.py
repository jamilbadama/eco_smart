import torch
from ecosmart.ai_infra.marker_detection import MarkerDetector

def test_marker_detection():
    print("Testing MarkerDetector with Specialized RoBERTa Model...")
    detector = MarkerDetector()
    
    test_cases = [
        ("I am feeling great today, very happy and full of energy!", "Low Risk / No Depression"),
        ("I feel a bit down lately, not much interest in things.", "Moderate Depression"),
        ("I am completely hopeless and can't get out of bed. Life is too hard.", "Severe Depression")
    ]
    
    for text, expected in test_cases:
        print(f"\nInput: {text}")
        markers = detector.extract_text_markers(text)
        print(f"Result: {markers}")
        
if __name__ == "__main__":
    test_marker_detection()
