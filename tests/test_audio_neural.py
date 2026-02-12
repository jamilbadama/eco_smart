import torch
import numpy as np
from ecosmart.ai_infra.marker_detection import MarkerDetector

def test_audio_markers():
    print("Testing MarkerDetector with Wav2Vec2 Audio Models...")
    detector = MarkerDetector()
    
    # Simulate 1 second of audio at 16kHz
    duration = 1.0
    fs = 16000
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    # 1. Sine wave (Neutral/Robotic)
    sine_wave = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    # 2. White noise (Anxious/Energy)
    noise = np.random.uniform(-0.1, 0.1, size=len(t))
    
    print("\n--- Testing Sine Wave (Simulated Neutral Speech) ---")
    markers_sine = detector.extract_audio_markers(sine_wave)
    print(f"Markers: {markers_sine}")
    
    print("\n--- Testing Noise (Simulated Agitated/Background) ---")
    markers_noise = detector.extract_audio_markers(noise)
    print(f"Markers: {markers_noise}")

if __name__ == "__main__":
    test_audio_markers()
