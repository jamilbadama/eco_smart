import cv2
import numpy as np
import base64
import torch
from datetime import datetime
from PIL import Image
from ..ai_infra.marker_detection import MarkerDetector
from ..ai_infra.triangulation import TriangulationEngine
from ..ai_infra.interpretation import InterpretationEngine

class RealtimeMonitorService:
    def __init__(self):
        self.marker_detector = MarkerDetector()
        self.triangulation_engine = TriangulationEngine()
        self.interpretation_engine = InterpretationEngine()
        
        # Buffer for rolling analysis
        self.video_buffer = []
        self.audio_buffer = []
        self.max_buffer_size = 50 # frames/chunks
        
        # Session Data Accumulation
        self.session_markers = []
        self.session_interpretations = []
        self.session_emotions = []
        
        # Emotion detection state
        self.emotion_counter = 0
        self.detect_emotion_every = 5 # approx every 1 second at 5fps
        self.last_emotion = {"label": "neutral", "score": 1.0}
        
    async def process_video_frame(self, base64_frame: str):
        """
        Receives a base64 encoded frame, decodes it, and extracts behavioral markers.
        """
        try:
            # Decode base64 to image
            encoded_data = base64_frame.split(',')[1] if ',' in base64_frame else base64_frame
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return None

            # Convert to gray for basic movement analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Simple movement proxy: contrast/intensity
            movement_proxy = np.mean(gray) / 255.0
            
            self.video_buffer.append(np.array([movement_proxy]))
            if len(self.video_buffer) > self.max_buffer_size:
                self.video_buffer.pop(0)
                
            # Periodic Emotion Detection
            self.emotion_counter += 1
            if self.emotion_counter >= self.detect_emotion_every:
                self.emotion_counter = 0
                # Convert CV2 frame to PIL for transformers pipeline
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                emotion_result = self.marker_detector.detect_emotion(pil_img)
                if emotion_result:
                    print(f"DEBUG: Detected Emotion: {emotion_result['label']} (Score: {emotion_result['score']:.2f})")
                    self.last_emotion = emotion_result
                else:
                    print("DEBUG: No emotion detected in this frame.")

            # Extract markers if we have enough data
            if len(self.video_buffer) >= 2:
                video_features = torch.tensor(np.array(self.video_buffer), dtype=torch.float32)
                markers = self.marker_detector.extract_video_markers(video_features)
                return markers
            
            return {"movement_mean": 0, "movement_std": 0}
            
        except Exception as e:
            print(f"Error processing video frame: {e}")
            return None

    async def process_audio_chunk(self, audio_data: list):
        """
        Processes a chunk of audio samples.
        """
        try:
            # audio_data is expected to be a list of floats (PCM)
            samples = np.array(audio_data)
            
            # Energy proxy
            energy = np.mean(np.abs(samples)) if len(samples) > 0 else 0
            
            self.audio_buffer.append(energy)
            if len(self.audio_buffer) > self.max_buffer_size:
                self.audio_buffer.pop(0)
                
            # Return current energy marker
            return {
                "energy_mean": float(np.mean(self.audio_buffer)),
                "current_energy": float(energy)
            }
        except Exception as e:
            print(f"Error processing audio chunk: {e}")
            return None

    def get_rolling_analysis(self):
        """
        Produces a fused analysis based on the current buffers.
        """
        # Mocking model outputs for real-time flow
        # In a real scenario, we might run the FusionModel on the buffer
        video_feat = np.mean(self.video_buffer) if self.video_buffer else 0
        audio_feat = np.mean(self.audio_buffer) if self.audio_buffer else 0
        
        # Heuristic risk calculation
        risk_prob = float(np.clip(0.3 + (video_feat * 0.2) + (audio_feat * 5.0), 0, 1))
        
        markers = {
            "video": {"movement_mean": video_feat},
            "audio": {"energy_mean": audio_feat},
            "text": {"sentiment": 0.5} # Real-time text requires ASR
        }
        
        # 2b. High-Level Hub Fusion (MMFformer-style)
        fusion_result = self.marker_detector.process_multimodal_fusion(markers)
        
        # Placeholder for model outputs (using heuristic risk for real-time proxy)
        model_outputs = {
            "binary_prob": risk_prob,
            "score": risk_prob * 24, # Map to PHQ-8 scale
            "text": [{"label": "neutral", "score": 0.5}],
            "audio": [{"label": "neutral", "score": 0.5}],
            "video": [{"label": "neutral", "score": 0.5}]
        }
        
        evidence = self.triangulation_engine.triangulate(model_outputs, markers, fusion_result=fusion_result)
        interpretation = self.interpretation_engine.interpret(evidence)
        
        analysis = {
            "markers": markers,
            "evidence": evidence,
            "interpretation": interpretation,
            "emotion": self.last_emotion
        }

        # Store for session summary
        self.session_markers.append(markers)
        self.session_interpretations.append(interpretation)
        self.session_emotions.append(self.last_emotion)

        return analysis

    def get_session_summary(self, session_id):
        """
        Compiles the final session data from accumulated history.
        """
        if not self.session_interpretations:
            return None

        # Take the last interpretation as the final one, but average the probability
        final_interpretation = self.session_interpretations[-1]
        
        # Calculate mean markers
        avg_movement = float(np.mean([m["video"]["movement_mean"] for m in self.session_markers])) if self.session_markers else 0
        avg_energy = float(np.mean([m["audio"]["energy_mean"] for m in self.session_markers])) if self.session_markers else 0
        
        # Get dominant emotion
        emotion_labels = [e["label"] for e in self.session_emotions]
        dominant_emotion = max(set(emotion_labels), key=emotion_labels.count) if emotion_labels else "neutral"

        return {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "analysis": {
                "markers": {
                    "text": {"sentiment": 0.5, "word_count": 0},
                    "audio": {"energy_mean": avg_energy},
                    "video": {"movement_mean": avg_movement},
                    "dominant_emotion": dominant_emotion
                },
                "evidence": {
                    "primary_signals": {
                        "binary_prob": final_interpretation.get("confidence", 0.5), # Proxy for prob
                        "score": final_interpretation.get("confidence", 0.5) * 24
                    },
                    "behavioral_markers": {
                        "audio": {"energy_mean": avg_energy},
                        "video": {"movement_mean": avg_movement}
                    },
                    "clinical_grounding": {},
                    # High-Level Fusion Summary (MMFformer-style)
                    "fusion_assessment": self.marker_detector.process_multimodal_fusion({
                        "text": {"sentiment": 0.5},
                        "audio": {"energy_mean": avg_energy},
                        "video": {"movement_mean": avg_movement}
                    }),
                    "provenance": {
                        "models": ["MMFformer_V1", "RealtimeFusion_V1"],
                        "markers": ["RealtimeMarkerDetector_V1"]
                    }
                },
                "interpretation": final_interpretation
            },
            "trends": {
                "status": "live_capture",
                "message": f"Real-time session captured with dominant emotion: {dominant_emotion}"
            }
        }
