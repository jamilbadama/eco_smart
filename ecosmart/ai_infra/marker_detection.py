import torch
import numpy as np
import os
try:
    from transformers import pipeline
except ImportError:
    print("Transformers not installed, text sentiment will differ.")
    pipeline = None

class MarkerDetector:
    def __init__(self, device='cpu'):
        self.device = device
        self.audio_feature_extractor = None
        self.audio_emotion_classifier = None
        
        if pipeline:
            try:
                # Load a more robust specialized depression classifier
                print("DEBUG: Loading text sentiment model (rafalposwiata/deproberta-large-depression)...")
                self.sentiment_analyzer = pipeline("text-classification", model="rafalposwiata/deproberta-large-depression", device=0 if torch.cuda.is_available() else -1)
                print("DEBUG: Text sentiment model loaded.")
            except Exception as e:
                print(f"Could not load specialized depression model: {e}")
            
            try:
                # Load a reliable PyTorch facial emotion classifier
                print("DEBUG: Loading facial emotion model (dima806/facial_emotions_image_detection)...")
                self.emotion_classifier = pipeline("image-classification", model="dima806/facial_emotions_image_detection", device=0 if torch.cuda.is_available() else -1)
                print("DEBUG: Facial emotion model loaded.")
            except Exception as e:
                print(f"Could not load facial emotion model: {e}")

            try:
                # Load Wav2Vec2 for speech emotion recognition
                print("DEBUG: Loading audio emotion model (ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition)...")
                self.audio_emotion_classifier = pipeline("audio-classification", model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition", device=0 if torch.cuda.is_available() else -1)
                print("DEBUG: Audio emotion model loaded.")
            except Exception as e:
                print(f"Could not load audio emotion model: {e}")

            try:
                # Load Wav2Vec2 feature extractor for monotone/pitch detection
                # We use the base model for extracting raw hidden states/features
                print("DEBUG: Loading Wav2Vec2 base model...")
                from transformers import Wav2Vec2Model, Wav2Vec2Processor
                self.audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
                self.audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(self.device)
                print("DEBUG: Wav2Vec2 base model loaded.")
            except Exception as e:
                print(f"Could not load Wav2Vec2 base model: {e}")

    def detect_emotion(self, image):
        """
        Classifies emotion from a PIL image or image path.
        """
        if not self.emotion_classifier:
            return None
            
        try:
            results = self.emotion_classifier(image)
            # Return the top emotion
            return results[0] if results else None
        except Exception as e:
            print(f"Emotion detection error: {e}")
            return None

    def extract_text_markers(self, text):
        """
        Extracts sentiment and word count from raw text.
        """
        markers = {}
        if not text:
            return {'sentiment': 0.5, 'word_count': 0}
            
        markers['word_count'] = len(text.split())
        
        if self.sentiment_analyzer:
            # Truncate to 512 tokens approx
            truncated_text = text[:1000] 
            try:
                result = self.sentiment_analyzer(truncated_text)[0]
                label = result['label'].lower()
                
                # Map specialized labels to a 'risk inverse' sentiment score (1.0 = Low Risk, 0.0 = High Risk)
                if label == 'not depression':
                    markers['sentiment'] = 1.0
                    markers['depression_label'] = 'not depression'
                elif label == 'moderate':
                    markers['sentiment'] = 0.3 # Increased sensitivity
                    markers['depression_label'] = 'moderate'
                elif label == 'severe':
                    markers['sentiment'] = 0.0
                    markers['depression_label'] = 'severe'
                else:
                    markers['sentiment'] = 0.5
                    markers['depression_label'] = label
            except Exception as e:
                print(f"Text marker extraction error: {e}")
                markers['sentiment'] = 0.5
        else:
            markers['sentiment'] = 0.5
            
        return markers

    def extract_audio_markers(self, audio_data, sample_rate=16000):
        """
        Extracts pitch stats, energy proxies, and audio emotion from raw audio or features.
        If audio_data is a numpy array (raw signal), uses Wav2Vec2.
        If audio_data is a 2D array, assumes pre-extracted features (COVAREP/Mel).
        """
        markers = {}
        
        # Handle raw audio signal
        if isinstance(audio_data, (np.ndarray, list, torch.Tensor)) and len(np.shape(audio_data)) == 1:
            raw_signal = np.array(audio_data)
            markers.update(self._extract_neural_audio_markers(raw_signal, sample_rate))
            return markers

        # Existing heuristic logic for pre-extracted feature blocks
        audio_features = audio_data
        if isinstance(audio_features, torch.Tensor):
            audio_features = audio_features.cpu().numpy()
            
        # Check if empty
        if audio_features.shape[0] == 0:
            return {'pitch_mean': 0, 'pitch_std': 0, 'energy_mean': 0}

        # Heuristic: If dim is small (<100), assume COVAREP. 
        # Col 0 = F0, Col 1 = VUV
        if audio_features.shape[1] < 100:
            f0 = audio_features[:, 0]
            vuv = audio_features[:, 1]
            
            # Filter unvoiced
            voiced_f0 = f0[vuv == 1]
            
            if len(voiced_f0) > 0:
                markers['pitch_mean'] = float(np.mean(voiced_f0))
                markers['pitch_std'] = float(np.std(voiced_f0))
            else:
                markers['pitch_mean'] = 0.0
                markers['pitch_std'] = 0.0
                
            # Energy proxy (mean of all features?) or just variation
            markers['activity_mean'] = float(np.mean(np.abs(audio_features)))
        else:
            # Mel Spectrogram
            # Mean energy
            markers['energy_mean'] = float(np.mean(audio_features))
            markers['energy_std'] = float(np.std(audio_features))
            
        return markers

    def _extract_neural_audio_markers(self, raw_signal, sample_rate=16000):
        """
        Analyzes raw WAV signal using Wav2Vec2 models.
        """
        results = {
            'energy_mean': float(np.mean(np.abs(raw_signal))),
            'pitch_std': 0.0, # Placeholder for neural pitch var
            'audio_emotion': 'neutral'
        }

        # 1. Audio Emotion Detection
        if self.audio_emotion_classifier:
            try:
                # The pipeline handles resampling if needed, but we assume 16kHz
                emo_results = self.audio_emotion_classifier(raw_signal)
                if emo_results:
                    results['audio_emotion_label'] = emo_results[0]['label']
                    results['audio_emotion_score'] = emo_results[0]['score']
            except Exception as e:
                print(f"Audio emotion error: {e}")

        # 2. Monotone/Prosody Detection via Wav2Vec2 Base
        if hasattr(self, 'audio_model') and self.audio_model:
            try:
                inputs = self.audio_processor(raw_signal, sampling_rate=sample_rate, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.audio_model(**inputs)
                    # Use variance of the last hidden state as a proxy for prosodic variability (monotone indicator)
                    hidden_states = outputs.last_hidden_state.squeeze(0).cpu().numpy()
                    results['pitch_std'] = float(np.std(hidden_states))
                    # Lower pitch_std in hidden space -> potential monotone speech
            except Exception as e:
                print(f"Wav2Vec2 feature error: {e}")

        return results

    def extract_video_markers(self, video_features):
        """
        Extracts movement markers from visual features.
        video_features: (seq_len, dim)
        """
        markers = {}
        if isinstance(video_features, torch.Tensor):
            video_features = video_features.cpu().numpy()
            
        if video_features.shape[0] < 2:
            return {'movement_mean': 0, 'movement_std': 0}
            
        # Calculate first derivative (velocity/movement)
        diff = video_features[1:] - video_features[:-1]
        
        # Mean absolute movement across all features
        movement = np.mean(np.abs(diff), axis=1)
        
        markers['movement_mean'] = float(np.mean(movement))
        markers['movement_std'] = float(np.std(movement))
        
        return markers

    def process_multimodal_fusion(self, markers):
        """
        Performs high-level spatio-temporal fusion inspired by MMFformer (2025).
        Combines semantic (text), acoustic (audio), and behavioral (video) signals.
        """
        fusion_score = 0.0
        details = []

        text_m = markers.get('text', {})
        audio_m = markers.get('audio', {})
        video_m = markers.get('video', {})

        # 1. Semantic Signal (Text) - High Weight (0.4)
        # We use the specialized depression model output
        text_risk = 1.0 - text_m.get('sentiment', 0.5)
        fusion_score += text_risk * 0.4
        if text_risk > 0.6: details.append("High clinical text risk")

        # 2. Acoustic Signal (Audio Affect & Prosody) - Weight (0.3)
        audio_emotion_risk = 0.0
        if audio_m.get('audio_emotion_label') in ['sad', 'disappointed']:
            audio_emotion_risk = 0.8
        elif audio_m.get('audio_emotion_label') == 'anxious':
            audio_emotion_risk = 0.6
            
        # Monotone check (neural pitch_std proxy)
        monotone_risk = 0.0
        pitch_var = audio_m.get('pitch_std', 0.5)
        if pitch_var < 0.2: # Low variability
            monotone_risk = 0.7
            
        audio_risk = max(audio_emotion_risk, monotone_risk)
        fusion_score += audio_risk * 0.3
        if audio_risk > 0.5: details.append("Acoustic affective/prosodic risk")

        # 3. Behavioral Signal (Video Movement) - Weight (0.3)
        # Reduced movement (psychomotor retardation) is a key MMFformer feature
        movement = video_m.get('movement_mean', 0.1)
        video_risk = 1.0 - np.clip(movement * 10, 0, 1) # Normalizing 0.1 -> 0.0 risk
        fusion_score += video_risk * 0.3
        if video_risk > 0.6: details.append("Psychomotor retardation markers detected")

        # 4. Cross-Modal Agreement Bonus
        # If text indicates depression AND audio is monotone, increase confidence/score
        if text_risk > 0.6 and monotone_risk > 0.5:
            fusion_score = np.clip(fusion_score + 0.1, 0, 1)
            details.append("Spatio-temporal agreement (Text + Audio)")

        return {
            "fusion_risk_score": float(np.round(fusion_score, 3)),
            "detected_patterns": details,
            "confidence": 0.85 # Heuristic for fusion model
        }
