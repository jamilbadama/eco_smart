import os
import pandas as pd
import torch
import numpy as np
from ..config import AUDIO_DIM, VIDEO_DIM, TEXT_DIM, MAX_SEQ_LEN, DATA_ROOT
from .preprocess import preprocess_audio, normalize_text, pad_or_truncate

class DataIngestionService:
    def __init__(self, adapter):
        self.adapter = adapter

    def get_session_data(self, session_id):
        return self.adapter.load_session(session_id)

class DAICWOZAdapter:
    def __init__(self, root_dir=DATA_ROOT):
        self.root_dir = root_dir
        
    def load_session(self, participant_id):
        """
        Loads a single session from the DAIC-WOZ structure.
        """
        # Try direct or in 'sessions' subdir
        possible_dirs = [
            os.path.join(self.root_dir, f"{participant_id}_P"),
            os.path.join(self.root_dir, participant_id),
            os.path.join(self.root_dir, 'sessions', f"{participant_id}_P"),
            os.path.join(self.root_dir, 'sessions', participant_id)
        ]
        
        session_dir = None
        for d in possible_dirs:
            if os.path.exists(d):
                session_dir = d
                break
        
        if not session_dir:
            raise FileNotFoundError(f"Session directory not found for {participant_id} in {self.root_dir}")

        data = {}
        
        # Audio
        wav_file = os.path.join(session_dir, f"{participant_id}_AUDIO.wav")
        mel_spec = preprocess_audio(wav_file)
        if mel_spec is not None:
            mel_spec = pad_or_truncate(mel_spec, MAX_SEQ_LEN, 80)
            data['audio'] = torch.tensor(mel_spec, dtype=torch.float32)
        else:
            data['audio'] = torch.zeros((MAX_SEQ_LEN, 80), dtype=torch.float32)

        # Video
        video_file = os.path.join(session_dir, f"{participant_id}_CLNF_features.txt")
        video_feat = self._load_csv_feature(video_file, VIDEO_DIM)
        data['video'] = torch.tensor(video_feat, dtype=torch.float32)

        # Text
        transcript_file = os.path.join(session_dir, f"{participant_id}_TRANSCRIPT.csv")
        raw_text = self._load_transcript(transcript_file)
        raw_text = normalize_text(raw_text)
        data['raw_text'] = raw_text
        
        # Tokenization (deferred to AI layer or done here if needed)
        # For now, we keep raw_text and can tokenize in the model wrapper.
        
        return data

    def _load_csv_feature(self, file_path, feature_dim):
        if not os.path.exists(file_path):
            return np.zeros((MAX_SEQ_LEN, feature_dim))
        try:
            df = pd.read_csv(file_path).select_dtypes(include=[np.number])
            features = df.values
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            features = np.clip(features, -1e4, 1e4)
            # Basic normalization
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0)
            features = (features - mean) / (std + 1e-6)
            return pad_or_truncate(features, MAX_SEQ_LEN, feature_dim)
        except:
            return np.zeros((MAX_SEQ_LEN, feature_dim))

    def _load_transcript(self, file_path):
        if not os.path.exists(file_path):
            return ""
        try:
            df = pd.read_csv(file_path, sep='\t')
            if 'speaker' in df.columns:
                df = df[df['speaker'] == 'Participant']
            if 'value' in df.columns:
                return " ".join(df['value'].astype(str).tolist())
            return ""
        except:
            return ""
