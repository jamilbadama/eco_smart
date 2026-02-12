import os
import numpy as np
import librosa
import re
import torch

def preprocess_audio(file_path, target_sr=16000, n_mels=80, n_fft=400, hop_length=160):
    """
    Loads audio, resamples, applies VAD (simple energy), and extracts Log Mel-spectrogram.
    """
    if not os.path.exists(file_path):
        return None
        
    try:
        # Load and resample
        y, sr = librosa.load(file_path, sr=target_sr)
        
        # Simple Energy-based VAD / Trimming
        # Trim leading/trailing silence
        y, _ = librosa.effects.trim(y, top_db=20)
        
        # Extract Mel-spectrogram
        melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
        log_melspec = librosa.power_to_db(melspec, ref=np.max)
        
        # Shape: (n_mels, time) -> (time, n_mels)
        return log_melspec.T
    except Exception as e:
        print(f"Error processing audio {file_path}: {e}")
        return None

def normalize_text(text):
    """
    Lowercase, remove disfluency markers (e.g., 'um', 'uh'), and punctuation.
    """
    if not isinstance(text, str):
        return ""
        
    text = text.lower()
    
    # Remove disfluencies (simple list)
    disfluencies = ['um', 'uh', 'er', 'ah']
    for d in disfluencies:
        text = re.sub(r'\b' + d + r'\b', '', text)
        
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def pad_or_truncate(features, max_len, feature_dim):
    """
    Utility to ensure feature tensors have consistent (max_len, feature_dim) shape.
    """
    seq_len = features.shape[0]
    if seq_len > max_len:
        features = features[:max_len, :]
    elif seq_len < max_len:
        padding = np.zeros((max_len - seq_len, features.shape[1]))
        features = np.vstack((features, padding))
        
    # Ensure dimension matches expected
    current_dim = features.shape[1]
    if current_dim > feature_dim:
        features = features[:, :feature_dim]
    elif current_dim < feature_dim:
        padding = np.zeros((max_len, feature_dim - current_dim))
        features = np.hstack((features, padding))
        
    return features
