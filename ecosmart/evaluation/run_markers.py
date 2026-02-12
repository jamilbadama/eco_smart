import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr
import sys
import os

# Add root to path
sys.path.append(os.getcwd())

from src.data_loader import DAICWOZDataset
from src.config import *
from src.markers import MarkerDetector

def run_marker_evaluation():
    print("Initializing Marker Experiment...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    detector = MarkerDetector(device=device)
    
    # Load Dev set
    print(f"Loading Dev Split from {DEV_SPLIT}")
    dataset = DAICWOZDataset(DEV_SPLIT, DATA_ROOT, mode='multimodal', return_raw_text=True)
    
    results = []
    
    print(f"Processing {len(dataset)} samples...")
    for i in tqdm(range(len(dataset))):
        try:
            data, binary_label, score_label, pid = dataset[i]
            
            # Text Markers
            raw_text = data.get('raw_text', "")
            text_markers = detector.extract_text_markers(raw_text)
            
            # Audio Markers
            audio_feat = data['audio'] # Torch tensor
            audio_markers = detector.extract_audio_markers(audio_feat)
            
            # Video Markers
            video_feat = data['video']
            video_markers = detector.extract_video_markers(video_feat)
            
            # Combine
            sample_res = {
                'pid': pid,
                'phq8_score': score_label.item(),
                'phq8_binary': binary_label.item(),
                **text_markers,
                **audio_markers,
                **video_markers
            }
            results.append(sample_res)
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
        
    df = pd.DataFrame(results)
    
    if len(df) == 0:
        print("No results generated.")
        return

    # Compute Correlations with PHQ8 Score
    print("\nMarker Correlations with PHQ-8 Score:")
    correlations = {}
    for col in df.columns:
        if col not in ['pid', 'phq8_score', 'phq8_binary']:
            # Drop NaNs
            valid_df = df[[col, 'phq8_score']].dropna()
            # Filter out -1 labels if any (though dev split usually has labels)
            valid_df = valid_df[valid_df['phq8_score'] != -1]
            
            if len(valid_df) > 1:
                corr, p = pearsonr(valid_df[col], valid_df['phq8_score'])
                correlations[col] = corr
                print(f"{col}: r={corr:.3f} (p={p:.3f})")
            else:
                print(f"{col}: Not enough data")
                
    # Save results
    os.makedirs('experiments', exist_ok=True)
    df.to_csv('experiments/marker_results.csv', index=False)
    print("\nSaved detailed marker results to experiments/marker_results.csv")

if __name__ == "__main__":
    run_marker_evaluation()
