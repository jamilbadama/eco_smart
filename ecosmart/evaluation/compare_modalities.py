import torch
from torch.utils.data import DataLoader
from src.data_loader import DAICWOZDataset
from src.models import AudioEncoder, VideoEncoder, TextEncoder, FusionModel, SingleModalityModel
from src.train import evaluate
from src.config import *
import pandas as pd
import os

def load_model(mode, device, use_raw_audio=False):
    model_path = f"experiments/model_{mode}.pth"
    if not os.path.exists(model_path):
        print(f"Warning: Model for {mode} not found at {model_path}")
        return None
        
    if mode == 'multimodal':
        audio_dim = MEL_DIM if use_raw_audio else AUDIO_DIM
        audio_enc = AudioEncoder(input_dim=audio_dim)
        video_enc = VideoEncoder()
        text_enc = TextEncoder()
        model = FusionModel(audio_enc, video_enc, text_enc)
    elif mode == 'audio':
        audio_dim = MEL_DIM if use_raw_audio else AUDIO_DIM
        model = SingleModalityModel(AudioEncoder(input_dim=audio_dim))
    elif mode == 'video':
        model = SingleModalityModel(VideoEncoder())
    elif mode == 'text':
        model = SingleModalityModel(TextEncoder())
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model

def compare_models(use_raw_audio=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Dev Set (or Test if available)
    dataset = DAICWOZDataset(DEV_SPLIT, DATA_ROOT, mode='multimodal', use_raw_audio=use_raw_audio)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    modes = ['audio', 'video', 'text', 'multimodal']
    results = []
    
    for mode in modes:
        print(f"Evaluating {mode}...")
        model = load_model(mode, device, use_raw_audio)
        if model:
            f1, rmse = evaluate(model, loader, device, mode)
            results.append({'Modality': mode, 'F1 Score': f1, 'RMSE': rmse})
        else:
            results.append({'Modality': mode, 'F1 Score': 'N/A', 'RMSE': 'N/A'})
            
    df = pd.DataFrame(results)
    print("\n=== Evaluation Comparison ===")
    print(df.to_markdown(index=False))
    
    # Save to file
    with open("experiments/comparison_results.md", "w") as f:
        f.write(df.to_markdown(index=False))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_raw_audio', action='store_true')
    args = parser.parse_args()
    
    compare_models(args.use_raw_audio)
