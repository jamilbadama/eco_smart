import argparse
from src.train import run_training
from src.config import FUSION_TYPE, USE_RAW_AUDIO

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='multimodal', choices=['audio', 'video', 'text', 'multimodal'])
    parser.add_argument('--use_raw_audio', action='store_true', default=USE_RAW_AUDIO, help='Use raw audio (Mel-spectrograms) instead of COVAREP')
    parser.add_argument('--fusion_type', type=str, default=FUSION_TYPE, choices=['early', 'late'], help='Fusion strategy')
    args = parser.parse_args()
    
    # Update config dynamically if needed, or pass to run_training
    # For simplicity, we might need to modify run_training signature
    run_training(args.mode, args.use_raw_audio, args.fusion_type)
