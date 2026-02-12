import torch
import sys
import os

# Add root to path
sys.path.append(os.getcwd())

from src.models import TextEncoder, FusionModel, AudioEncoder, VideoEncoder
from src.config import TEXT_DIM, AUDIO_DIM, VIDEO_DIM

def test_text_encoder():
    print("Testing TextEncoder...")
    try:
        model = TextEncoder()
        # Batch size 2, seq len 10
        x = torch.randint(0, 1000, (2, 10))
        mask = torch.ones((2, 10))
        
        # Test with mask
        out = model(x, mask)
        print(f"Output shape with mask: {out.shape}")
        assert out.shape == (2, 64)
        
        # Test tuple input
        out = model((x, mask))
        print(f"Output shape with tuple: {out.shape}")
        assert out.shape == (2, 64)
        print("TextEncoder passed.")
        return True
    except Exception as e:
        print(f"TextEncoder failed: {e}")
        return False

def test_fusion_model():
    print("Testing FusionModel...")
    try:
        audio_enc = AudioEncoder(input_dim=AUDIO_DIM)
        video_enc = VideoEncoder(input_dim=VIDEO_DIM)
        text_enc = TextEncoder()
        model = FusionModel(audio_enc, video_enc, text_enc)
        
        bs = 2
        # Audio: (batch, seq_len, dim)
        audio = torch.randn(bs, 50, AUDIO_DIM)
        # Video: (batch, seq_len, dim)
        video = torch.randn(bs, 50, VIDEO_DIM)
        # Text: (batch, seq_len)
        text = torch.randint(0, 1000, (bs, 10))
        text_mask = torch.ones((bs, 10))
        
        out_b, out_s = model(audio, video, text, text_mask)
        print(f"Binary output shape: {out_b.shape}")
        print(f"Score output shape: {out_s.shape}")
        assert out_b.shape == (bs, 1)
        assert out_s.shape == (bs, 1)
        print("FusionModel passed.")
        return True
    except Exception as e:
        print(f"FusionModel failed: {e}")
        return False

if __name__ == "__main__":
    success = True
    if not test_text_encoder(): success = False
    if not test_fusion_model(): success = False
    
    if success:
        print("All tests passed!")
        exit(0)
    else:
        print("Some tests failed.")
        exit(1)
