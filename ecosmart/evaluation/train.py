import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .data_loader import DAICWOZDataset
from .models import AudioEncoder, VideoEncoder, TextEncoder, FusionModel, SingleModalityModel
from .config import *
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, mean_squared_error

def train_epoch(model, loader, optimizer, criterion_binary, criterion_score, device, mode='multimodal'):
    model.train()
    total_loss = 0
    
    for batch in tqdm(loader, desc="Training"):
        data, binary_label, score_label, _ = batch
        
        binary_label = binary_label.to(device).unsqueeze(1)
        score_label = score_label.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        
        if mode == 'multimodal':
            audio = data['audio'].to(device)
            video = data['video'].to(device)
            text = data['text'].to(device)
            text_mask = data.get('text_mask').to(device) if 'text_mask' in data else None
            
            binary_out, score_out = model(audio, video, text, text_mask)
        elif mode == 'audio':
            audio = data['audio'].to(device)
            binary_out, score_out = model(audio)
        elif mode == 'video':
            video = data['video'].to(device)
            binary_out, score_out = model(video)
        elif mode == 'text':
            text = data['text'].to(device)
            text_mask = data.get('text_mask').to(device) if 'text_mask' in data else None
            binary_out, score_out = model((text, text_mask))
            
        # Multi-task loss
        loss_b = criterion_binary(binary_out, binary_label)
        loss_s = criterion_score(score_out, score_label)
        loss = loss_b + loss_s
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)

def evaluate(model, loader, device, mode='multimodal'):
    model.eval()
    preds_binary = []
    targets_binary = []
    preds_score = []
    targets_score = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            data, binary_label, score_label, _ = batch
            
            binary_label = binary_label.to(device).unsqueeze(1)
            score_label = score_label.to(device).unsqueeze(1)
            
            if mode == 'multimodal':
                audio = data['audio'].to(device)
                video = data['video'].to(device)
                text = data['text'].to(device)
                text_mask = data.get('text_mask').to(device) if 'text_mask' in data else None
                binary_out, score_out = model(audio, video, text, text_mask)
            elif mode == 'audio':
                audio = data['audio'].to(device)
                binary_out, score_out = model(audio)
            elif mode == 'video':
                video = data['video'].to(device)
                binary_out, score_out = model(video)
            elif mode == 'text':
                text = data['text'].to(device)
                text_mask = data.get('text_mask').to(device) if 'text_mask' in data else None
                binary_out, score_out = model((text, text_mask))
            
            preds_binary.extend(torch.sigmoid(binary_out).cpu().numpy())
            targets_binary.extend(binary_label.cpu().numpy())
            preds_score.extend(score_out.cpu().numpy())
            targets_score.extend(score_label.cpu().numpy())
            
    # Filter out NaNs from targets and predictions before metric calculation
    preds_binary = np.array(preds_binary)
    targets_binary = np.array(targets_binary)
    preds_score = np.array(preds_score)
    targets_score = np.array(targets_score)
    
    # Mask for valid binary labels (not -1 and not NaN)
    mask_b = (targets_binary != -1) & (~np.isnan(targets_binary))
    if mask_b.sum() > 0:
        f1 = f1_score(targets_binary[mask_b], preds_binary[mask_b] > 0.5)
    else:
        f1 = 0.0
        
    # Mask for valid score labels (not -1 and not NaN)
    mask_s = (targets_score != -1) & (~np.isnan(targets_score)) & (~np.isnan(preds_score))
    if mask_s.sum() > 0:
        rmse = np.sqrt(mean_squared_error(targets_score[mask_s], preds_score[mask_s]))
    else:
        rmse = 0.0
    
    return f1, rmse

def run_training(mode='multimodal', use_raw_audio=False, fusion_type='early'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_dataset = DAICWOZDataset(TRAIN_SPLIT, DATA_ROOT, mode=mode, use_raw_audio=use_raw_audio)
    dev_dataset = DAICWOZDataset(DEV_SPLIT, DATA_ROOT, mode=mode, use_raw_audio=use_raw_audio)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    if mode == 'multimodal':
        audio_dim = MEL_DIM if use_raw_audio else AUDIO_DIM
        audio_enc = AudioEncoder(input_dim=audio_dim)
        video_enc = VideoEncoder()
        text_enc = TextEncoder()
        model = FusionModel(audio_enc, video_enc, text_enc, fusion_type=fusion_type).to(device)
    elif mode == 'audio':
        audio_dim = MEL_DIM if use_raw_audio else AUDIO_DIM
        model = SingleModalityModel(AudioEncoder(input_dim=audio_dim)).to(device)
    elif mode == 'video':
        model = SingleModalityModel(VideoEncoder()).to(device)
    elif mode == 'text':
        model = SingleModalityModel(TextEncoder()).to(device)
        
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion_binary = nn.BCEWithLogitsLoss()
    criterion_score = nn.MSELoss()
    
    # Logging
    log_file = f"experiments/log_{mode}.csv"
    with open(log_file, "w") as f:
        f.write("epoch,train_loss,dev_f1,dev_rmse\n")
    
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion_binary, criterion_score, device, mode)
        f1, rmse = evaluate(model, dev_loader, device, mode)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss:.4f} - Dev F1: {f1:.4f} - Dev RMSE: {rmse:.4f}")
        
        with open(log_file, "a") as f:
            f.write(f"{epoch+1},{train_loss},{f1},{rmse}\n")
        
    # Save model
    torch.save(model.state_dict(), f"experiments/model_{mode}.pth")
