import torch
import torch.nn as nn
from .config import AUDIO_DIM, VIDEO_DIM, TEXT_DIM
from transformers import DistilBertModel

class AudioEncoder(nn.Module):
    def __init__(self, input_dim=AUDIO_DIM, hidden_dim=128, num_layers=2):
        super(AudioEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 64)
        
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        # We can use the last hidden state or max pooling over time.
        # Let's use last hidden state for simplicity, or mean pooling.
        output, (h_n, c_n) = self.lstm(x)
        # h_n: (num_layers*2, batch, hidden_dim)
        # Concatenate last forward and backward hidden states
        hidden = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        return self.fc(hidden)

class VideoEncoder(nn.Module):
    def __init__(self, input_dim=VIDEO_DIM, hidden_dim=128, num_layers=2):
        super(VideoEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 64)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        hidden = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        return self.fc(hidden)

class TextEncoder(nn.Module):
    def __init__(self, input_dim=TEXT_DIM, hidden_dim=128):
        super(TextEncoder, self).__init__()
        # Load pre-trained DistilBert
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        # Freeze BERT parameters to speed up training for MVP (optional: unfreeze last layer)
        for param in self.bert.parameters():
            param.requires_grad = False
            
        # Project BERT output (768) to match other modalities (64)
        # We can use a simple linear layer or an LSTM on top of BERT embeddings.
        # For Eco-SMART MVP, let's use a linear projection of the CLS token.
        self.fc = nn.Sequential(
            nn.Linear(768, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, x, mask=None):
        # Handle tuple input (from SingleModalityModel)
        if isinstance(x, (tuple, list)):
            if len(x) >= 2:
                x, mask = x[0], x[1]
            else:
                x = x[0]

        # x: (batch, seq_len) of token ids
        # mask: (batch, seq_len) of attention mask
        
        if mask is None:
            # Create dummy mask (all ones) if not provided, though it's recommended to provide it
            mask = torch.ones_like(x)
            
        # DistilBert forward
        outputs = self.bert(input_ids=x, attention_mask=mask)
        
        # Get CLS token state (first token)
        last_hidden_state = outputs.last_hidden_state # (batch, seq_len, 768)
        cls_state = last_hidden_state[:, 0, :] # (batch, 768)
        
        return self.fc(cls_state)

class FusionModel(nn.Module):
    def __init__(self, audio_enc, video_enc, text_enc, fusion_type='early'):
        super(FusionModel, self).__init__()
        self.audio_enc = audio_enc
        self.video_enc = video_enc
        self.text_enc = text_enc
        self.fusion_type = fusion_type
        
        if self.fusion_type == 'early':
            # Fusion layer
            # 64 (audio) + 64 (video) + 64 (text) = 192
            self.fusion_fc = nn.Sequential(
                nn.Linear(64 * 3, 64),
                nn.ReLU(),
                nn.Dropout(0.5)
            )
            
            # Heads
            self.binary_head = nn.Linear(64, 1)
            self.regression_head = nn.Linear(64, 1)
            
        elif self.fusion_type == 'late':
            # Modality-specific heads for late fusion
            self.audio_head_b = nn.Linear(64, 1)
            self.audio_head_s = nn.Linear(64, 1)
            
            self.video_head_b = nn.Linear(64, 1)
            self.video_head_s = nn.Linear(64, 1)
            
            self.text_head_b = nn.Linear(64, 1)
            self.text_head_s = nn.Linear(64, 1)
            
            # Learnable weights for averaging (Binary and Score)
            self.fusion_weights_b = nn.Parameter(torch.ones(3))
            self.fusion_weights_s = nn.Parameter(torch.ones(3))

    def forward(self, audio, video, text, text_mask=None):
        a_emb = self.audio_enc(audio)
        v_emb = self.video_enc(video)
        t_emb = self.text_enc(text, text_mask)
        
        if self.fusion_type == 'early':
            # Concatenate
            fused = torch.cat((a_emb, v_emb, t_emb), dim=1)
            fused = self.fusion_fc(fused)
            
            binary_out = self.binary_head(fused)
            score_out = self.regression_head(fused)
            
        elif self.fusion_type == 'late':
            # Independent predictions
            ab = self.audio_head_b(a_emb)
            as_ = self.audio_head_s(a_emb)
            
            vb = self.video_head_b(v_emb)
            vs = self.video_head_s(v_emb)
            
            tb = self.text_head_b(t_emb)
            ts = self.text_head_s(t_emb)
            
            # Softmax weights
            w_b = torch.softmax(self.fusion_weights_b, dim=0)
            w_s = torch.softmax(self.fusion_weights_s, dim=0)
            
            # Weighted average
            binary_out = w_b[0] * ab + w_b[1] * vb + w_b[2] * tb
            score_out = w_s[0] * as_ + w_s[1] * vs + w_s[2] * ts
            
        return binary_out, score_out

class SingleModalityModel(nn.Module):
    def __init__(self, encoder):
        super(SingleModalityModel, self).__init__()
        self.encoder = encoder
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.binary_head = nn.Linear(32, 1)
        self.regression_head = nn.Linear(32, 1)
        
    def forward(self, x):
        emb = self.encoder(x)
        feat = self.fc(emb)
        return self.binary_head(feat), self.regression_head(feat)
