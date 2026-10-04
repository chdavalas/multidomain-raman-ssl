# Code based on the vit_1d from the lucidrains repo: 
# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_1d.py
# MIT license; see THIRD_PARTY_LICENSES/LICENSE-vit-pytorch.txt and LICENSE.

import torch
from torch import nn
import torch.nn.functional as F
import math
import logging

from einops import rearrange

logger = logging.getLogger(__name__)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
    
class spectra_encoder(nn.Module):
    def __init__(self, input_dim, dim, depth, heads, mlp_dim, 
                 pos_embedding_type='sin_cos', dim_head=64, 
                 dropout=0., emb_dropout=0.):
        super().__init__()
        self.pos_embedding_type = pos_embedding_type
        self.dim = dim
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.LayerNorm(dim)
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        if pos_embedding_type == 'trainable':
            self.pos_embedding = nn.Parameter(torch.randn(1, 1, dim))
            logger.info("Trainable positional embeddings were used.")
        elif pos_embedding_type == 'sin_cos':
            self.register_buffer('pos_embedding', None)
            logger.info("Sine and cosine positional embeddings were used.")
        elif pos_embedding_type == 'fourier':
            logger.info("Using Fourier positional encoding.")
        else:
            logger.info("No positional embeddings were used.")
        
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
    
    def compute_sin_cos_pos_embedding(self, seq_len, device):
        position = torch.arange(seq_len, device=device).unsqueeze(0).unsqueeze(-1)
        div_term = torch.exp(
            torch.arange(0, self.dim, 2, device=device) * 
            (-math.log(10000.0) / self.dim)
        )
        pos_embedding = torch.zeros(1, seq_len, self.dim, device=device)
        pos_embedding[0, :, 0::2] = torch.sin(position * div_term)
        pos_embedding[0, :, 1::2] = torch.cos(position * div_term)
        return pos_embedding
    
    def compute_fourier_pos_embedding(self, wavenumbers, D):
        """
        wavenumbers: [B, T] tensor of input wavenumbers
        D: embedding dimension (should equal to self.dim)
        Returns: [B, T, D] Fourier positional encodings
        """
        B, T = wavenumbers.shape

        # Dynamic w_min and w_max per batch
        w_min = wavenumbers.min()
        w_max = wavenumbers.max()

        device = wavenumbers.device
        i = torch.arange(D / 2, device=device).float()

        log_w_min = torch.log(w_min)
        log_w_max = torch.log(w_max)
        log_w_i = log_w_min + i * (log_w_max - log_w_min) / (D / 2 - 1)
        w_i = torch.exp(log_w_i)  # Shape: [D/2]

        angles = 2 * math.pi * wavenumbers.unsqueeze(-1) / w_i  # [B, T, D/2]
        pos_embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)  # [B, T, D]
        return pos_embedding


    def forward(self, x):
        # x shape: [B, T, 2] where x[:,:,0] = wavenumbers, x[:,:,1] = intensities
        B, T, _ = x.shape
        wavenumbers = x[:, :, 0]  # [B, T]

        # Project [wavenumber, intensity] into model dimension
        x = self.input_proj(x)  # [B, T, dim]

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, T+1, dim]

        # Positional encoding
        if self.pos_embedding_type == 'fourier':
            pos_embedding = self.compute_fourier_pos_embedding(wavenumbers, self.dim)  # [B, T, dim]
            # Add a zero vector for CLS token
            cls_pe = torch.zeros(B, 1, self.dim, device=x.device)
            pos_embedding = torch.cat((cls_pe, pos_embedding), dim=1)  # [B, T+1, dim]
            x = x + pos_embedding
        elif self.pos_embedding_type == 'trainable':
            pos_embedding = self.pos_embedding.expand(B, T + 1, -1)
            x = x + pos_embedding
        elif self.pos_embedding_type == 'sin_cos':
            pos_embedding = self.compute_sin_cos_pos_embedding(T + 1, x.device)
            x = x + pos_embedding

        x = self.dropout(x)
        x = self.transformer(x)
        return x