"""
Shared model components used in both autoencoder and transition network.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import HID_DIM, EMB_DIM, LATENT_DIM, N_EXPERTS, K, HIDDEN_DIM

class Encoder(nn.Module):
    """GRU-based encoder for sequence encoding."""
    def __init__(self, vocab_size: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMB_DIM)
        self.gru = nn.GRU(EMB_DIM, HID_DIM, batch_first=True)
        
    def forward(self, x):
        _, h = self.gru(self.emb(x))
        return h.squeeze(0)

class Decoder(nn.Module):
    """GRU-based decoder for sequence generation."""
    def __init__(self, vocab_size: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMB_DIM)
        self.gru = nn.GRU(EMB_DIM, HID_DIM, batch_first=True)
        self.fc = nn.Linear(HID_DIM, vocab_size)
        
    def forward(self, tok, h):
        e = self.emb(tok).unsqueeze(1)
        out, h2 = self.gru(e, h)
        return self.fc(out.squeeze(1)), h2

class SparseMoE(nn.Module):
    """Sparse Mixture of Experts layer."""
    def __init__(self, k: int = 2):
        super().__init__()
        self.n_experts = N_EXPERTS
        self.k = k
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(LATENT_DIM, HIDDEN_DIM),
                nn.ReLU(),
                nn.Linear(HIDDEN_DIM, LATENT_DIM)
            ) for _ in range(N_EXPERTS)
        ])
        self.gate = nn.Linear(LATENT_DIM, N_EXPERTS)
        
    def forward(self, z):
        w = F.softmax(self.gate(z), dim=-1)
        topw, idx = w.topk(self.k, dim=-1)
        out = torch.zeros_like(z)
        for b in range(z.size(0)):
            for wi, ei in zip(topw[b], idx[b]):
                out[b] += wi * self.experts[ei](z[b:b+1]).squeeze(0)
        return out

class KeyValueMemory(nn.Module):
    """Key-Value Memory module."""
    def __init__(self):
        super().__init__()
        self.keys = nn.Parameter(torch.randn(K, LATENT_DIM))
        self.values = nn.Parameter(torch.randn(K, LATENT_DIM))
        
    def forward(self, z):
        scores = torch.matmul(z, self.keys.t())
        attn = F.softmax(scores, dim=-1)
        mem = torch.matmul(attn, self.values)
        return mem

class GatedMLPBlock(nn.Module):
    """Gated MLP block with residual connection."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(LATENT_DIM, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.gate = nn.Linear(LATENT_DIM, LATENT_DIM)
        
    def forward(self, x):
        h = F.gelu(self.fc1(x))
        h = self.fc2(h)
        g = torch.sigmoid(self.gate(x))
        return x + g * h

class HyperAdapter(nn.Module):
    """Hyper network for generating decoder biases."""
    def __init__(self, vocab_size: int):
        super().__init__()
        self.hyper = nn.Linear(LATENT_DIM, vocab_size)
        
    def forward(self, z):
        return self.hyper(z) 