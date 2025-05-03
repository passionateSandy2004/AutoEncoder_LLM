"""
Composite transition network for mapping between latent spaces.
"""
import torch
import torch.nn as nn
from models.components import SparseMoE, KeyValueMemory, GatedMLPBlock, HyperAdapter

class CompositeTransitionNet(nn.Module):
    """Composite transition network combining multiple components."""
    def __init__(self, vocab_size: int):
        super().__init__()
        self.moe = SparseMoE()
        self.mem = KeyValueMemory()
        self.gmlp1 = GatedMLPBlock()
        self.gmlp2 = GatedMLPBlock()
        self.adapter = HyperAdapter(vocab_size)
        
    def forward(self, z):
        """
        Forward pass through the transition network.
        
        Args:
            z: Input latent representation
            
        Returns:
            z2hat: Predicted output latent representation
            bias_hat: Predicted decoder bias
        """
        # Process through components
        x = self.moe(z)
        m = self.mem(z)
        x = x + m
        x = self.gmlp1(x)
        x = self.gmlp2(x)
        
        return x, self.adapter(x)  # (z2hat, bias_hat) 