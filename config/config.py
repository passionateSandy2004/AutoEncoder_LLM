"""
Configuration parameters for the autoencoder-based language model.
"""

# Model Architecture Parameters
EMB_DIM = 64          # Embedding dimension
HID_DIM = 32          # Hidden dimension for GRU
LATENT_DIM = 124      # Latent space dimension
VOCAB_SIZE = None     # Will be set during runtime

# Training Parameters
BATCH_SIZE = 16
LR = 1e-4             # Learning rate
AE_EPOCHS = 6000      # Autoencoder training epochs
TRANS_EPOCHS = 5000   # Transition network training epochs

# Special Tokens
PAD = 0
SOS = 1
EOS = 2

# Device Configuration
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model Components
N_EXPERTS = 4         # Number of experts in SparseMoE
K = 256               # Number of memory slots in KeyValueMemory
HIDDEN_DIM = 128      # Hidden dimension for MLP blocks 