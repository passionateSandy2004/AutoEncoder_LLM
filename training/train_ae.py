"""
Autoencoder training module for initial training.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.autoencoder import Seq2SeqAE
from config.config import DEVICE, PAD, LR

def train_autoencoder(train_loader, vocab_size: int, epochs: int):
    """
    Initial training of the autoencoder model.
    
    Args:
        train_loader: DataLoader for training data
        vocab_size: Size of vocabulary
        epochs: Number of training epochs
        
    Returns:
        Trained autoencoder model
    """
    # Initialize model and optimizer
    model = Seq2SeqAE(vocab_size).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    
    # Training loop
    for epoch in tqdm(range(epochs), desc="Training Autoencoder"):
        model.train()
        total_loss = 0
        
        for q_tokens, a_tokens in train_loader:
            # Combine question and answer tokens for training
            x = torch.cat([q_tokens, a_tokens], dim=0).to(DEVICE)
            optimizer.zero_grad()
            
            # Forward pass
            logits, _ = model(x, teacher_forcing=0.5)
            
            # Compute loss
            L = logits[:, 1:].reshape(-1, vocab_size)
            T = x[:, 1:].reshape(-1)
            loss = criterion(L, T)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # Print progress
        if (epoch + 1) % 200 == 0 or epoch == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"[AE] Epoch {epoch + 1}/{epochs}  Loss={avg_loss:.4f}")
            
    return model

def save_autoencoder(model, vocab, inv_vocab, max_len, save_path: str):
    """
    Save the trained autoencoder model and related data.
    
    Args:
        model: Trained autoencoder model
        vocab: Vocabulary dictionary
        inv_vocab: Inverse vocabulary dictionary
        max_len: Maximum sequence length
        save_path: Path to save the model
    """
    import os
    import pickle
    
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Save model weights
    torch.save(model.state_dict(), f"{save_path}/autoencoder.pth")
    
    # Save configuration
    config = {
        'vocab': vocab,
        'inv_vocab': inv_vocab,
        'max_len': max_len,
        'vocab_size': len(vocab)
    }
    
    with open(f"{save_path}/config.pkl", 'wb') as f:
        pickle.dump(config, f)
        
    print(f"✅ Autoencoder saved to {save_path}/") 