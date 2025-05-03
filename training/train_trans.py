"""
Transition network training module for initial training.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from models.transition import CompositeTransitionNet
from config.config import DEVICE, PAD, SOS, LR

def train_transition_network(train_loader, ae_model, vocab_size: int, epochs: int):
    """
    Initial training of the transition network model.
    
    Args:
        train_loader: DataLoader for training data
        ae_model: Trained autoencoder model
        vocab_size: Size of vocabulary
        epochs: Number of training epochs
        
    Returns:
        Trained transition network model
    """
    # Initialize model and optimizer
    trans = CompositeTransitionNet(vocab_size).to(DEVICE)
    optimizer = optim.Adam(trans.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    
    # Freeze autoencoder
    for p in ae_model.parameters():
        p.requires_grad = False
    ae_model.dec.train()
    
    # Training loop
    for epoch in tqdm(range(epochs), desc="Training Transition Network"):
        trans.train()
        total_loss = 0
        
        for q_tokens, a_tokens in train_loader:
            q_tokens = q_tokens.to(DEVICE)
            a_tokens = a_tokens.to(DEVICE)
            
            # Encode question and answer
            with torch.no_grad():
                hq = ae_model.enc(q_tokens)
                zq = ae_model.fc_enc(hq)
                ha = ae_model.enc(a_tokens)
                za = ae_model.fc_enc(ha)
            
            # Forward pass through transition network
            z2hat, bias_hat = trans(zq)
            
            # Initialize decoder hidden state
            hid = torch.tanh(ae_model.fc_dec(z2hat)).unsqueeze(0)
            
            # Decode with injected bias
            inp = torch.full((a_tokens.size(0),), SOS, device=DEVICE)
            seq_loss = 0.0
            count = 0
            
            for t in range(1, a_tokens.size(1)):
                logits, hid = ae_model.dec(inp, hid)
                logits = logits + bias_hat
                
                # Compute loss only for non-padding tokens
                mask = (a_tokens[:, t] != PAD)
                if mask.any():
                    seq_loss += criterion(logits[mask], a_tokens[:, t][mask])
                    count += 1
                    
                inp = a_tokens[:, t]
                
            if count == 0:
                continue
                
            # Backward pass
            loss = seq_loss / count
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(trans.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
        # Print progress
        if (epoch + 1) % 50 == 0 or epoch == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"[T] Epoch {epoch + 1}/{epochs}  Loss={avg_loss:.4f}")
            
    return trans

def save_transition_network(model, save_path: str):
    """
    Save the trained transition network model.
    
    Args:
        model: Trained transition network model
        save_path: Path to save the model
    """
    import os
    
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Save model weights
    torch.save(model.state_dict(), f"{save_path}/transition.pth")
    print(f"✅ Transition network saved to {save_path}/") 