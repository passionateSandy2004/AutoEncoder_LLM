"""
Fine-tuning module for domain adaptation.
"""
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.autoencoder import Seq2SeqAE
from models.transition import CompositeTransitionNet
from data.data_loader import load_qa_pairs, build_vocab, create_dataloaders
from utils.tokenizer import Tokenizer

# === HYPERPARAMS ===
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
AE_FINETUNE_EPOCHS = 3000    # epochs for autoencoder fine-tuning
TRANS_FINETUNE_EPOCHS = 3000 # epochs for transition network fine-tuning
LR_AE = 5e-4                 # learning rate for AE fine-tuning
LR_TRANS = 1e-4              # learning rate for transition network fine-tuning
BATCH_SIZE = 16

def load_pretrained_models(model_dir: str):
    """
    Load pretrained models and their configuration.
    
    Args:
        model_dir: Directory containing pretrained models
        
    Returns:
        Tuple of (ae_model, trans_model, config)
    """
    # Load config
    with open(f'{model_dir}/config.pkl', 'rb') as f:
        cfg = pickle.load(f)
    
    # Initialize models
    ae = Seq2SeqAE(cfg['vocab_size']).to(DEVICE)
    trans = CompositeTransitionNet(cfg['vocab_size']).to(DEVICE)
    
    # Load weights
    ae.load_state_dict(torch.load(f'{model_dir}/autoencoder.pth', map_location=DEVICE))
    trans.load_state_dict(torch.load(f'{model_dir}/transition.pth', map_location=DEVICE))
    
    return ae, trans, cfg

def fine_tune_autoencoder(ae_model, train_loader, epochs: int):
    """
    Fine-tune the autoencoder on new domain data.
    
    Args:
        ae_model: Pretrained autoencoder model
        train_loader: DataLoader for new domain data
        epochs: Number of fine-tuning epochs
    """
    optimizer = optim.Adam(ae_model.parameters(), lr=LR_AE)
    criterion = nn.CrossEntropyLoss(ignore_index=cfg['PAD'])
    
    for epoch in tqdm(range(epochs), desc="Fine-tuning Autoencoder"):
        ae_model.train()
        total_loss = 0
        
        for q_tokens, a_tokens in train_loader:
            q_tokens = q_tokens.to(DEVICE)
            a_tokens = a_tokens.to(DEVICE)
            
            # Forward pass
            logits, _ = ae_model(q_tokens, teacher_forcing=0.5)
            
            # Compute loss
            L = logits[:, 1:].reshape(-1, cfg['vocab_size'])
            T = q_tokens[:, 1:].reshape(-1)
            loss = criterion(L, T)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch + 1) % 100 == 0 or epoch == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"[AE FT] Epoch {epoch + 1}/{epochs}  Loss={avg_loss:.4f}")

def fine_tune_transition_network(trans_model, ae_model, train_loader, epochs: int):
    """
    Fine-tune the transition network on new domain data.
    
    Args:
        trans_model: Pretrained transition network model
        ae_model: Fine-tuned autoencoder model
        train_loader: DataLoader for new domain data
        epochs: Number of fine-tuning epochs
    """
    optimizer = optim.Adam(trans_model.parameters(), lr=LR_TRANS)
    criterion = nn.CrossEntropyLoss(ignore_index=cfg['PAD'])
    
    # Freeze autoencoder
    for p in ae_model.parameters():
        p.requires_grad = False
    ae_model.dec.train()
    
    for epoch in tqdm(range(epochs), desc="Fine-tuning Transition Network"):
        trans_model.train()
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
            z2hat, bias_hat = trans_model(zq)
            
            # Initialize decoder hidden state
            hid = torch.tanh(ae_model.fc_dec(z2hat)).unsqueeze(0)
            
            # Decode with injected bias
            inp = torch.full((a_tokens.size(0),), cfg['SOS'], device=DEVICE)
            seq_loss = 0.0
            count = 0
            
            for t in range(1, a_tokens.size(1)):
                logits, hid = ae_model.dec(inp, hid)
                logits = logits + bias_hat
                
                # Compute loss only for non-padding tokens
                mask = (a_tokens[:, t] != cfg['PAD'])
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
            torch.nn.utils.clip_grad_norm_(trans_model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch + 1) % 50 == 0 or epoch == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"[T FT] Epoch {epoch + 1}/{epochs}  Loss={avg_loss:.4f}")

def save_fine_tuned_models(ae_model, trans_model, cfg, save_dir: str):
    """
    Save fine-tuned models and configuration.
    
    Args:
        ae_model: Fine-tuned autoencoder model
        trans_model: Fine-tuned transition network model
        cfg: Configuration dictionary
        save_dir: Directory to save models
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save models
    torch.save(ae_model.state_dict(), f'{save_dir}/autoencoder.pth')
    torch.save(trans_model.state_dict(), f'{save_dir}/transition.pth')
    
    # Save config
    with open(f'{save_dir}/config.pkl', 'wb') as f:
        pickle.dump(cfg, f)
    
    print(f"✅ Fine-tuned models saved to {save_dir}/")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Fine-tune models on new domain data')
    parser.add_argument('--pretrained-dir', type=str, required=True,
                      help='Directory containing pretrained models')
    parser.add_argument('--data', type=str, required=True,
                      help='Path to new domain data file')
    parser.add_argument('--save-dir', type=str, required=True,
                      help='Directory to save fine-tuned models')
    args = parser.parse_args()
    
    # Load pretrained models
    print("Loading pretrained models...")
    ae_model, trans_model, cfg = load_pretrained_models(args.pretrained_dir)
    
    # Load and preprocess new domain data
    print("Loading new domain data...")
    qa_pairs = load_qa_pairs(args.data)
    vocab, inv_vocab = build_vocab(qa_pairs)
    tokenizer = Tokenizer(vocab)
    train_loader, _ = create_dataloaders(qa_pairs, tokenizer, cfg['max_len'], BATCH_SIZE)
    
    # Fine-tune autoencoder
    print("\nFine-tuning autoencoder...")
    fine_tune_autoencoder(ae_model, train_loader, AE_FINETUNE_EPOCHS)
    
    # Fine-tune transition network
    print("\nFine-tuning transition network...")
    fine_tune_transition_network(trans_model, ae_model, train_loader, TRANS_FINETUNE_EPOCHS)
    
    # Save fine-tuned models
    print("\nSaving fine-tuned models...")
    save_fine_tuned_models(ae_model, trans_model, cfg, args.save_dir)
    
    print("\nFine-tuning complete!")

if __name__ == '__main__':
    main()
