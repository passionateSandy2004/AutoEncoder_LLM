"""
Sequence-to-sequence autoencoder model.
"""
import torch
import torch.nn as nn
from models.components import Encoder, Decoder
from config.config import HID_DIM, LATENT_DIM

class Seq2SeqAE(nn.Module):
    """Sequence-to-sequence autoencoder with latent space."""
    def __init__(self, vocab_size: int):
        super().__init__()
        self.enc = Encoder(vocab_size)
        self.dec = Decoder(vocab_size)
        self.fc_enc = nn.Linear(HID_DIM, LATENT_DIM)
        self.fc_dec = nn.Linear(LATENT_DIM, HID_DIM)
        
    def forward(self, x, teacher_forcing: float = 1.0):
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input sequence tensor
            teacher_forcing: Probability of using teacher forcing
            
        Returns:
            outputs: Decoder output logits
            z: Latent representation
        """
        B, T = x.size()
        
        # Encode input sequence
        h = self.enc(x)                    # B×HID_DIM
        z = self.fc_enc(h)                 # B×LATENT_DIM
        
        # Initialize decoder hidden state
        hid0 = torch.tanh(self.fc_dec(z)).unsqueeze(0)  # 1×B×HID_DIM
        
        # Generate output sequence
        outputs = torch.zeros(B, T, self.dec.fc.out_features, device=x.device)
        inp = x[:, 0]                      # Always start with <sos>
        
        for t in range(1, T):
            logits, hid0 = self.dec(inp, hid0)
            outputs[:, t] = logits
            
            # Teacher forcing
            if torch.rand(1).item() < teacher_forcing:
                inp = x[:, t]
            else:
                inp = logits.argmax(1)
                
        return outputs, z

def reconstruct_from_latents(z, model, max_len, sos, eos, inv_vocab, device):
    """
    Reconstruct text from latent representations.
    
    Args:
        z: Latent representation tensor
        model: Trained autoencoder model
        max_len: Maximum sequence length
        sos: Start of sequence token
        eos: End of sequence token
        inv_vocab: Inverse vocabulary mapping
        device: Device to use for computation
        
    Returns:
        Reconstructed text string
    """
    # Normalize input shape
    if z.dim() == 1:
        z = z.unsqueeze(0)
    elif z.dim() > 2:
        z = z.view(-1, z.size(-1))
        
    B = z.size(0)
    
    # Initialize decoder
    h0 = torch.tanh(model.fc_dec(z))
    hidden = h0.unsqueeze(0)
    
    # Greedy decoding
    input_tok = torch.full((B,), sos, dtype=torch.long, device=device)
    out_ids = [[] for _ in range(B)]
    
    for _ in range(max_len):
        logits, hidden = model.dec(input_tok, hidden)
        input_tok = logits.argmax(1)
        for i, tok in enumerate(input_tok.tolist()):
            if tok != eos:
                out_ids[i].append(tok)
                
    # Convert to text
    def ids_to_str(ids):
        return " ".join(inv_vocab[i] for i in ids if i in inv_vocab)
        
    return ids_to_str(out_ids[0]) if B == 1 else [ids_to_str(ids) for ids in out_ids] 