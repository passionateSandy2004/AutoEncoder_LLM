"""
Text tokenization utilities.
"""
import torch
from typing import List, Dict

class Tokenizer:
    """Tokenizes text into sequences of integers."""
    def __init__(self, vocab: Dict[str, int]):
        self.vocab = vocab
        self.unk_token = '<unk>'
        self.pad_token = '<pad>'
        self.sos_token = '<sos>'
        self.eos_token = '<eos>'

    def tokenize(self, text: str, max_len: int) -> torch.Tensor:
        """
        Tokenize a text sequence.
        
        Args:
            text: Input text to tokenize
            max_len: Maximum sequence length
            
        Returns:
            Tensor of token indices
        """
        from config.config import PAD, SOS, EOS
        
        # Split text into words and convert to token indices
        words = text.split()
        tokens = [SOS] + [self.vocab.get(w, self.vocab.get(self.unk_token, PAD)) 
                         for w in words] + [EOS]
        
        # Pad or truncate to max_len
        if len(tokens) < max_len:
            tokens += [PAD] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]
            tokens[-1] = EOS
            
        return torch.tensor(tokens, dtype=torch.long)

    def detokenize(self, tokens: torch.Tensor, inv_vocab: Dict[int, str]) -> str:
        """
        Convert token indices back to text.
        
        Args:
            tokens: Tensor of token indices
            inv_vocab: Inverse vocabulary mapping
            
        Returns:
            Decoded text string
        """
        from config.config import PAD, SOS, EOS
        
        words = []
        for tok in tokens:
            if tok.item() == EOS:
                break
            if tok.item() not in [PAD, SOS]:
                words.append(inv_vocab[tok.item()])
        return ' '.join(words)

def get_max_sequence_length(qa_pairs: List[tuple]) -> int:
    """
    Calculate maximum sequence length from the dataset.
    
    Args:
        qa_pairs: List of (question, answer) tuples
        
    Returns:
        Maximum sequence length
    """
    max_len = 0
    for q, a in qa_pairs:
        q_len = len(q.split())
        a_len = len(a.split())
        max_len = max(max_len, q_len, a_len)
    return max_len + 2  # +2 for SOS and EOS tokens 