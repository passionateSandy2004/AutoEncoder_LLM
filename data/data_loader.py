"""
Data loading and preprocessing utilities.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

class QADataset(Dataset):
    """Dataset for question-answer pairs."""
    def __init__(self, qa_pairs: List[Tuple[str, str]], tokenizer, max_len: int):
        self.qa_pairs = qa_pairs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        q, a = self.qa_pairs[idx]
        q_tokens = self.tokenizer.tokenize(q, self.max_len)
        a_tokens = self.tokenizer.tokenize(a, self.max_len)
        return q_tokens, a_tokens

def load_qa_pairs(file_path: str) -> List[Tuple[str, str]]:
    """Load question-answer pairs from a file."""
    qa_pairs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            q, a = line.strip().split('\t')
            qa_pairs.append((q, a))
    return qa_pairs

def build_vocab(qa_pairs: List[Tuple[str, str]]) -> Tuple[dict, dict]:
    """Build vocabulary from the dataset."""
    from config.config import PAD, SOS, EOS
    
    vocab = {'<pad>': PAD, '<sos>': SOS, '<eos>': EOS}
    for q, a in qa_pairs:
        for w in q.split() + a.split():
            if w not in vocab:
                vocab[w] = len(vocab)
    inv_vocab = {i: w for w, i in vocab.items()}
    return vocab, inv_vocab

def create_dataloaders(qa_pairs: List[Tuple[str, str]], 
                      tokenizer,
                      max_len: int,
                      batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders."""
    from torch.utils.data import random_split
    
    dataset = QADataset(qa_pairs, tokenizer, max_len)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader 