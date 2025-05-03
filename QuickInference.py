import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

# === 0) Device ===
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", DEVICE)

# === 1) Load vocab & config ===
with open('saved_model2FineTune/config.pkl', 'rb') as f:
    cfg = pickle.load(f)

vocab      = cfg['vocab']
inv_vocab  = cfg['inv_vocab']
max_len    = cfg['max_len']
PAD, SOS, EOS = cfg['PAD'], cfg['SOS'], cfg['EOS']
EMB_DIM    = cfg['EMB_DIM']
HID_DIM    = cfg['HID_DIM']
LATENT_DIM = cfg['LATENT_DIM']
VOCAB_SIZE = cfg['VOCAB_SIZE']

# === 2) Tokenizer helper ===
def tokenize_seq(txt, max_len):
    toks = [SOS] + [vocab.get(w, PAD) for w in txt.split()] + [EOS]
    if len(toks) < max_len:
        toks += [PAD] * (max_len - len(toks))
    else:
        toks = toks[:max_len]
        toks[-1] = EOS
    return torch.tensor(toks, dtype=torch.long)

# === 3) Re-define model classes ===
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, EMB_DIM, padding_idx=PAD)
        self.gru = nn.GRU(EMB_DIM, HID_DIM, batch_first=True)
    def forward(self, x):
        _, h = self.gru(self.emb(x))
        return h.squeeze(0)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, EMB_DIM, padding_idx=PAD)
        self.gru = nn.GRU(EMB_DIM, HID_DIM, batch_first=True)
        self.fc  = nn.Linear(HID_DIM, VOCAB_SIZE)
    def forward(self, tok, h):
        e = self.emb(tok).unsqueeze(1)
        out, h2 = self.gru(e, h)
        return self.fc(out.squeeze(1)), h2

class Seq2SeqAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc    = Encoder()
        self.dec    = Decoder()
        self.fc_enc = nn.Linear(HID_DIM, LATENT_DIM)
        self.fc_dec = nn.Linear(LATENT_DIM, HID_DIM)
    def forward(self, x, teacher_forcing=0.0):
        B,T = x.size()
        h    = self.enc(x)
        z    = self.fc_enc(h)
        hid0 = torch.tanh(self.fc_dec(z)).unsqueeze(0)
        outputs = torch.zeros(B, T, VOCAB_SIZE, device=x.device)
        inp     = x[:,0]
        for t in range(1, T):
            logits, hid0 = self.dec(inp, hid0)
            outputs[:,t] = logits
            inp = logits.argmax(1)
        return outputs, z

# Transition net components
class SparseMoE(nn.Module):
    def __init__(self, latent_dim, n_experts=4, hidden=128, k=2):
        super().__init__()
        self.n_experts = n_experts; self.k = k
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(latent_dim, hidden), nn.ReLU(),
                          nn.Linear(hidden, latent_dim))
            for _ in range(n_experts)
        ])
        self.gate = nn.Linear(latent_dim, n_experts)
    def forward(self, z):
        w = F.softmax(self.gate(z), dim=-1)
        topw, idx = w.topk(self.k, dim=-1)
        out = torch.zeros_like(z)
        for b in range(z.size(0)):
            for wi, ei in zip(topw[b], idx[b]):
                out[b] += wi * self.experts[ei](z[b:b+1]).squeeze(0)
        return out

class KeyValueMemory(nn.Module):
    def __init__(self, latent_dim, K=256):
        super().__init__()
        self.keys   = nn.Parameter(torch.randn(K, latent_dim))
        self.values = nn.Parameter(torch.randn(K, latent_dim))
    def forward(self, z):
        scores = torch.matmul(z, self.keys.t())
        attn   = F.softmax(scores, dim=-1)
        return torch.matmul(attn, self.values)

class GatedMLPBlock(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.fc1  = nn.Linear(latent_dim, hidden_dim)
        self.fc2  = nn.Linear(hidden_dim, latent_dim)
        self.gate = nn.Linear(latent_dim, latent_dim)
    def forward(self, x):
        h = F.gelu(self.fc1(x))
        h = self.fc2(h)
        g = torch.sigmoid(self.gate(x))
        return x + g * h

class HyperAdapter(nn.Module):
    def __init__(self, latent_dim, decoder_out_dim):
        super().__init__()
        self.hyper = nn.Linear(latent_dim, decoder_out_dim)
    def forward(self, z):
        return self.hyper(z)

class CompositeTransitionNet(nn.Module):
    def __init__(self, latent_dim, hidden=128, n_experts=4, K=256):
        super().__init__()
        self.moe     = SparseMoE(latent_dim, n_experts, hidden)
        self.mem     = KeyValueMemory(latent_dim, K)
        self.gmlp1   = GatedMLPBlock(latent_dim, hidden)
        self.gmlp2   = GatedMLPBlock(latent_dim, hidden)
        self.adapter = HyperAdapter(latent_dim, VOCAB_SIZE)
    def forward(self, z):
        x = self.moe(z)
        m = self.mem(z)
        x = x + m
        x = self.gmlp1(x)
        x = self.gmlp2(x)
        return x, self.adapter(x)

# === 4) Instantiate & load weights ===
ae    = Seq2SeqAE().to(DEVICE)
trans = CompositeTransitionNet(LATENT_DIM, hidden=128, n_experts=4, K=256).to(DEVICE)

ae.load_state_dict(   torch.load('saved_model2FineTune/autoencoder.pth',    map_location=DEVICE) )
trans.load_state_dict(torch.load('saved_model2FineTune/composite_trans.pth',  map_location=DEVICE) )

ae.eval()
trans.eval()

print("✅ Models loaded on", DEVICE)

# === 5) Inference helper ===
def answer_question(q):
    with torch.no_grad():
        tq = tokenize_seq(q, max_len).unsqueeze(0).to(DEVICE)
        zq = ae.fc_enc(ae.enc(tq))
        z2, bias = trans(zq)
        hid = torch.tanh(ae.fc_dec(z2)).unsqueeze(0)
        inp = torch.tensor([SOS], device=DEVICE)
        words=[]
        for _ in range(max_len):
            logits, hid = ae.dec(inp, hid)
            logits = logits + bias
            tok = logits.argmax(-1).item()
            if tok == EOS: break
            words.append(inv_vocab[tok])
            inp = torch.tensor([tok], device=DEVICE)
        return " ".join(words)

# === 6) Put Your Questions Here ===
questions_to_test = [
    "Which prize did Frederick Buechner create?",
    "What institute at Notre Dame studies  the reasons for violent conflict?",
    # from your saved qa_pairs
]
for q in questions_to_test:
    print("Q:", q)
    print("A:", answer_question(q))
    print()
