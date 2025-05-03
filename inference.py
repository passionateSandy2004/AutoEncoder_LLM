"""
Inference module for question answering.
"""
import argparse
import torch
import pickle
from models.autoencoder import Seq2SeqAE
from models.transition import CompositeTransitionNet
from config.config import DEVICE, SOS, EOS
from utils.tokenizer import Tokenizer

def load_models(model_dir: str):
    """Load trained models and configuration."""
    # Load configuration
    with open(f"{model_dir}/config.pkl", 'rb') as f:
        config = pickle.load(f)
        
    # Initialize models
    ae_model = Seq2SeqAE(config['vocab_size']).to(DEVICE)
    trans_model = CompositeTransitionNet(config['vocab_size']).to(DEVICE)
    
    # Load weights
    ae_model.load_state_dict(torch.load(f"{model_dir}/autoencoder.pth"))
    trans_model.load_state_dict(torch.load(f"{model_dir}/transition.pth"))
    
    return ae_model, trans_model, config

def answer_question(question: str, ae_model, trans_model, tokenizer, max_len: int, inv_vocab: dict):
    """Generate answer for a given question."""
    ae_model.eval()
    trans_model.eval()
    
    with torch.no_grad():
        # Tokenize and encode question
        tokens = tokenizer.tokenize(question, max_len).unsqueeze(0).to(DEVICE)
        zq = ae_model.fc_enc(ae_model.enc(tokens))
        
        # Get predicted answer latent and bias
        z2hat, bias_hat = trans_model(zq)
        
        # Initialize decoder
        hid = torch.tanh(ae_model.fc_dec(z2hat)).unsqueeze(0)
        inp = torch.tensor([SOS], device=DEVICE)
        
        # Generate answer
        words = []
        for _ in range(max_len):
            logits, hid = ae_model.dec(inp, hid)
            logits = logits + bias_hat
            tok = logits.argmax(-1).item()
            if tok == EOS:
                break
            words.append(inv_vocab[tok])
            inp = torch.tensor([tok], device=DEVICE)
            
        return " ".join(words)

def main():
    parser = argparse.ArgumentParser(description='Generate answers using trained model')
    parser.add_argument('--model-dir', type=str, default='saved_model',
                      help='Directory containing trained models')
    parser.add_argument('--question', type=str, required=True,
                      help='Question to answer')
    args = parser.parse_args()
    
    # Load models and configuration
    print("Loading models...")
    ae_model, trans_model, config = load_models(args.model_dir)
    
    # Create tokenizer
    tokenizer = Tokenizer(config['vocab'])
    
    # Generate answer
    answer = answer_question(
        args.question,
        ae_model,
        trans_model,
        tokenizer,
        config['max_len'],
        config['inv_vocab']
    )
    
    print("\nQuestion:", args.question)
    print("Answer:", answer)

if __name__ == '__main__':
    main() 