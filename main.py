"""
Main entry point for the autoencoder-based language model.
"""
import argparse
import torch
import pickle
from data.data_loader import load_qa_pairs, build_vocab, create_dataloaders
from utils.tokenizer import Tokenizer, get_max_sequence_length
from training.train_ae import train_autoencoder, save_autoencoder
from training.train_trans import train_transition_network, save_transition_network
from fnetune import fine_tune_autoencoder, fine_tune_transition_network, save_fine_tuned_models
from models.autoencoder import Seq2SeqAE
from models.transition import CompositeTransitionNet
from config.config import (
    BATCH_SIZE, AE_EPOCHS, TRANS_EPOCHS, DEVICE,
    PAD, SOS, EOS
)

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

def main():
    parser = argparse.ArgumentParser(description='Train or fine-tune autoencoder-based language model')
    parser.add_argument('--data', type=str, default='data.txt',
                      help='Path to the data file')
    parser.add_argument('--train-ae', action='store_true',
                      help='Train the autoencoder')
    parser.add_argument('--train-trans', action='store_true',
                      help='Train the transition network')
    parser.add_argument('--finetune', action='store_true',
                      help='Fine-tune models on new domain data')
    parser.add_argument('--pretrained-dir', type=str,
                      help='Directory containing pretrained models (required for fine-tuning)')
    parser.add_argument('--save-dir', type=str, default='saved_models',
                      help='Directory to save models')
    parser.add_argument('--ae-epochs', type=int, default=AE_EPOCHS,
                      help='Number of epochs for autoencoder training/fine-tuning')
    parser.add_argument('--trans-epochs', type=int, default=TRANS_EPOCHS,
                      help='Number of epochs for transition network training/fine-tuning')
    args = parser.parse_args()
    
    if args.finetune and not args.pretrained_dir:
        parser.error("--pretrained-dir is required for fine-tuning")
    
    # Load and preprocess data
    print("Loading data...")
    qa_pairs = load_qa_pairs(args.data)
    
    if args.train_ae:
        # For initial training, build new vocabulary
        vocab, inv_vocab = build_vocab(qa_pairs)
        max_len = get_max_sequence_length(qa_pairs)
    else:
        # For transition network training or fine-tuning, load existing vocabulary
        try:
            with open(f"{args.save_dir}/config.pkl", 'rb') as f:
                cfg = pickle.load(f)
            vocab = cfg['vocab']
            inv_vocab = cfg['inv_vocab']
            max_len = cfg['max_len']
        except FileNotFoundError:
            print("Error: Could not find saved model configuration. Please train the autoencoder first.")
            return
    
    # Create tokenizer
    tokenizer = Tokenizer(vocab)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        qa_pairs, tokenizer, max_len, BATCH_SIZE
    )
    
    if args.finetune:
        # Fine-tuning mode
        print("\nLoading pretrained models...")
        ae_model, trans_model, cfg = load_pretrained_models(args.pretrained_dir)
        
        print("\nFine-tuning autoencoder...")
        fine_tune_autoencoder(ae_model, train_loader, args.ae_epochs)
        
        print("\nFine-tuning transition network...")
        fine_tune_transition_network(trans_model, ae_model, train_loader, args.trans_epochs)
        
        print("\nSaving fine-tuned models...")
        save_fine_tuned_models(ae_model, trans_model, cfg, args.save_dir)
        
    else:
        # Initial training mode
        if args.train_ae:
            print("\nTraining autoencoder...")
            ae_model = train_autoencoder(train_loader, len(vocab), args.ae_epochs)
            save_autoencoder(ae_model, vocab, inv_vocab, max_len, args.save_dir)
        else:
            # Load pretrained autoencoder
            print("\nLoading pretrained autoencoder...")
            ae_model = Seq2SeqAE(len(vocab)).to(DEVICE)
            ae_model.load_state_dict(torch.load(f"{args.save_dir}/autoencoder.pth"))
            
        if args.train_trans:
            print("\nTraining transition network...")
            trans_model = train_transition_network(
                train_loader, ae_model, len(vocab), args.trans_epochs
            )
            save_transition_network(trans_model, args.save_dir)
        
    print("\nTraining complete!")

if __name__ == '__main__':
    main()
