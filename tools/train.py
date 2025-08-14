# train.py - Complete training script

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Adam, criterion: torch.nn.CrossEntropyLoss, device: torch.device, teacher_forcing_ratio:float=0.5,name:str =''):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    print(name)
    progress_bar = tqdm(dataloader, desc='Training')
    for batch in progress_bar:
        src, tgt, src_lengths, _ = batch
        src: torch.Tensor = src.to(device)
        tgt: torch.Tensor = tgt.to(device)
        # print("source",src,"target", tgt)
        # break
        # print(src_lengths,tgt_lengths)
        optimizer.zero_grad()
        
        if name=='rnn' or name=='lstm':  # Seq2Seq model
            # Remove <eos> from target for input
            output: torch.Tensor = model(src, src_lengths, tgt[:, :-1], teacher_forcing_ratio) #have to modify this 
            
            # Reshape for loss calculation
            output = output.reshape(-1, output.shape[-1])
            tgt = tgt[:, 1:].reshape(-1)  # Remove <sos> from target
            
        else:  # Transformer model
            
            # print("src",src,"tgt",tgt)
            output = model(src, tgt[:, :-1])  # [<sos>, w1, w2, w3] - remove <eos>
            
            # Reshape for loss calculation
            output = output.reshape(-1, output.shape[-1])
          
            tgt = tgt[:, 1:].reshape(-1)
            
       

        loss: torch.Tensor = criterion(output, tgt)
        # print(loss)
        loss.backward()
        
        # Gradient clipping

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        # print(model.parameters())
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    # print(total_loss)
    return total_loss / len(dataloader)

def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.CrossEntropyLoss, device: torch.device,name:str):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            src, tgt, src_lengths, _ = batch
            src: torch.Tensor = src.to(device)
            tgt: torch.Tensor = tgt.to(device)
            
            if name=='rnn' or name=='lstm':  # Seq2Seq model
                output: torch.Tensor = model(src, src_lengths, tgt[:, :-1], teacher_forcing_ratio=0)
                output = output.reshape(-1, output.shape[-1])
                tgt = tgt[:, 1:].reshape(-1)
            else:  # Transformer model
                # tgt_mask = model.generate_square_subsequent_mask(tgt.size(1) - 1).to(device)
                output = model(src, tgt[:, :-1])
                output = output.reshape(-1, output.shape[-1])
                # print(output)
                tgt = tgt[:, 1:].reshape(-1)
                # tokens = torch.argmax(torch.nn.functional.log_softmax(output,dim=-1), dim=-1)
                # print(tokens)
                # words = [tgt_vocab.idx2word[idx.item()] for idx in tokens]
                # print(words)
            loss: torch.Tensor = criterion(output, tgt)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def get_noam_scheduler(optimizer: optim.Adam, d_model: int, warmup_steps: int):
    """
    Returns a LambdaLR scheduler with the 'Noam' schedule:
       lr = base_lr * d_model^{-0.5} * min(step^{-0.5}, step * warmup^{-1.5})
    You should set optimizer's base_lr = 1 and let the scheduler handle the scaling.
    """
    def lr_lambda(step):
        if step == 0:
            return 0.0
        scale = (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)
        return scale

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, num_epochs:int=30, 
                learning_rate:float=0.001, device='cuda', model_name='model'):
    """Full training loop"""
    
    # Create directory for saving models
    os.makedirs('checkpoints', exist_ok=True)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # note lr=1.0 here
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device,name=model_name)
        
        # Evaluate
        val_loss = evaluate(model, val_loader, criterion, device,model_name)
        
        # Store losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Step scheduler
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, f'checkpoints/best_{model_name}.pt')
            print(f"Saved best model with val loss: {val_loss:.4f}")
        
        # Early stopping check
        if epoch > 10 and val_losses[-1] > val_losses[-2] > val_losses[-3]:
            print("Early stopping triggered")
            break
        debug_model_output(model, test_loader, src_vocab, tgt_vocab, device,model_name)
    return train_losses, val_losses

def load_model(model: nn.Module, checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def plot_losses(train_losses, val_losses, title="Training Curves", save_dir="plots"):
    """
    Plot and save training and validation loss curves

    Args:
        train_losses (list): List of training losses per epoch
        val_losses (list): List of validation losses per epoch
        title (str): Title for the plot
        save_dir (str): Directory to save the plot
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Create the plot
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Add some styling
    plt.tight_layout()

    # Save the plot
    # Extract model name from title for filename
    model_name = title.split()[0].lower()
    filename = f"{model_name}_training_curves.png"
    filepath = os.path.join(save_dir, filename)

    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to: {filepath}")

    # Optionally display the plot
    plt.show()

    # Close the figure to free memory
    plt.close()

# Main training script
if __name__ == "__main__":
    import sys
    import os

    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    from data_utils import *
    from problem_models.transformer_model_p3 import Transformer #p3 or p4
    from problem_models.baseline_models_p1 import create_rnn_model, create_lstm_model
    #from problem_models.baseline_models_p2 import create_rnn_model, create_lstm_model
    from evaluate import calculate_scores, debug_model_output
    import numpy as np 
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    src_sentences, tgt_sentences = load_data('data/sentence_pairs_large.tsv'
                                             ,max_len=20) #max_len tunes how long each example is 
    # print(src_sentences)
    print("Building vocabularies...")
    src_vocab = Vocabulary()
    # print("src_vocab",src_vocab)
    tgt_vocab = Vocabulary()
    src_vocab.build_vocab(src_sentences, min_freq=1)
    tgt_vocab.build_vocab(tgt_sentences, min_freq=1)
    print(f"Source vocab size: {len(src_vocab)}")
    print(f"Target vocab size: {len(tgt_vocab)}") 


    sample_size =len(src_sentences)//5 #tune this divisor to use less data & make training faster 
    indices = np.random.choice(len(src_sentences), size=sample_size, replace=False)
    
    src_sentences = [src_sentences[i] for i in indices]
    tgt_sentences = [tgt_sentences[i] for i in indices]
    # print(type(src_sentences))
    # print(src_sentences)
    print(f"Loaded {len(src_sentences)} sentence pairs")
    
    # Split data
    train_size = int(0.8 * len(src_sentences))
    
    val_size = int(0.1 * len(src_sentences))
    
    train_src = src_sentences[:train_size]
    train_tgt = tgt_sentences[:train_size]
    val_src = src_sentences[train_size:train_size+val_size]
    val_tgt = tgt_sentences[train_size:train_size+val_size]
    test_src = src_sentences[train_size+val_size:]
    test_tgt = tgt_sentences[train_size+val_size:]
    
    # print("train",train_tgt)
   
    
    # Create datasets
    train_dataset = TranslationDataset(train_src, train_tgt, src_vocab, tgt_vocab)
    val_dataset = TranslationDataset(val_src, val_tgt, src_vocab, tgt_vocab)
    test_dataset = TranslationDataset(test_src, test_tgt, src_vocab, tgt_vocab)
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                           shuffle=False, collate_fn=collate_fn)
    
    # Model selection based on command line argument
    if len(sys.argv) > 1:
        model_type = sys.argv[1].lower()
    else:
        model_type = 'transformer'
    
    print(f"\nTraining {model_type} model...")
    learning_rate=None
    if model_type == 'transformer':
        model = Transformer(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            d_model=256,     #experiment with these! be mindful of training time tradeoffs 
            num_heads=4,
            num_layers=2,
            d_ff=512
        ).to(device)
        learning_rate = 2e-4
    elif model_type == 'rnn':
        model = create_rnn_model(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            embed_size=256,
            hidden_size=512,
            num_layers=2
        ).to(device)
        learning_rate=0.001
    elif model_type == 'lstm':
        model = create_lstm_model(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            embed_size=256,
            hidden_size=512,
            num_layers=2
        ).to(device)
        learning_rate=0.001
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    #load model
    # model = load_model(model,"./checkpoints/best_{model_name}.pt")
    # Train model
    train_losses, val_losses = train_model(
        model, train_loader, val_loader,
        test_loader,
        num_epochs=5,
        learning_rate=learning_rate,      
        device=device,
        model_name=model_type
    )
    
    # Plot losses
    plot_losses(train_losses, val_losses, f"{model_type.upper()} Training Curves")
     

    debug_model_output(model, test_loader, src_vocab, tgt_vocab, device,model_type)

    # print(f"\nCalculating BLEU score for {model_type}...")
    bleu, bert_f1 = calculate_scores(model, test_loader, src_vocab, tgt_vocab, device,model_type) 
    print(f"{model_type} BLEU Score: {bleu:.2f}")
    print(f"{model_type} BERT F1 Score: {bert_f1}")
    # print("\nTraining complete!")