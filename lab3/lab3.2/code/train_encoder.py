import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt

# --- MLM MASKING ---
# --- MLM MASKING ---
def mask_tokens(input_ids, vocab_size, mask_token_id, pad_token_id, mlm_prob=0.15 ):
    '''
    MLM masking
    Args:
        input_ids: Input IDs tensor of shape [batch_size, seq_len]
        vocab_size: Size of vocabulary
        mask_token_id: ID of [MASK] token
        pad_token_id: ID of [PAD] token
        mlm_prob: Probability of masking tokens (default: 0.15)
    Returns:
        masked_input_ids: Tensor with some tokens replaced by [MASK]/random/original
        labels: Ground truth tokens with -100 for non-masked positions
    '''
    labels = input_ids.clone()
    # Create probability matrix where 1 = mask candidate
    probability_matrix = torch.full(labels.shape, mlm_prob)
    
    # Don't mask padding tokens
    special_tokens_mask = (input_ids == pad_token_id)

    device = input_ids.device
    probability_matrix = torch.full(labels.shape, mlm_prob, device=device)
    
    # Determine which tokens to mask
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # Only compute loss on masked tokens
    
    # 80% of the time: replace with [MASK]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=device)).bool() & masked_indices
    input_ids[indices_replaced] = mask_token_id
    
    # 10% of the time: replace with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5, device=device)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long, device=device)
    input_ids[indices_random] = random_words[indices_random]
    
    # 10% of the time: keep original word (but still predict it)
    
    return input_ids, labels

def train_bert(model, train_dataloader, tokenizer, val_dataloader=None, epochs=3, lr=5e-4, device='cuda', tag=None):
    '''
    Training loop for BERT
    Args:
        model: BERT model (Encoder class)
        dataloader: DataLoader providing (input_ids, token_type_ids, attention_mask) for training
        tokenizer: Tokenizer used for decoding samples
        val_dataloader: DataLoader for validation (optional)
        epochs: Number of training epochs
        lr: Learning rate (default: 5e-5)
        device: Device to run on ('cuda' or 'cpu')

    Returns:
        model: Trained BERT model
    '''
    # Setup training components
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)  # Ignore non-masked tokens
    writer = SummaryWriter()  # TensorBoard logging
    
    # Get token IDs
    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.pad_token_id
    
    train_losses = []
    val_losses = [] if val_dataloader is not None else None

    global_step = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch in progress_bar:
            # Prepare batch
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch.get('token_type_ids', torch.zeros_like(input_ids)).to(device)
            attention_mask = batch.get('attention_mask', torch.ones_like(input_ids)).to(device)
            
            # Apply MLM masking
            masked_inputs, labels = mask_tokens(
                input_ids, 
                vocab_size=tokenizer.vocab_size,
                mask_token_id=mask_token_id,
                pad_token_id=pad_token_id
            )
            labels = labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits, _ = model(masked_inputs, token_type_ids, attention_mask)
            
            # Reshape for loss calculation
            loss = criterion(
                logits.view(-1, tokenizer.vocab_size), 
                labels.view(-1)
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Logging
            epoch_loss += loss.item()
            global_step += 1
            writer.add_scalar('Loss/train', loss.item(), global_step)
            
            # progress_bar.set_postfix({'loss': loss.item()})
        
        # Epoch summary
        avg_train = epoch_loss / len(train_dataloader)
        train_losses.append(avg_train)
        writer.add_scalar('Loss/train_epoch', avg_train, epoch)
        
        #validation
        if val_dataloader is not None:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc=f'Val Epoch {epoch+1}/{epochs}'):
                    input_ids = batch['input_ids'].to(device)
                    token_type_ids = batch.get('token_type_ids', torch.zeros_like(input_ids)).to(device)
                    attention_mask = batch.get('attention_mask', torch.ones_like(input_ids)).to(device)
                    
                    masked_inputs, labels = mask_tokens(
                        input_ids, 
                        vocab_size=tokenizer.vocab_size,
                        mask_token_id=mask_token_id,
                        pad_token_id=pad_token_id
                    )
                    labels = labels.to(device)
                    
                    logits, _ = model(masked_inputs, token_type_ids, attention_mask)
                    
                    loss = criterion(
                        logits.view(-1, tokenizer.vocab_size), 
                        labels.view(-1)
                    )
                    
                    val_loss += loss.item()
            
            avg_val = val_loss / len(val_dataloader)
            val_losses.append(avg_val)
            writer.add_scalar('Loss/val', avg_val, epoch)

            print(f"Epoch {epoch+1} - Train Loss: {avg_train:.4f}, Val Loss: {avg_val:.4f}")
        else:
            print(f"Epoch {epoch+1} - Train Loss: {avg_train:.4f}")            

    
    writer.close()



    ### Plotting Loss Curves ###
    # plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 13,
        'legend.fontsize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
    })

    
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)

    # Plot training loss across all epochs
    train_epochs = range(1, epochs + 1)
    ax.plot(
        train_epochs,
        train_losses,
        label='Train Loss',
        marker='o',
        markersize=5,
        linewidth=2,
    )

    # Plot validation loss only where it exists
    if val_losses is not None:
        val_epochs = range(1, len(val_losses) + 1)
        ax.plot(
            val_epochs,
            val_losses,
            label='Validation Loss',
            marker='s',
            markersize=5,
            linewidth=2,
        )

    # Labels, title, ticks
    ax.set_xlabel('Epoch', labelpad=10)
    ax.set_ylabel('Loss',  labelpad=10)
    # ax.set_title('Training vs. Validation Loss', pad=15)
    ax.set_title(f'Train vs Val Loss\n{tag}', pad=15)
    ax.set_ylim(5, 11)
    ax.set_xticks(train_epochs)

    # Light horizontal grid, clean spines
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    # Legend + layout
    ax.legend(loc='upper right', frameon=False)
    fig.tight_layout()

    # Save high-res figure
    if tag is not None:
        save_path = f"../figure/loss_curve_{tag}.png"
    else:
        save_path = "../figure/loss_curve.png"

    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved loss curve to {save_path}")




    ## save model and tokenizer

    output_dir = "./bert_final_model"
    os.makedirs(output_dir, exist_ok=True) 

    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin")) 

    tokenizer.save_pretrained(output_dir)  

    print(f"Tokenizer saved to '{output_dir}'")
    if val_losses is not None:
        model.final_val_loss = val_losses[-1]
        model.min_val_loss = min(val_losses)
        if len(val_losses) >= 5:
            model.avg_val_loss_last5 = sum(val_losses[-5:]) / 5
        else:
            model.avg_val_loss_last5 = sum(val_losses) / len(val_losses)
    else:
        model.final_val_loss = None
        model.min_val_loss = None
        model.avg_val_loss_last3 = None
        
    return model