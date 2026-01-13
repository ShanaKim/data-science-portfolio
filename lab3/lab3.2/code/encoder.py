import torch
import torch.nn as nn
import torch.nn.functional as F

# --- BERT-STYLE MODEL ---
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        '''
        Multi-head self-attention
        Args:
            hidden_size: Hidden size of the model
            num_heads: Number of attention heads
        '''
        super().__init__()
        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Linear transformations for Q, K, V
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        # Final linear layer
        self.fc_out = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, mask=None):
        '''
        Forward pass for multi-head self-attention
        Args:
            x: Input (batch_size, seq_len, hidden_size)
            mask: Attention mask (batch_size, seq_len, seq_len)
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
            Attention weights tensor of shape (batch_size, num_heads, seq_len, seq_len)
        '''
        
        batch_size, seq_len, _ = x.shape
        
        # Project inputs to Q, K, V
        Q = self.query(x)  # (batch_size, seq_len, hidden_size)
        K = self.key(x)    # (batch_size, seq_len, hidden_size)
        V = self.value(x)  # (batch_size, seq_len, hidden_size)
        
        # Reshape to separate heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Shapes: (batch_size, num_heads, seq_len, head_dim)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.head_dim**0.5
        # scores shape: (batch_size, num_heads, seq_len, seq_len)
        
        # Apply mask if provided
        # if mask is not None:
        #     scores = scores.masked_fill(mask == 0, float('-inf'))

        # Apply pad-mask: mask is True at pad positions
        if mask is not None:
           scores = scores.masked_fill(mask, float('-inf'))
        
        # Softmax to get attention weights (pads now have zero weight)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        # output shape: (batch_size, num_heads, seq_len, head_dim)

        # Concatenate heads and apply final linear layer
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.fc_out(output)
        
        return output, attention_weights

class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout=0.1):
        '''
        TODO: Implement feed-forward network
        Args:
            hidden_size: Hidden size of the model
            intermediate_size: Intermediate size of the model
        '''
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Typically BERT uses GELU activation
        self.activation = F.gelu

    def forward(self, x):
        '''
        Forward pass for feed-forward network
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        '''
        # First linear layer + activation
        x = self.linear1(x)
        x = self.activation(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Second linear layer
        x = self.linear2(x)
        
        return x

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(hidden_size, num_heads)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ffn = FeedForward(hidden_size, intermediate_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        '''
        Forward pass for transformer block
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            mask: Optional attention mask of shape (batch_size, seq_len, seq_len)
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        '''
        # Self-attention part with residual connection
        attn_output, attn_weights = self.attn(x, mask)
        attn_output = self.dropout(attn_output)
        x = x + attn_output  # Residual connection
        x = self.ln1(x)      # Layer normalization
        
        # Feed-forward part with residual connection
        ffn_output = self.ffn(x)
        ffn_output = self.dropout(ffn_output)
        x = x + ffn_output  # Residual connection
        x = self.ln2(x)     # Layer normalization
        
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, num_heads=4, num_layers=4,
                 intermediate_size=512, max_len=512, dropout=0.1):
        '''
        Encoder
        Args:
            vocab_size: Vocabulary size
            hidden_size: Hidden size of the model
            num_heads: Number of attention heads
            num_layers: Number of layers
            intermediate_size: Intermediate size of the model
            max_len: Maximum length of the input
        '''
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(max_len, hidden_size)
        self.type_emb = nn.Embedding(2, hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, intermediate_size)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.mlm_head = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, token_type_ids, attention_mask):
        '''
        Forward pass for encoder
        Args:
            input_ids: Token ids [batch_size, seq_len]
            token_type_ids: Segment ids [batch_size, seq_len] (optional)
            attention_mask: Attention mask [batch_size, seq_len] (optional)
        Returns:
            logits: MLM prediction logits [batch_size, seq_len, vocab_size]
            hidden_states: All layer outputs (optional)
        '''
        batch_size, seq_len = input_ids.shape
        
        # Create position ids [0, 1, 2, ..., seq_len-1]
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Default token type ids to 0 if not provided
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        # Get embeddings
        token_embeddings = self.token_emb(input_ids)
        position_embeddings = self.pos_emb(position_ids)
        type_embeddings = self.type_emb(token_type_ids)
        
        # Combine embeddings
        embeddings = token_embeddings + position_embeddings + type_embeddings
        embeddings = self.dropout(embeddings)
        
        # # Prepare attention mask (convert to [batch_size, 1, 1, seq_len] for transformer)
        # if attention_mask is not None:
        #     attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        #     attention_mask = (1.0 - attention_mask) * -10000.0
        
        # # Pass through transformer layers
        # hidden_states = embeddings
        # for layer in self.layers:
        #     hidden_states = layer(hidden_states, attention_mask)

        if attention_mask is not None:
    # attention_mask: 1 for real, 0 for pad
            pad_mask = attention_mask.eq(0)            # True where pad
            pad_mask = pad_mask.unsqueeze(1).unsqueeze(2)
        else:
            pad_mask = None

        # Then pass pad_mask into each layer:
        hidden_states = embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states, pad_mask)


        
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        # MLM head
        logits = self.mlm_head(hidden_states)
        
        return logits, hidden_states
