import numpy as np
from .tensor import tensor, tensor_from_numpy, Tensor
from .module import Module, Parameter
from .modules_basic import (
    Embedding,
    Dropout,
    LayerNorm1d,
    Linear
)
from .tensor_ops import TensorBackend
from .nn import (
    max,
    softmax,
    dropout,
    GELU,
)
from typing import Any, Dict, Optional, Sequence, Tuple

datatype = np.float32


class MultiHeadAttention(Module):
    def __init__(self, n_embd: int, n_head: int, causal: bool=True, p_dropout: float=0.1, bias: bool=True, backend: TensorBackend=None):
        super().__init__()
        """Implements Multi-Head Attention as described in "Attention Is All You Need"

        Args:
            n_embd: Dimensionality of embeddings and hidden states
            n_head: Number of heads
            p_dropout: Dropout ratio for dropout layer
            causal: If True, then apply a causal mask during self-attention
            bias: If True, then apply a bias in Linear layers
        
        Attributes:
            q_projection: Linear layer projecting input to Q matrix
            k_projection: Linear layer projecting input to K matrix
            v_projection: Linear layer projecting input to V matrix
            out_projection: Linear output projection layer
            dropout: Dropout layer
        """
        self.backend = backend
        self.n_embd = n_embd 
        self.n_head = n_head
        self.causal = causal
        self.attn_hidden_dim = n_embd // n_head

        ### BEGIN ASSIGN3_3
        self.q_projection = Linear(self.n_embd, self.n_embd, bias=bias, backend=backend)
        self.k_projection = Linear(self.n_embd, self.n_embd, bias=bias, backend=backend) 
        self.v_projection =  Linear(self.n_embd, self.n_embd, bias=bias, backend=backend) 
        self.out_projection = Linear(self.n_embd, self.n_embd, bias=bias, backend=backend) 
        self.dropout = Dropout(p_dropout) 
        ### END ASSIGN3_3
        
    def _transpose(self, a: Tensor, x, y) -> Tensor:
        order = list(range(a.dims))
        order[y], order[x] = order[x], order[y]
        return a._new(a._tensor.permute(*order))

    def create_causal_mask(self, seq_len):
        """
        Create a causal mask for self-attention to prevent information leakage.
        
        Generates a triangular mask where each position can only attend to previous
        positions and itself. Upper triangle contains -inf, lower triangle contains 0.

        Args:
            seq_len (int): Length of the sequence

        Returns:
            Tensor: Causal mask of shape (1, 1, seq_len, seq_len) with -inf above
                    diagonal and 0 on/below diagonal. Will be broadcasted to full
                    attention tensor shape during computation.
        """
        # Returns a 1x1xTxt triangular causal mask for Q @ K^T (You will implicitly broadcast it to BxHxTxT)
        mask = -np.finfo(datatype).max * np.triu(np.ones((1, 1, seq_len, seq_len), dtype=datatype), 1)
        return tensor_from_numpy(mask, backend=self.backend)

    def project_to_query_key_value(self, x):
        """
        Project input embeddings to Query, Key, and Value matrices for self-attention.
        
        Args:
            x (Tensor): Input embeddings of shape (batch_size, seq_len, n_embd)

        Returns:
            tuple: (q, kT, v) where:
                - q: Query matrix of shape (batch_size, num_heads, seq_len, attn_hidden_dim)
                - kT: Transposed key matrix of shape (batch_size, num_heads, attn_hidden_dim, seq_len)
                - v: Value matrix of shape (batch_size, num_heads, seq_len, attn_hidden_dim)
        """
        batch_size, seq_len, n_embd = x.shape
        ### BEGIN ASSIGN3_3
        q0 = self.q_projection(x).view(batch_size, seq_len, self.n_head, self.attn_hidden_dim)
        q0.requires_grad_(True)
        q1 = q0.contiguous()
        q1.requires_grad_(True)
        q = self._transpose(q1, 1, 2)

        k = self._transpose(self.k_projection(x).view(batch_size, seq_len, self.n_head, self.attn_hidden_dim).contiguous(), 1, 2)
        q = self._transpose(self.q_projection(x).view(batch_size, seq_len, self.n_head, self.attn_hidden_dim).contiguous(), 1, 2)
        v = self._transpose(self.v_projection(x).view(batch_size, seq_len, self.n_head, self.attn_hidden_dim).contiguous(), 1, 2)
        kT = self._transpose(k, -2, -1)
        
        kT.requires_grad_(True)
        q.requires_grad_(True)
        v.requires_grad_(True)
        ### END ASSIGN3_3
        return q, kT, v
    
    def self_attention(self, q, kT, v):
        """
        Compute self-attention: softmax((q @ kT) / sqrt(attn_hidden_dim)) @ v.
        
        Args:
            q (Tensor): Query matrix of shape (batch_size, num_heads, seq_len, attn_hidden_dim)
            kT (Tensor): Transposed key matrix of shape (batch_size, num_heads, attn_hidden_dim, seq_len)
            v (Tensor): Value matrix of shape (batch_size, num_heads, seq_len, attn_hidden_dim)

        Returns:
            Tensor: Attention output of shape (batch_size, seq_len, n_embd)
        """
        batch_size, num_head, queries_len, q_dim = q.shape
        _, _, k_dim, _ = kT.shape
        _, _, _, v_dim = v.shape
        assert q_dim == k_dim == v_dim
        result = None
        
        ### BEGIN ASSIGN3_3
        attn = (q @ kT ) / np.sqrt(self.attn_hidden_dim)
        if self.causal:
           attn = attn + self.create_causal_mask(queries_len)
        attn = softmax(attn, dim=3)
        attn = self.dropout(attn)
        result = attn @ v
        ### END ASSIGN3_3
        
        return result

    def forward(self, x):
        """
        Compute multi-head attention with optional causal masking.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, n_embd)

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, n_embd)
        """
        batch_size, seq_len, n_embd = x.shape
        ### BEGIN ASSIGN3_3
        q, kT, v = self.project_to_query_key_value(x)
        attn = self.self_attention(q, kT, v)
        
        attn.requires_grad_(True)
        attn = self._transpose(attn, 1, 2)
        attn = attn.contiguous()
        
        attn = attn.view(batch_size, seq_len, n_embd)
        result = attn.contiguous()
        result.requires_grad_(True)
        return self.out_projection(result) 
        ### END ASSIGN3_3


class FeedForward(Module):
    def __init__(self, n_embd: int, middle_dim: int=256, p_dropout: float=0.1, bias: bool=True, backend: TensorBackend=None):
        super().__init__()
        """
        Initialize a feed-forward network module.
        
        Args:
            n_embd (int): Input and output dimension
            middle_dim (int): Hidden layer dimension, default 256
            p_dropout (float): Dropout probability, default 0.1
            bias (bool): Whether to use bias in linear layers, default True
            backend (TensorBackend): Backend for tensor operations
            
        Attributes:
            linear_in (Linear): First linear layer
            linear_out (Linear): Second linear layer
            dropout (Dropout): Dropout layer
        """
        ### BEGIN ASSIGN3_3
        self.linear_in =  Linear(n_embd, middle_dim, bias=bias, backend=backend)
        self.linear_out =  Linear(middle_dim, n_embd, bias=bias, backend=backend)
        self.dropout = Dropout(p_dropout)
        ### END ASSIGN3_3

    def forward(self, x):
        """
        Forward pass through feed-forward network with GELU activation and dropout.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, n_embd)

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, n_embd)
        """
        batch_size, seq_len, n_embd = x.shape

        ### BEGIN ASSIGN3_3
        x = self.linear_in(x)
        x = GELU(x)
        x = self.linear_out(x)
        x = self.dropout(x)
        
        assert x.shape[0] == batch_size and x.shape[1] == seq_len and x.shape[2] == n_embd
        ### END ASSIGN3_3

        return x
    

class TransformerLayer(Module):
    def __init__(self, n_embd: int, n_head: int, p_dropout: float=0.1, ln_eps: float=1e-5, bias: bool=True, backend: TensorBackend=None):
        super().__init__()
        """
        Initialize a transformer layer with pre-layer normalization.
        
        Args:
            n_embd (int): Embedding dimension
            n_head (int): Number of attention heads
            p_dropout (float): Dropout probability, default 0.1
            ln_eps (float): Layer normalization epsilon, default 1e-5
            bias (bool): Whether to use bias in linear layers, default True
            backend (TensorBackend): Backend for tensor operations
            
        Attributes:
            ln_1 (LayerNorm1d): First layer normalization before attention
            ln_2 (LayerNorm1d): Second layer normalization after attention
            attention (MultiHeadAttention): Multi-head attention layer
            ff (FeedForward): Feed-forward network layer
        """
        ### BEGIN ASSIGN3_3
        self.ln_1 = LayerNorm1d(dim=n_embd, eps=ln_eps, backend=backend) 
        self.ln_2 = LayerNorm1d(dim=n_embd, eps=ln_eps, backend=backend)
        self.attention = MultiHeadAttention(n_embd=n_embd, n_head=n_head, p_dropout=p_dropout, bias=bias, backend=backend) 
        self.ff = FeedForward(n_embd=n_embd, p_dropout=p_dropout, bias=bias, backend=backend) 
        ### END ASSIGN3_3

    def forward(self, x):
        """
        Forward pass through transformer layer with pre-layer normalization.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, n_embd)
        
        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, n_embd)
        """
        batch_size, seq_len, n_embd = x.shape
        ### BEGIN YOUR SOLUTION
        input_1 = x
        y = self.ln_1(input_1)
        y = self.attention(y)

        input_2 = y + input_1
        z = self.ln_2(input_2) 
        z = self.ff(z)
        
        return z + input_2
        ### END YOUR SOLUTION


class DecoderLM(Module):
    def __init__(
        self, 
        n_vocab: int,
        n_embd: int,
        n_head: int,
        n_positions: int,
        p_dropout: float=0.1,
        ln_eps: float=1e-5, 
        bias: bool=True,
        backend: TensorBackend=None
    ):
        super().__init__()
        """
        Initialize a decoder-only transformer language model.
        
        Args:
            n_vocab (int): Vocabulary size
            n_embd (int): Embedding dimension
            n_head (int): Number of attention heads
            n_positions (int): Maximum sequence length
            p_dropout (float): Dropout probability, default 0.1
            ln_eps (float): Layer normalization epsilon, default 1e-5
            bias (bool): Whether to use bias in linear layers, default True
            backend (TensorBackend): Backend for tensor operations
            
        Attributes:
            token_embeddings (Embedding): Token embedding layer
            position_embeddings (Embedding): Position embedding layer
            t_layer_1 (TransformerLayer): First transformer layer
            t_layer_2 (TransformerLayer): Second transformer layer
            t_layer_3 (TransformerLayer): Third transformer layer
            t_layer_4 (TransformerLayer): Fourth transformer layer
            dropout (Dropout): Dropout layer before transformer layers
            ln (LayerNorm1d): Final layer normalization
            lm_head (Linear): Language model head for vocabulary projection
        """
        self.backend = backend
        self.n_embd = n_embd
        self.n_vocab = n_vocab
        ### BEGIN ASSIGN3_3
        self.token_embeddings = Embedding(num_embeddings=self.n_vocab, embedding_dim=n_embd, backend=backend)  
        self.position_embeddings = Embedding(num_embeddings=n_positions, embedding_dim=n_embd, backend=backend) 
        self.t_layer_1 = TransformerLayer(n_embd=self.n_embd, n_head=n_head, p_dropout=p_dropout, ln_eps=ln_eps, bias=bias, backend=backend) 
        self.t_layer_2 = TransformerLayer(n_embd=self.n_embd, n_head=n_head, p_dropout=p_dropout, ln_eps=ln_eps, bias=bias, backend=backend) 
        self.t_layer_3 = TransformerLayer(n_embd=self.n_embd, n_head=n_head, p_dropout=p_dropout, ln_eps=ln_eps, bias=bias, backend=backend) 
        self.t_layer_4 = TransformerLayer(n_embd=self.n_embd, n_head=n_head, p_dropout=p_dropout, ln_eps=ln_eps, bias=bias, backend=backend) 
        self.dropout = Dropout(p_dropout)
        self.ln = LayerNorm1d(dim=self.n_embd, eps=ln_eps, backend=backend) 
        self.lm_head = Linear(self.n_embd, self.n_vocab, bias=False, backend=backend) 
        ### END ASSIGN3_3
    
    def forward(self, idx):
        """
        Forward pass through decoder-only transformer language model.
        
        Args:
            idx (Tensor): Input token indices of shape (batch_size, seq_len)
        
        Returns:
            Tensor: Logits of shape (batch_size, seq_len, n_vocab)
        """
        
        batch_size, seq_len = idx.shape

        ### BEGIN ASSIGN3_3
        # 1. Get token embeddings of shape (batch_size, seq_len, n_embd)
        # 2. Create positional embeddings of shape (1, seq_len, n_embd):
        #    - Create position ids tensor [0, 1, 2, ..., seq_len-1] of shape (1, seq_len)
        #    - Pass through positional embedding layer
        #    - Ensure output shape is (1, seq_len, n_embd)
        # 3. Add token and positional embeddings
        # 4. Apply dropout
        # 5. Pass through transformer layers (t_layer_1 to t_layer_4)
        # 6. Apply final layer normalization
        # 7. Project to vocabulary size using lm_head
        
        tok_embeddings = self.token_embeddings(idx)

        pos_ids = tensor_from_numpy(np.arange(seq_len).reshape(1, -1), backend=self.backend)
        pos_embeddings = self.position_embeddings(pos_ids)
        b, s, e = pos_embeddings.shape
        assert b == 1 and s == seq_len and e == self.n_embd
        
        emb = self.dropout(tok_embeddings + pos_embeddings)
        
        l_output = self.t_layer_1(emb)
        l_output = self.t_layer_2(l_output)
        l_output = self.t_layer_3(l_output)
        l_output = self.t_layer_4(l_output)
        
        out = self.ln(l_output)
        return self.lm_head(out)
        
        ### END ASSIGN3_3
