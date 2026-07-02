"""
SolsticePulse — a GPT-2-style decoder-only Transformer language model.

Architecture summary (GPT-2-like):
    - Decoder-only Transformer (causal self-attention, no encoder).
    - Learned absolute position embeddings added to token embeddings
      (same scheme as GPT-2; not rotary/ALiBi).
    - Pre-normalization: norm -> sublayer -> residual add, for both the
      attention and MLP blocks (GPT-2 also uses pre-norm, but with
      LayerNorm instead of RMSNorm — see below).
    - MLP block: Linear -> GELU -> Linear (same expansion-ratio design
      as GPT-2's feed-forward block, default 4x hidden_size).
    - Weight tying between the input token embedding and the output
      LM head (also used in GPT-2).

Difference from vanilla GPT-2 — RMSNorm instead of LayerNorm:
    GPT-2 normalizes activations with LayerNorm, which re-centers
    (subtracts the mean) AND re-scales (divides by the std) each
    activation vector, then applies a learned per-channel gain and bias.

    This model instead uses RMSNorm (Root Mean Square Layer Norm),
    which only re-scales — it divides each vector by its root-mean-square
    magnitude and applies a learned per-channel gain (no bias, no mean
    subtraction). RMSNorm is cheaper to compute (no mean/variance pair,
    just a mean-of-squares) and is used in many modern LLMs (e.g. LLaMA)
    because it tends to be more stable at scale while being roughly as
    effective as LayerNorm for Transformer training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Unlike nn.LayerNorm, RMSNorm does NOT subtract the mean (no
    re-centering) and has no bias term — it only rescales each vector
    by its root-mean-square value and applies a learned per-channel
    weight. This makes it slightly cheaper than LayerNorm and is the
    normalization used in models like LLaMA, in place of GPT-2's
    original LayerNorm.
    """
    def __init__(self, hidden_size, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon
        # Learned per-channel scale (gamma). No bias/beta term, unlike LayerNorm.
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        # Root-mean-square over the last (hidden) dimension.
        # epsilon avoids division by zero for near-zero activations.
        rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.epsilon)
        # Rescale by RMS, then apply the learned gain. Note: no mean subtraction here.
        return x * self.weight / rms


class MLP(nn.Module):
    """
    GPT-2-style feed-forward (position-wise) block:
    Linear (expand) -> GELU -> Dropout -> Linear (project back).
    Typically intermediate_size = 4 * hidden_size, same as GPT-2.
    """
    def __init__(self, hidden_size, intermediate_size, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)   # expand
        self.fc2 = nn.Linear(intermediate_size, hidden_size)   # project back down
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # GPT-2 uses GELU (approx) activation

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class SolsticePulseConfig:
    """
    Hyperparameter container, analogous to GPT2Config in Hugging Face.
    Defaults below match GPT-2 small (124M-ish) dimensions.
    """
    def __init__(self, vocab_size=50257, max_position_embeddings=2048, hidden_size=768,
                 num_attention_heads=12, intermediate_size=3072, num_hidden_layers=12,
                 dropout_rate=0.1):
        self.vocab_size = vocab_size                          # GPT-2 BPE vocab size
        self.max_position_embeddings = max_position_embeddings  # max sequence length supported
        self.hidden_size = hidden_size                        # model/embedding dimension (d_model)
        self.num_attention_heads = num_attention_heads         # number of attention heads
        self.intermediate_size = intermediate_size             # MLP hidden dim (usually 4x hidden_size)
        self.num_hidden_layers = num_hidden_layers              # number of decoder blocks
        self.dropout_rate = dropout_rate


class Attention(nn.Module):
    """
    Standard multi-head causal self-attention (GPT-2-style), computed
    "manually" (not via scaled_dot_product_attention) so the additive
    mask logic is explicit.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        # Separate projections for Q, K, V (GPT-2 uses a single fused
        # c_attn projection internally, but the math is equivalent).
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)  # output projection (c_proj)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.size()

        # Project to Q, K, V then reshape to [batch, heads, seq_len, head_dim]
        # so each head attends independently.
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention scores: QK^T / sqrt(d_head)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply additive attention mask (causal mask and/or padding mask).
        # Masked positions get a large negative value so softmax ~= 0 there.
        if attention_mask is not None:
            # Broadcast a 2D or 3D mask up to 4D [batch, heads, seq_len, seq_len]
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)

            attention_mask = attention_mask.expand_as(scores)
            scores = scores + attention_mask

        # Normalize scores into probabilities over the key dimension.
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values, then merge heads back into hidden_size.
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # Final output projection.
        output = self.out_proj(context)
        return output


class DecoderBlock(nn.Module):
    """
    A single GPT-2-style Transformer decoder block using
    PRE-normalization (norm applied before the sublayer, residual added
    after) for both attention and MLP:

        x = x + Attention(RMSNorm(x))
        x = x + MLP(RMSNorm(x))

    GPT-2 uses the same pre-norm residual structure, but with LayerNorm
    in place of RMSNorm.
    """
    def __init__(self, config):
        super().__init__()
        self.attention = Attention(config)
        self.attn_norm = RMSNorm(config.hidden_size)   # pre-attention RMSNorm
        self.mlp = MLP(config.hidden_size, config.intermediate_size, config.dropout_rate)
        self.mlp_norm = RMSNorm(config.hidden_size)    # pre-MLP RMSNorm
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x, attention_mask=None):
        # --- Self-attention sub-block (pre-norm + residual) ---
        residual = x
        x = self.attn_norm(x)
        x = self.attention(x, attention_mask)
        x = residual + self.dropout(x)

        # --- Feed-forward sub-block (pre-norm + residual) ---
        residual = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = residual + self.dropout(x)

        return x


class SolsticePulse(nn.Module):
    """
    Full GPT-2-like decoder-only language model:
    token embedding + learned position embedding -> N decoder blocks
    -> final RMSNorm -> LM head (tied to token embedding weights).
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token + learned absolute position embeddings (GPT-2 style,
        # as opposed to sinusoidal or rotary position encodings).
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # Stack of N decoder blocks.
        self.layers = nn.ModuleList([DecoderBlock(config) for _ in range(config.num_hidden_layers)])

        # Final normalization before the LM head (GPT-2 also has a
        # final LayerNorm here; we use RMSNorm for consistency).
        self.final_norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        # Weight tying: input embedding and output projection share
        # weights (reduces params, standard GPT-2 trick).
        self.lm_head.weight = self.token_embedding.weight

        self.dropout = nn.Dropout(config.dropout_rate)

        # Apply custom weight initialization to all submodules.
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # GPT-2-style init: small-std normal for weights, zeros for biases.
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.size()

        # Build position ids [0, 1, ..., seq_len-1] and look up embeddings.
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        x = self.dropout(token_embeds + position_embeds)

        if attention_mask is None:
            # No padding mask given -> build a causal (autoregressive) mask:
            # upper-triangular part (future tokens) is set to -1e9 so
            # softmax zeroes those positions out. Each token can only
            # attend to itself and earlier tokens.
            mask = torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device) * -1e9)
            mask = mask.view(1, 1, seq_len, seq_len)  # [1, 1, seq_len, seq_len]
        else:
            # Convert a 0/1 padding mask into additive form:
            # 1 (keep) -> 0.0, 0 (pad) -> -1e9.
            mask = (1.0 - attention_mask) * -1e9
            mask = mask.view(batch_size, 1, 1, seq_len)  # [batch_size, 1, 1, seq_len]

        # Broadcast the mask across all attention heads.
        mask = mask.expand(-1, self.config.num_attention_heads, -1, -1)

        # Run through each decoder block sequentially.
        for layer in self.layers:
            x = layer(x, mask)

        # Final norm, then project hidden states to vocab-size logits.
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits


# Configuration matching GPT-2 "small" scale (~124M params).
config = SolsticePulseConfig(
    vocab_size=50257,
    max_position_embeddings=2048,
    hidden_size=768,
    num_attention_heads=12,
    intermediate_size=3072,
    num_hidden_layers=12,
    dropout_rate=0.1
)
