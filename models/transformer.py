import math
import torch
import torch.nn as nn


# ============================================================
# MULTI-HEAD SELF-ATTENTION
# ============================================================

class MultiHeadSelfAttention(nn.Module):

    def __init__(self, hidden_size=768, num_heads=12):
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_heads"
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Query, Key and Value projections
        self.q_proj = nn.Linear(
            hidden_size,
            hidden_size
        )

        self.k_proj = nn.Linear(
            hidden_size,
            hidden_size
        )

        self.v_proj = nn.Linear(
            hidden_size,
            hidden_size
        )

        # Output projection
        self.out_proj = nn.Linear(
            hidden_size,
            hidden_size
        )

    def split_heads(self, x):
        """
        Input:
            [batch, sequence_length, hidden_size]

        Output:
            [batch, num_heads, sequence_length, head_dim]
        """

        batch_size, sequence_length, _ = x.size()

        x = x.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        )

        x = x.transpose(1, 2)

        return x

    def combine_heads(self, x):
        """
        Input:
            [batch, num_heads, sequence_length, head_dim]

        Output:
            [batch, sequence_length, hidden_size]
        """

        batch_size = x.size(0)
        sequence_length = x.size(2)

        x = x.transpose(1, 2)

        x = x.contiguous().view(
            batch_size,
            sequence_length,
            self.hidden_size
        )

        return x

    def forward(self, x, attention_mask=None):

        # ----------------------------------------------------
        # STEP 1: QUERY, KEY AND VALUE
        # ----------------------------------------------------

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # ----------------------------------------------------
        # STEP 2: SPLIT INTO ATTENTION HEADS
        # ----------------------------------------------------

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # ----------------------------------------------------
        # STEP 3: CALCULATE Q × K^T
        # ----------------------------------------------------

        attention_scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )

        # ----------------------------------------------------
        # STEP 4: SCALE BY sqrt(head_dim)
        # ----------------------------------------------------

        attention_scores = attention_scores / math.sqrt(
            self.head_dim
        )

        # ----------------------------------------------------
        # STEP 5: CAUSAL ATTENTION MASK
        # ----------------------------------------------------

        sequence_length = x.size(1)

        causal_mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device
            )
        )

        causal_mask = causal_mask.view(
            1,
            1,
            sequence_length,
            sequence_length
        )

        attention_scores = attention_scores.masked_fill(
            causal_mask == 0,
            float("-inf")
        )

        # Optional additional mask
        if attention_mask is not None:

            attention_scores = attention_scores.masked_fill(
                attention_mask == 0,
                float("-inf")
            )

        # ----------------------------------------------------
        # STEP 6: SOFTMAX
        # ----------------------------------------------------

        attention_weights = torch.softmax(
            attention_scores,
            dim=-1
        )

        # ----------------------------------------------------
        # STEP 7: ATTENTION WEIGHTS × VALUE
        # ----------------------------------------------------

        attention_output = torch.matmul(
            attention_weights,
            V
        )

        # ----------------------------------------------------
        # STEP 8: COMBINE ATTENTION HEADS
        # ----------------------------------------------------

        attention_output = self.combine_heads(
            attention_output
        )

        # ----------------------------------------------------
        # STEP 9: OUTPUT PROJECTION
        # ----------------------------------------------------

        attention_output = self.out_proj(
            attention_output
        )

        return attention_output, attention_weights


# ============================================================
# FEED-FORWARD NETWORK
# ============================================================

class FeedForward(nn.Module):

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072
    ):
        super().__init__()

        self.fc1 = nn.Linear(
            hidden_size,
            intermediate_size
        )

        self.activation = nn.GELU()

        self.fc2 = nn.Linear(
            intermediate_size,
            hidden_size
        )

    def forward(self, x):

        # Expand
        x = self.fc1(x)

        # Non-linear activation
        x = self.activation(x)

        # Project back
        x = self.fc2(x)

        return x


# ============================================================
# TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(nn.Module):

    def __init__(
        self,
        hidden_size=768,
        num_heads=12,
        intermediate_size=3072
    ):
        super().__init__()

        # Self-attention
        self.attention = MultiHeadSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads
        )

        # First Layer Normalization
        self.norm1 = nn.LayerNorm(
            hidden_size
        )

        # Feed-forward network
        self.feed_forward = FeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size
        )

        # Second Layer Normalization
        self.norm2 = nn.LayerNorm(
            hidden_size
        )

    def forward(self, x, attention_mask=None):

        # ====================================================
        # SELF-ATTENTION + RESIDUAL + NORM
        # ====================================================

        attention_output, attention_weights = self.attention(
            x,
            attention_mask
        )

        # Residual connection 1
        x = x + attention_output

        # Layer Normalization 1
        x = self.norm1(x)

        # ====================================================
        # FEED-FORWARD NETWORK + RESIDUAL + NORM
        # ====================================================

        feed_forward_output = self.feed_forward(x)

        # Residual connection 2
        x = x + feed_forward_output

        # Layer Normalization 2
        x = self.norm2(x)

        return x, attention_weights


# ============================================================
# TRANSFORMER CALCULATION FOR PIPELINE
# ============================================================

def transformer_calculation(model, embeddings, token_ids=None, tokenizer=None):
    """
    Step 3: Transformer Calculations
    Passes input embeddings through the 12 Transformer blocks of GPT-2:
    - Multi-Head Self-Attention (12 heads, causal attention mask)
    - Layer Normalization & Residual Connections
    - Feed-Forward MLP (768 -> 3072 -> 768)
    - Output Hidden States across all 12 layers
    """
    print("\n" + "=" * 60)
    print("          STEP 3: TRANSFORMER CALCULATION")
    print("=" * 60)

    # 1. Architecture details
    n_layer = model.config.n_layer
    n_head = model.config.n_head
    n_embd = model.config.n_embd
    head_dim = n_embd // n_head
    ffn_dim = 4 * n_embd

    print("Architecture Specifications:")
    print(f"  Transformer Layers (Blocks) : {n_layer}")
    print(f"  Attention Heads per Layer   : {n_head}")
    print(f"  Hidden Dimension (d_model)  : {n_embd}")
    print(f"  Head Dimension (d_k)        : {head_dim} ({n_embd} / {n_head})")
    print(f"  Feed-Forward Intermediate   : {ffn_dim} (4 x {n_embd})")
    print(f"  Input Embeddings Shape      : {list(embeddings.shape)}")

    # 2. Forward pass through GPT-2 Transformer blocks
    with torch.no_grad():
        transformer_output = model.transformer(
            inputs_embeds=embeddings,
            output_attentions=True,
            output_hidden_states=True
        )

    last_hidden_state = transformer_output.last_hidden_state
    hidden_states = transformer_output.hidden_states
    attentions = transformer_output.attentions

    # 3. Layer-by-layer hidden state evolution (L2 norm)
    print("\nHidden State Evolution (Representation Norm per Layer):")
    print("  Layer        | Vector L2 Norm (Last Token) | Status")
    print("  -------------+-----------------------------+-----------------------")
    for layer_idx, hs in enumerate(hidden_states):
        norm_val = hs[0, -1].norm().item()
        if layer_idx == 0:
            label = "Input Emb"
            desc = "Raw Token + Pos Embeddings"
        elif layer_idx == n_layer:
            label = f"Block {layer_idx:>2}"
            desc = "Final Block Output (Pre-ln_f)"
        else:
            label = f"Block {layer_idx:>2}"
            desc = f"Attention + FFN Block {layer_idx}"
        print(f"  {label:<12} | {norm_val:>27.3f} | {desc}")

    # 4. Multi-Head Attention Analysis & Visualization
    if attentions is not None and len(attentions) > 0:
        attn_layer = attentions[-1]  # shape: [batch, num_heads, seq_len, seq_len]
        print(f"\nMulti-Head Attention Tensor Shape: {list(attn_layer.shape)}")
        print("  [batch_size, num_heads, seq_len (queries), seq_len (keys)]")

        # Get token labels for display if tokenizer/token_ids available
        token_labels = []
        seq_len = embeddings.shape[1]
        if tokenizer is not None and token_ids is not None:
            token_labels = [repr(tokenizer.decode([tid])) for tid in token_ids]
        else:
            token_labels = [f"Token_{i}" for i in range(seq_len)]

        # Visualize attention weights for the last token in Head 0 of the final layer
        head_idx = 0
        last_token_attn = attn_layer[0, head_idx, -1, :].tolist()
        print(f"\nAttention Map: Where the LAST token looks (Layer {n_layer}, Head {head_idx}):")
        print("  Token                | Attn Weight | Attention Distribution")
        print("  ---------------------+-------------+-----------------------------")
        for tok_lbl, w in zip(token_labels, last_token_attn):
            bar = "#" * int(w * 35)
            print(f"  {tok_lbl:<20} | {w:>10.4f}  | {bar}")

        print("\nNote: Causal masking prevents tokens from attending to subsequent tokens.")

    # 5. Custom Transformer Block Demonstration
    print("\nModular Custom PyTorch Block Check:")
    custom_block = TransformerBlock(
        hidden_size=n_embd,
        num_heads=n_head,
        intermediate_size=ffn_dim
    )
    with torch.no_grad():
        custom_out, custom_weights = custom_block(embeddings)
    print(f"  Custom TransformerBlock Output Shape: {list(custom_out.shape)}")
    print(f"  Custom MultiHeadSelfAttention Weights: {list(custom_weights.shape)}")

    # 6. Final Output Hidden State
    print(f"\nFinal Transformer Output Shape: {list(last_hidden_state.shape)} [batch, seq_len, 768]")
    print("These enriched contextual embeddings now feed into the LM prediction head.")
    print("-" * 60)

    return transformer_output


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("             TRANSFORMER BLOCK TEST")
    print("=" * 60)

    batch_size = 1
    sequence_length = 4
    hidden_size = 768
    num_heads = 12
    intermediate_size = 3072

    x = torch.randn(
        batch_size,
        sequence_length,
        hidden_size
    )

    transformer = TransformerBlock(
        hidden_size=hidden_size,
        num_heads=num_heads,
        intermediate_size=intermediate_size
    )

    output, attention_weights = transformer(x)

    print(f"Input shape           : {list(x.shape)}")
    print(f"Output shape          : {list(output.shape)}")
    print(f"Attention weights     : {list(attention_weights.shape)}")
    print("Transformer block unit test passed successfully.")