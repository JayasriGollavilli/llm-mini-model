import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output encoding across platforms
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch
from transformers import AutoModelForCausalLM
from common.config import MODEL_NAME


def create_embeddings(token_ids, model=None, model_name=MODEL_NAME):
    """
    Step 2: Token + Position Embeddings
    Combines token embeddings (wte) with positional embeddings (wpe).
    """
    # Load GPT-2 model if not already provided
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="eager"
        )
        model.eval()

    # Convert token IDs into tensor of shape [1, sequence_length]
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    seq_len = input_ids.shape[1]

    # Token embeddings lookup from wte (50,257 x 768)
    token_embeddings = model.transformer.wte(input_ids)

    # Position IDs: [0, 1, 2, ..., seq_len - 1]
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

    # Position embeddings lookup from wpe (1,024 x 768)
    position_embeddings = model.transformer.wpe(position_ids)

    # Final input embeddings: element-wise addition of token + position embeddings
    final_embeddings = token_embeddings + position_embeddings

    print("\n" + "=" * 60)
    print("           STEP 2: TOKEN & POSITION EMBEDDINGS")
    print("=" * 60)
    print(f"Sequence Length          : {seq_len} tokens")
    print(f"Token Vocab Matrix (wte) : {tuple(model.transformer.wte.weight.shape)} (vocab_size, hidden_dim)")
    print(f"Position Matrix    (wpe) : {tuple(model.transformer.wpe.weight.shape)} (max_positions, hidden_dim)")
    print(f"Input IDs Tensor Shape   : {list(input_ids.shape)}")
    print(f"Position IDs Shape       : {list(position_ids.shape)}")
    print(f"Token Embedding Shape    : {list(token_embeddings.shape)}")
    print(f"Position Embedding Shape : {list(position_embeddings.shape)}")
    print(f"Final Embedding Shape    : {list(final_embeddings.shape)} [batch, seq_len, hidden_dim]")

    print("\nEmbedding Vector Preview (first 6 of 768 dimensions per token):")
    print("  Pos | Token ID | Vector (wte + wpe truncated)")
    print("  ----+----------+---------------------------------------------------")
    for pos in range(seq_len):
        tid = token_ids[pos]
        sample_vals = " ".join(f"{v:+.3f}" for v in final_embeddings[0, pos, :6].tolist())
        print(f"  {pos:>3} | {tid:>8} | [{sample_vals} ...]")

    print("\nExplanation: Positional vectors are added to provide the Transformer")
    print("with word-order information, as self-attention is permutation-invariant.")
    print("-" * 60)

    return model, final_embeddings


if __name__ == "__main__":
    from models.tokenization import tokenize_text
    _, ids = tokenize_text("I love machine learning")
    create_embeddings(ids)