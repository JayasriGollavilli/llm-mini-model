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

from transformers import AutoTokenizer
from common.config import MODEL_NAME


def tokenize_text(text, model_name=MODEL_NAME):
    """
    Step 1: Tokenization
    Converts raw input text into Byte-Pair Encoding (BPE) tokens and token IDs.
    """
    # Load GPT-2 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Convert text into GPT-2 BPE tokens
    tokens = tokenizer.tokenize(text)

    # Convert text into token IDs
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    print("\n" + "=" * 60)
    print("                STEP 1: TOKENIZATION")
    print("=" * 60)
    print(f"Input Text      : {text!r}")
    print(f"Characters      : {len(text)}")
    print(f"Words           : {len(text.split())}")
    print(f"Tokenizer       : Byte-Pair Encoding (BPE) - {model_name}")
    print(f"Vocabulary Size : {tokenizer.vocab_size:,}")
    print(f"Number of Tokens: {len(tokens)}")

    print("\nToken Breakdown:")
    print("  Index | Token (BPE)          | Token ID | Decoded")
    print("  ------+----------------------+----------+-------------------")
    for idx, (tok, tid) in enumerate(zip(tokens, token_ids)):
        decoded_piece = tokenizer.decode([tid])
        display_tok = tok.replace("\u0120", "Ġ ")  # Clarify leading space
        print(f"  {idx:>5} | {repr(display_tok):<20} | {tid:>8} | {repr(decoded_piece):<18}")

    print(f"\nFinal Token IDs Vector : {token_ids}")
    print(f"Decoded Check          : {tokenizer.decode(token_ids)!r}")
    print("\nNote: 'Ġ' prefix marks a leading whitespace token in GPT-2 BPE.")
    print("-" * 60)

    return tokenizer, token_ids


if __name__ == "__main__":
    sample_text = "I love machine learning"
    tokenize_text(sample_text)