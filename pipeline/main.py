import sys
from pathlib import Path

# Ensure UTF-8 output encoding across platforms
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project root to sys.path to allow execution from any working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.tokenization import tokenize_text
from models.embeddings import create_embeddings
from models.transformer import transformer_calculation
from inference.text_generation import generate_text
from common.config import MAX_NEW_TOKENS


def run_pipeline(text: str = "I love machine learning"):
    print("\n" + "#" * 60)
    print("           GPT-2 COMPLETE END-TO-END PIPELINE")
    print("#" * 60)
    print(f"Input Prompt: {text!r}\n")

    # Step 1: Tokenization (Text -> BPE Tokens -> Token IDs)
    tokenizer, token_ids = tokenize_text(text)

    # Step 2: Token + Position Embeddings (wte + wpe)
    model, embeddings = create_embeddings(token_ids)

    # Step 3: Transformer calculations (12 Attention + FFN blocks)
    transformer_output = transformer_calculation(
        model,
        embeddings,
        token_ids=token_ids,
        tokenizer=tokenizer
    )

    # Step 4: Text Generation (LM Head -> Probabilities -> Autoregressive decoding)
    generated_text = generate_text(
        model,
        tokenizer,
        token_ids,
        max_new_tokens=MAX_NEW_TOKENS
    )

    print("\n" + "=" * 60)
    print("                    FINAL OUTPUT")
    print("=" * 60)
    print(generated_text)
    print("=" * 60 + "\n")

    return generated_text


if __name__ == "__main__":
    prompt = "I love machine learning"
    run_pipeline(prompt)