from transformers import AutoTokenizer


MODEL_NAME = "gpt2"


def tokenize_text(text):
    # Load GPT-2 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Convert text into GPT-2 tokens
    tokens = tokenizer.tokenize(text)

    # Convert text into token IDs
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    print("\n========== TOKENIZATION ==========")
    print("Input Text:", text)
    print("Tokens:", tokens)
    print("Token IDs:", token_ids)

    return tokenizer, token_ids