from models.tokenization import tokenize_text
from models.embeddings import create_embeddings
from models.transformer import transformer_calculation
from inference.text_generation import generate_text
from common.config import MAX_NEW_TOKENS


text = "I love machine learning"

print("\n========== GPT-2 MINI PIPELINE ==========")

# Step 1: Tokenization
tokenizer, token_ids = tokenize_text(text)

# Step 2: Token + Position Embeddings
model, embeddings = create_embeddings(token_ids)

# Step 3: Transformer calculations
transformer_output = transformer_calculation(
    model,
    embeddings
)

# Step 4: Text Generation
generated_text = generate_text(
    model,
    tokenizer,
    token_ids,
    MAX_NEW_TOKENS
)

print("\n========== FINAL OUTPUT ==========")
print(generated_text)