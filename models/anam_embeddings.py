import torch
from transformers import AutoModelForCausalLM

MODEL_NAME = "gpt2"


def create_embeddings(token_ids):

    # Load GPT-2 model
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # Convert token IDs into tensor
    input_ids = torch.tensor([token_ids])

    # Token embeddings
    token_embeddings = model.transformer.wte(input_ids)

    # Position IDs
    position_ids = torch.arange(
        input_ids.shape[1]
    ).unsqueeze(0)

    # Position embeddings
    position_embeddings = model.transformer.wpe(position_ids)

    # Final input embeddings
    final_embeddings = (
        token_embeddings + position_embeddings
    )

    print("\n========== ANAM: EMBEDDINGS ==========")

    print(
        "Token Embedding Shape:",
        token_embeddings.shape
    )

    print(
        "Position Embedding Shape:",
        position_embeddings.shape
    )

    print(
        "Final Embedding Shape:",
        final_embeddings.shape
    )

    return model, final_embeddings