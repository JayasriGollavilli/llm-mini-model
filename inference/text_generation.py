import torch


def generate_text(model, tokenizer, token_ids, max_new_tokens=20):

    print("\n==========  TEXT GENERATION ==========")

    input_ids = torch.tensor([token_ids])

    for step in range(max_new_tokens):

        with torch.no_grad():
            outputs = model(input_ids=input_ids)

        logits = outputs.logits

        # Get predictions for the last token
        next_token_logits = logits[:, -1, :]

        # Select token with highest probability
        next_token = torch.argmax(
            next_token_logits,
            dim=-1
        ).unsqueeze(0)

        # Add predicted token to input
        input_ids = torch.cat(
            [input_ids, next_token],
            dim=1
        )

        print(
            f"Step {step + 1}:",
            tokenizer.decode(next_token[0])
        )

    generated_text = tokenizer.decode(
        input_ids[0],
        skip_special_tokens=True
    )

    print("\nGenerated Text:")
    print(generated_text)

    return generated_text