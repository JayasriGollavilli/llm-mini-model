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
import torch.nn.functional as F
from common.config import MAX_NEW_TOKENS, TEMPERATURE, TOP_K, GREEDY


def generate_text(
    model,
    tokenizer,
    token_ids,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    top_k=TOP_K,
    greedy=GREEDY
):
    """
    Step 4: Text Generation
    Performs autoregressive decoding from transformer hidden states:
    - Language Model Head projection: hidden_dim (768) -> vocab_size (50,257)
    - Softmax probability distribution over vocabulary
    - Top candidate token analysis
    - Iterative autoregressive loop appending each new token
    """
    print("\n" + "=" * 60)
    print("             STEP 4: TEXT GENERATION")
    print("=" * 60)

    input_ids = torch.tensor([token_ids], dtype=torch.long)
    prompt_text = tokenizer.decode(token_ids)
    print(f"Prompt Text       : {prompt_text!r}")
    print(f"Decoding Mode     : {'Greedy (Argmax)' if greedy else f'Sampling (Temp={temperature}, Top-k={top_k})'}")
    print(f"Max New Tokens    : {max_new_tokens}")

    # Initial forward pass to show LM Head projection and top probabilities
    with torch.no_grad():
        initial_outputs = model(input_ids=input_ids)
    initial_logits = initial_outputs.logits[0, -1]  # shape: [50257]
    initial_probs = F.softmax(initial_logits, dim=-1)

    print(f"\nLM Head Projection Output (Logits): {list(initial_logits.shape)} [vocab_size]")
    print(f"Sum of Softmax Probabilities       : {initial_probs.sum().item():.6f}")

    # Top-5 candidates for the first predicted token
    top5 = torch.topk(initial_probs, 5)
    print("\nTop 5 Candidates for the First Generated Token:")
    print("  Rank | Token                | Probability | Visual Distribution")
    print("  -----+----------------------+-------------+-----------------------------")
    for rank, (prob_val, idx) in enumerate(zip(top5.values.tolist(), top5.indices.tolist()), 1):
        token_str = tokenizer.decode([idx])
        bar = "#" * int(prob_val * 40)
        print(f"  {rank:>4} | {repr(token_str):<20} | {prob_val * 100:>10.2f}% | {bar}")

    print("\nAutoregressive Generation Steps:")
    print("  Step | Generated Token      | Probability | Current Generated Text")
    print("  -----+----------------------+-------------+---------------------------------------")

    for step in range(1, max_new_tokens + 1):
        with torch.no_grad():
            outputs = model(input_ids=input_ids)

        logits = outputs.logits[0, -1]

        if greedy:
            probs = F.softmax(logits, dim=-1)
            next_token_id = torch.argmax(probs).view(1)
            next_token_prob = probs[next_token_id].item()
        else:
            scaled_logits = logits / temperature
            kth_val = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1))).values[-1]
            scaled_logits[scaled_logits < kth_val] = -float("inf")
            probs = F.softmax(scaled_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            next_token_prob = probs[next_token_id].item()

        # Append next token to input_ids sequence
        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)

        token_piece = tokenizer.decode(next_token_id)
        current_sentence = tokenizer.decode(input_ids[0], skip_special_tokens=True)

        # Truncate preview if too long for table
        display_preview = current_sentence
        if len(display_preview) > 36:
            display_preview = "..." + display_preview[-33:]

        print(f"  {step:>4} | {repr(token_piece):<20} | {next_token_prob * 100:>10.2f}% | {display_preview}")

        # Stop if end-of-sequence token is generated
        if next_token_id.item() == tokenizer.eos_token_id:
            print(f"\n[EOS] End-of-sequence token reached at step {step}.")
            break

    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    continuation = generated_text[len(prompt_text):]

    print("\nGeneration Complete:")
    print(f"  Prompt       : {prompt_text}")
    print(f"  Continuation : {continuation.strip()}")
    print("-" * 60)

    return generated_text


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    mod = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager")
    mod.eval()
    t_ids = tok.encode("I love machine learning", add_special_tokens=False)
    generate_text(mod, tok, t_ids, max_new_tokens=10)