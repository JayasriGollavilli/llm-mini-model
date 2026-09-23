import os
import warnings
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# Suppress framework noise for clean execution
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")


def print_model_parameters(model):
    """Calculates and prints model parameter statistics."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size_mb = (total_params * 4) / (1024 ** 2)

    print("\n" + "=" * 70)
    print("                    MODEL PARAMETER REPORT")
    print("=" * 70)
    print(f"  • Architecture:           {model.config.architectures[0] if model.config.architectures else 'CausalLM'}")
    print(f"  • Total Parameters:       {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"  • Trainable Parameters:   {trainable_params:,}")
    print(f"  • Model Footprint Size:   ~{param_size_mb:.2f} MB")
    print("=" * 70 + "\n")


def inspect_llm_step(prompt_text: str, tokenizer, model, device):
    """Executes a single step of the forward pass with full tensor tracing."""
    print("=" * 70)
    print("                 LLM INFERENCE PIPELINE — STEP 1")
    print("=" * 70)

    # [1] RAW INPUT
    print("\n[1] RAW INPUT")
    print("-" * 70)
    print(f'"{prompt_text}"')

    # [2] TOKENIZATION
    input_ids_list = tokenizer.encode(prompt_text)
    tokens = [tokenizer.decode([tid]) for tid in input_ids_list]
    print("\n[2] TOKENIZATION (Subword Split)")
    print("-" * 70)
    print(tokens)

    # [3] TOKEN IDs
    print("\n[3] TOKEN IDs (Vocabulary Mapping)")
    print("-" * 70)
    print(input_ids_list)

    input_ids = torch.tensor([input_ids_list]).to(device)

    # [4] EMBEDDING
    wte = model.transformer.wte
    token_embeds = wte(input_ids)
    print("\n[4] TOKEN EMBEDDINGS (wte)")
    print("-" * 70)
    print(f"Embedding Dimension: {token_embeds.shape[-1]}")
    print(f"Tensor Shape: {list(token_embeds.shape)}")
    print(f"Sample Vector for First Token ('{tokens[0]}') [First 5 values]:")
    print([round(v, 4) for v in token_embeds[0, 0, :5].detach().cpu().tolist()])

    # [5] POSITION ENCODING
    wpe = model.transformer.wpe
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
    pos_embeds = wpe(position_ids)
    
    print("\n[5] POSITION EMBEDDINGS (wpe)")
    print("-" * 70)
    print(f"Position Tensor Shape: {list(pos_embeds.shape)}")
    print(f"Sample Vector for Position 0 [First 5 values]:")
    print([round(v, 4) for v in pos_embeds[0, 0, :5].detach().cpu().tolist()])

    # Combined Embeddings
    hidden_states = token_embeds + pos_embeds
    print("\n[COMBINED EMBEDDINGS: Token + Position]")
    print(f"Input Tensor Shape to Transformer Block 1: {list(hidden_states.shape)}")

    # [6] TRANSFORMER INTERNAL LAYERS PASS
    print("\n[6] TRANSFORMER BLOCKS (Attention & Feed-Forward Layers)")
    print("-" * 70)
    
    with torch.no_grad():
        outputs = model(
            input_ids, 
            output_attentions=True, 
            output_hidden_states=True
        )

    all_hidden_states = outputs.hidden_states
    all_attentions = outputs.attentions
    n_layers = model.config.n_layer

    print(f"Total Layers Processed: {n_layers}")
    
    if all_attentions is not None and len(all_attentions) > 0:
        print("\n--- Layer 1 Internal State ---")
        print(f"Attention Weights Shape: {list(all_attentions[0].shape)} [Batch, Heads, Seq_Len, Seq_Len]")
        attn_val = all_attentions[0][0, 0, -1, 0].item()
        print(f"Attention Weight (Last Token -> First Token '{tokens[0]}'): {attn_val:.4f}")
        print(f"Hidden State Output Shape: {list(all_hidden_states[1].shape)}")
        
        print(f"\n--- Layer {n_layers} Internal State ---")
        print(f"Attention Weights Shape: {list(all_attentions[-1].shape)}")
        print(f"Final Hidden State Tensor Shape: {list(all_hidden_states[-1].shape)}")
        print(f"Final Vector for Last Token ('{tokens[-1]}') [First 5 values]:")
        print([round(v, 4) for v in all_hidden_states[-1][0, -1, :5].detach().cpu().tolist()])

    # [7] LOGITS
    logits = outputs.logits
    vocab_size = model.config.vocab_size
    print("\n[7] LOGITS (LM Head Projection)")
    print("-" * 70)
    print(f"Vocabulary Size: {vocab_size:,}")
    print(f"Logits Tensor Shape: {list(logits.shape)} [Batch, Seq_Len, Vocab_Size]")
    
    last_token_logits = logits[0, -1, :]
    print(f"Raw Logits for Last Token [First 5 raw values]:")
    print([round(v, 4) for v in last_token_logits[:5].detach().cpu().tolist()])

    # [8] PROBABILITIES
    probs = F.softmax(last_token_logits, dim=-1)
    top_k = 5
    top_probs, top_indices = torch.topk(probs, top_k)

    print("\n[8] PROBABILITIES (Softmax Distribution)")
    print("-" * 70)
    print(f"Top {top_k} Candidate Next Tokens:")
    for rank, (p, idx) in enumerate(zip(top_probs, top_indices), start=1):
        token_str = tokenizer.decode([idx])
        print(f"  Rank {rank}: '{token_str}' | Probability: {p.item():.4f} ({p.item()*100:.2f}%) | Logit Value: {last_token_logits[idx].item():.2f}")

    # [9] NEXT TOKEN SELECTION
    selected_id = top_indices[0].item()
    selected_token = tokenizer.decode([selected_id])
    print("\n[9] NEXT TOKEN SELECTION")
    print("-" * 70)
    print(f"Selected Token: '{selected_token}' (ID: {selected_id})")

    # [10] UPDATED CONTEXT
    generated_text = prompt_text + selected_token
    print("\n[10] GENERATED OUTPUT CONTEXT")
    print("-" * 70)
    print(f'"{generated_text}"')
    print("=" * 70 + "\n")

    return selected_token, generated_text


def run_full_generation(prompt_text: str, max_gen_tokens: int = 3):
    """Orchestrates model loading, parameter reporting, and auto-regressive generation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"

    print("\n[+] Initializing model and tokenizer...")
    config = AutoConfig.from_pretrained(model_name)
    config.output_attentions = True
    config.output_hidden_states = True

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config).to(device)
    model.eval()

    # Report Parameters
    print_model_parameters(model)

    current_prompt = prompt_text
    
    # Auto-Regressive Text Generation Loop
    for step in range(1, max_gen_tokens + 1):
        if step == 1:
            # Full detailed printout for Step 1
            _, current_prompt = inspect_llm_step(
                prompt_text=current_prompt, 
                tokenizer=tokenizer, 
                model=model, 
                device=device
            )
        else:
            # Quiet background generation for remaining tokens
            input_ids = tokenizer.encode(current_prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(input_ids)
            next_token_id = torch.argmax(outputs.logits[0, -1, :]).item()
            next_token_str = tokenizer.decode([next_token_id])
            current_prompt += next_token_str

    # Final Output Summary
    print("============================================================")
    print("            AUTO-REGRESSIVE TEXT GENERATION COMPLETE")
    print("============================================================")
    print(f"  • Input Prompt:        '{prompt_text}'")
    print(f"  • Tokens Generated:    {max_gen_tokens}")
    print(f"  • Final Output Text:   '{current_prompt}'")
    print("============================================================\n")


if __name__ == "__main__":
    user_input = input("\nEnter custom text (Press Enter for default 'The capital of India is'): ").strip()
    if not user_input:
        user_input = "The capital of India is"

    tokens_to_generate = input("Enter number of tokens to generate (Press Enter for default 3): ").strip()
    tokens_to_generate = int(tokens_to_generate) if tokens_to_generate.isdigit() else 3

    run_full_generation(user_input, max_gen_tokens=tokens_to_generate)