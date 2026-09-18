"""
=============================================================================
 LLM PIPELINE VISUALIZER
 Traces a user sentence through every stage of a Large Language Model:
 
   User Input -> Tokenizer -> Tokens -> Token IDs -> Embeddings
   -> Transformer (Attention + Feed Forward x N) -> Output Probabilities
   -> Next Token -> Repeat Process -> Generated Text
 
 Model : GPT-2 (124M parameters, 12 layers, 768 dims, 50257 vocab)
 Run   : python llm_pipeline.py
=============================================================================
"""
 
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
 
# ---------------------------------------------------------------- settings
MODEL_NAME     = "gpt2"
MAX_NEW_TOKENS = 20
TEMPERATURE    = 0.8      # lower = focused, higher = creative
TOP_K          = 50
GREEDY         = False    # True = always pick highest probability token
 
LINE = "=" * 78
 
 
def banner(stage_no, title):
    print("\n" + LINE)
    print(f"  STAGE {stage_no}: {title}")
    print(LINE)
 
 
def pause():
    input("\n   -- press ENTER to continue --")
 
 
# ============================================================ LOAD THE MODEL
def load_model():
    print("\nLoading GPT-2 ... (first run downloads ~500 MB, please wait)")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(
        MODEL_NAME,
        output_hidden_states=True,
        output_attentions=True,
    )
    model.eval()
    print("Model loaded successfully.")
    return tokenizer, model
 
 
# =========================================================== STAGE 1 : INPUT
def stage_user_input():
    banner(1, "USER INPUT")
    text = input("\n   Enter your sentence: ").strip()
    if not text:
        text = "The capital of France is"
        print(f"   (empty input - using default) {text}")
    print(f"\n   Raw text     : {text!r}")
    print(f"   Characters   : {len(text)}")
    print(f"   Words        : {len(text.split())}")
    return text
 
 
# ======================================================= STAGE 2 : TOKENIZER
def stage_tokenizer(tokenizer, text):
    banner(2, "TOKENIZER  ->  TOKENS")
    tokens = tokenizer.tokenize(text)
 
    print(f"\n   Algorithm     : Byte-Pair Encoding (BPE)")
    print(f"   Vocab size    : {tokenizer.vocab_size}")
    print(f"   Token count   : {len(tokens)}")
    print("\n   Index | Token")
    print("   ------+--------------------")
    for i, t in enumerate(tokens):
        print(f"   {i:>5} | {t!r}")
    print("\n   NOTE: 'G-with-dot' prefix marks a leading space in GPT-2 BPE.")
    return tokens
 
 
# ======================================================= STAGE 3 : TOKEN IDs
def stage_token_ids(tokenizer, tokens):
    banner(3, "TOKEN IDs")
    ids = tokenizer.convert_tokens_to_ids(tokens)
 
    print("\n   Each token is mapped to an integer index in the vocabulary.\n")
    print("   Token                | ID")
    print("   ---------------------+--------")
    for t, i in zip(tokens, ids):
        print(f"   {repr(t):<20} | {i}")
 
    print(f"\n   Input ID vector: {ids}")
    print(f"   Decoded back   : {tokenizer.decode(ids)!r}")
    return torch.tensor([ids])
 
 
# ====================================================== STAGE 4 : EMBEDDINGS
def stage_embeddings(model, input_ids, tokens):
    banner(4, "EMBEDDINGS")
 
    with torch.no_grad():
        tok_emb = model.transformer.wte(input_ids)
        pos_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
        pos_emb = model.transformer.wpe(pos_ids)
        embeddings = tok_emb + pos_emb
 
    print(f"\n   Token embedding      wte : {tuple(tok_emb.shape)}")
    print(f"   Positional embedding wpe : {tuple(pos_emb.shape)}")
    print(f"   Final input          sum : {tuple(embeddings.shape)}")
    print("   (batch, sequence_length, hidden_dim=768)")
 
    print("\n   First 6 of 768 dimensions per token:\n")
    print("   Token                | Vector (truncated)")
    print("   ---------------------+---------------------------------------")
    for i, t in enumerate(tokens):
        vals = " ".join(f"{v:+.3f}" for v in embeddings[0, i, :6].tolist())
        print(f"   {repr(t):<20} | [{vals} ...]")
 
    print("\n   Positional embeddings are ADDED so the model knows word order.")
    return embeddings
 
 
# ===================================================== STAGE 5 : TRANSFORMER
def stage_transformer(model, tokenizer, input_ids, tokens):
    banner(5, "TRANSFORMER  (Attention + Feed Forward, x12)")
 
    with torch.no_grad():
        out = model(input_ids)
 
    n_layers = model.config.n_layer
    n_heads  = model.config.n_head
 
    print(f"\n   Layers (blocks)        : {n_layers}")
    print(f"   Attention heads/layer  : {n_heads}")
    print(f"   Hidden dimension       : {model.config.n_embd}")
    print(f"   Feed-forward inner dim : {4 * model.config.n_embd}")
    print(f"   Attention tensor shape : {tuple(out.attentions[0].shape)}")
    print("   (batch, heads, query_tokens, key_tokens)")
 
    print("\n   Hidden state evolution (norm of last token's vector):\n")
    print("   Layer | Vector norm")
    print("   ------+-------------")
    for L in range(0, n_layers + 1, max(1, n_layers // 6)):
        norm = out.hidden_states[L][0, -1].norm().item()
        tag = "input" if L == 0 else f"block {L}"
        print(f"   {tag:<5} | {norm:>10.3f}")
 
    print("\n   ATTENTION: where the LAST token looks (final layer, head 0):\n")
    attn = out.attentions[-1][0, 0, -1]
    for t, a in zip(tokens, attn.tolist()):
        bar = "#" * int(a * 40)
        print(f"   {repr(t):<20} {a:5.3f} |{bar}")
 
    print("\n   Rows sum to 1.0 (softmax). Causal mask blocks future tokens.")
    return out
 
 
# ============================================== STAGE 6 : OUTPUT PROBABILITIES
def stage_probabilities(tokenizer, out):
    banner(6, "OUTPUT PROBABILITIES")
 
    logits = out.logits[0, -1]
    probs = F.softmax(logits, dim=-1)
 
    print(f"\n   Logits vector shape : {tuple(logits.shape)}  (one score per vocab word)")
    print(f"   After softmax, sum  : {probs.sum().item():.6f}")
 
    top = torch.topk(probs, 10)
    print("\n   TOP 10 CANDIDATES FOR THE NEXT TOKEN:\n")
    print("   Rank | Token           |   Prob  | Bar")
    print("   -----+-----------------+---------+" + "-" * 32)
    for r, (p, idx) in enumerate(zip(top.values.tolist(), top.indices.tolist()), 1):
        tok = tokenizer.decode([idx])
        bar = "#" * int(p * 40)
        print(f"   {r:>4} | {repr(tok):<15} | {p*100:6.2f}% | {bar}")
 
    best = top.indices[0].item()
    print(f"\n   Highest probability token: {tokenizer.decode([best])!r}")
    return probs
 
 
# ========================================= STAGE 7 : NEXT TOKEN + REPEAT LOOP
def stage_generation(tokenizer, model, input_ids):
    banner(7, "NEXT TOKEN  ->  REPEAT PROCESS  (autoregressive loop)")
 
    print(f"\n   Mode        : {'GREEDY (argmax)' if GREEDY else 'SAMPLING'}")
    print(f"   Temperature : {TEMPERATURE}")
    print(f"   Top-k       : {TOP_K}")
    print(f"   Max tokens  : {MAX_NEW_TOKENS}\n")
 
    print("   Step | New token        | Prob   | Text so far")
    print("   -----+------------------+--------+" + "-" * 34)
 
    generated = input_ids.clone()
 
    for step in range(1, MAX_NEW_TOKENS + 1):
        with torch.no_grad():
            logits = model(generated).logits[0, -1]
 
        if GREEDY:
            probs = F.softmax(logits, dim=-1)
            next_id = torch.argmax(probs).view(1)
        else:
            logits = logits / TEMPERATURE
            kth = torch.topk(logits, TOP_K).values[-1]
            logits[logits < kth] = -float("inf")     # top-k filtering
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
 
        p = probs[next_id].item()
        generated = torch.cat([generated, next_id.unsqueeze(0)], dim=1)
 
        piece = tokenizer.decode(next_id)
        preview = tokenizer.decode(generated[0])
        if len(preview) > 32:
            preview = "..." + preview[-29:]
        print(f"   {step:>4} | {repr(piece):<16} | {p*100:5.1f}% | {preview}")
 
        if next_id.item() == tokenizer.eos_token_id:
            print("\n   End-of-sequence token reached. Loop stops.")
            break
 
    print("\n   Each new token was APPENDED to the input and fed back in.")
    print("   That feedback arrow is the whole of 'Repeat process'.")
    return generated
 
 
# ============================================== STAGE 8 : FINAL GENERATED TEXT
def stage_final(tokenizer, generated, original):
    banner(8, "GENERATED TEXT")
    full = tokenizer.decode(generated[0])
    print(f"\n   Your input     : {original}")
    print(f"   Model continued: {full[len(original):]}")
    print(f"\n   FULL OUTPUT:\n\n   {full}\n")
    print(f"   Total tokens in final sequence: {generated.shape[1]}")
 
 
# ==================================================================== DRIVER
def main():
    print(LINE)
    print("        LARGE LANGUAGE MODEL - STAGE BY STAGE PIPELINE DEMO")
    print(LINE)
 
    tokenizer, model = load_model()
 
    while True:
        text       = stage_user_input();                       pause()
        tokens     = stage_tokenizer(tokenizer, text);         pause()
        input_ids  = stage_token_ids(tokenizer, tokens);       pause()
        stage_embeddings(model, input_ids, tokens);            pause()
        out        = stage_transformer(model, tokenizer,
                                       input_ids, tokens);     pause()
        stage_probabilities(tokenizer, out);                   pause()
        generated  = stage_generation(tokenizer, model, input_ids)
        stage_final(tokenizer, generated, text)
 
        print("\n" + LINE)
        again = input("  Run again with a new sentence? (y/n): ").strip().lower()
        if again != "y":
            print("\n  Done. Exiting.\n")
            break
 
 
if __name__ == "__main__":
    main()
 