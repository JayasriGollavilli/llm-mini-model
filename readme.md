# GPT-2 Mini Model Pipeline

An educational and modular implementation of the complete GPT-2 inference pipeline:
**Tokenization $\rightarrow$ Embeddings $\rightarrow$ Transformer $\rightarrow$ Text Generation**.

---

## Architecture & Workflow

```
[Raw User Text]
       │
       ▼
1. Tokenization (models/tokenization.py)
   ├── Byte-Pair Encoding (BPE)
   └── Token to Token ID vocabulary mapping (vocab size: 50,257)
       │
       ▼
2. Embeddings (models/embeddings.py)
   ├── Token Embeddings (wte: 50,257 x 768)
   ├── Position Embeddings (wpe: 1,024 x 768)
   └── Summed Input Embeddings (1 x seq_len x 768)
       │
       ▼
3. Transformer Calculation (models/transformer.py)
   ├── 12 Transformer Blocks
   │   ├── Layer Normalization 1 (ln_1)
   │   ├── Multi-Head Self-Attention (12 heads, causal mask)
   │   ├── Residual Connection (x + attn)
   │   ├── Layer Normalization 2 (ln_2)
   │   ├── Feed-Forward Network / MLP (768 -> 3072 -> 768)
   │   └── Residual Connection (x + ffn)
   ├── Final Layer Normalization (ln_f)
   └── Contextual Hidden States (1 x seq_len x 768)
       │
       ▼
4. Text Generation (inference/text_generation.py)
   ├── LM Head Projection (768 -> 50,257 logits)
   ├── Softmax Probability Distribution
   ├── Candidate Token Selection (Greedy / Top-k Sampling)
   └── Autoregressive Decoding Loop -> Final Generated Text
```

---

## Project Structure

```
llm-mini-model/
│
├── common/
│   ├── __init__.py
│   └── config.py               # Model name, token limits, sampling parameters
│
├── models/
│   ├── __init__.py
│   ├── tokenization.py         # Step 1: BPE tokenization and vocabulary lookup
│   ├── embeddings.py           # Step 2: Token (wte) & position (wpe) embeddings
│   └── transformer.py          # Step 3: 12-block Transformer & custom PyTorch block
│
├── inference/
│   ├── __init__.py
│   └── text_generation.py      # Step 4: LM head projection & autoregressive loop
│
├── pipeline/
│   ├── __init__.py
│   └── main.py                 # Pipeline orchestrator
│
├── main.py                     # Root CLI entry point
├── requirements.txt            # Python dependencies (torch, transformers, etc.)
└── README.md                   # Documentation
```

---

## How to Run

### 1. Run Complete Pipeline (Default Prompt)
```bash
python main.py
```
or
```bash
python pipeline/main.py
```

### 2. Run Complete Pipeline with a Custom Prompt
```bash
python main.py --text "Artificial intelligence is"
```

### 3. Run Individual Stages Independently
Each stage script can also be executed standalone to inspect that specific layer:
```bash
# Step 1: Tokenization
python models/tokenization.py

# Step 2: Embeddings
python models/embeddings.py

# Step 3: Transformer Block Unit Test
python models/transformer.py

# Step 4: Text Generation
python inference/text_generation.py
```

---

## Pipeline Stages Explained

### Step 1: Tokenization (`models/tokenization.py`)
Converts raw string input into GPT-2 Byte-Pair Encoding (BPE) tokens. Each token represents a subword unit, and the `Ġ` prefix denotes a preceding whitespace. Each token is mapped to a vocabulary index ($0 \le \text{ID} < 50,257$).

### Step 2: Token + Position Embeddings (`models/embeddings.py`)
Maps discrete token IDs into 768-dimensional vector space:
- **Token Embeddings ($W_{te}$)**: Retrieves token semantic representations.
- **Position Embeddings ($W_{pe}$)**: Injects positional coordinates ($0, 1, \dots, N-1$) so the model understands token order.
- **Combined Embeddings**: $E = W_{te}(\text{tokens}) + W_{pe}(\text{positions})$.

### Step 3: Transformer Calculations (`models/transformer.py`)
Passes input embeddings through the 12 Transformer layers:
- Multi-Head Self-Attention with 12 parallel attention heads ($d_k = 64$).
- Causal triangular attention mask enforcing autoregressive property (tokens cannot attend to future tokens).
- 2-layer Feed-Forward Network ($768 \rightarrow 3072 \rightarrow 768$) with GELU activation.
- Pre-layer normalization and residual skip connections.
- Outputs enriched contextual representations of shape `[1, sequence_length, 768]`.

### Step 4: Text Generation (`inference/text_generation.py`)
Projects the transformer's last hidden state back to the vocabulary dimension ($768 \rightarrow 50,257$) using the tied language model head. Computes softmax probabilities, selects the next token, appends it to the sequence, and repeats autoregressively until completion.
