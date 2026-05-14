# Encoder-Decoder Transformers

## Encoder

### Self-Attention

Input: `I = [t × d]`

Weight matrices — all `[d × d]`:

```
Q = I·Wq    [t × d]   — what am I looking for?
K = I·Wk    [t × d]   — what do I contain?
V = I·Wv    [t × d]   — what do I actually emit if attended to?
```

**Formula:**
```
Attention(Q, K, V) = softmax(Q·Kᵀ / √d) · V
```

**Step 1:** `U = Q·Kᵀ` → `[t × t]`  
`U[r, c]` = how much token `r` wants to attend to token `c`. Correlation matrix. Diagonal should be high. For causal (decoder) masking, zero out the upper triangle.

**Step 2:** `A = softmax(U / √d)` → `[t × t]`  
Softmax taken row-wise — each row sums to 1. Row `r` = "given token `r`, how much attention weight goes to each other token."  
`√d` prevents dot products from getting too large before softmax.

**Step 3:** `Output = A·V` → `[t × d]`  
Each output row `r` is a weighted sum of all value vectors. Token `r`'s new representation, after pulling in information from all other tokens according to attention weights.

**Summary:**
- Q/K compute where to look
- V computes what to take from there
- Output shape `[t × d]` — same as input, ready for next layer

---

### MLP

Applied per-token, identically, with the same weights across all tokens.
For compute they can be combined as a matrix

```
x → Linear(d, 4d) → ReLU/GELU → Linear(4d, d) → output
```

Parameters:
- `W1: [d × 4d]`, `b1: [4d]`
- `W2: [4d × d]`, `b2: [d]`

Single sequence: `[1 × d] · [d × 4d]` → `[t × 4d]` → `[1 × d]`
Full sequence: `[t × d] · [d × 4d]` → `[t × 4d]` → `[t × d]`

Same `W1, W2` applied to every token — logically independent, physically a batched matmul. Trivially parallelizable.

`4d` hidden dim is standard — from the original paper, kept in most models.

**Summary**:
- Dimension stays the same
- MLP is where each token processes individually what it's learnt

--- 

### Encoder Summary 
- Embeddings + Positional Encoding 
- Self Attention - relating tokens
- MLP - each token processes
- Step 2 and 3 repeated N times 
- Ouput [t x d] rich contextual vector per input token. Each vector now knows about the full sentence. 
