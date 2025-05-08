import numpy as np
import math
import time
import sys
import os
import logging

# Add repo root to path to import from root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config as model_config
import tokenizer as model_tokenizer
import utils as model_utils

logger = logging.getLogger(__name__)

USE_FLOAT32 = True

def softmax(x):
    """
    Computes softmax along the last dimension.

    Formula: softmax(x_i) = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))

    Where:
    - i: specific index position in the input array being calculated
    - j: iterator across all indices in the same dimension as i in the denominator sum
    - max(x): maximum value in the array along the softmax dimension

    Shape:
        x: (..., n)
        output: (..., n) with values in range (0, 1) summing to 1 along last dimension
    """
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / (np.sum(exp_x, axis=-1, keepdims=True) + 1e-10)


def silu(x):
    """
    SiLU (Sigmoid Linear Unit) activation function.

    Formula: silu(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x)))

    Shape:
        x: (any shape)
        output: same shape as input
    """
    x = np.clip(x, -88.0, 88.0)
    return x * (1.0 / (1.0 + np.exp(-x)))


def compute_cos_sin_cache(head_dim, max_seq_len, base=10000):
    """
    Precomputes cosine and sine values for rotary position embeddings.

    Args:
        head_dim: Dimension of each attention head
        max_seq_len: Maximum sequence length to precompute
        base: Base value for the inverse frequency calculations (rope_theta)

    Returns:
        cos, sin: Precomputed cosine and sine tensors for rotary embeddings
    """
    # Create inverse frequency tensor
    inv_freq = 1.0 / (base ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))

    # Create position tensor for each position up to max_seq_len
    t = np.arange(max_seq_len, dtype=np.float32)

    # Compute freqs = t * inv_freq (for each position and dim)
    freqs = np.outer(t, inv_freq)

    # Compute cos and sin
    cos = np.cos(freqs).astype(model_config.NP_DTYPE)
    sin = np.sin(freqs).astype(model_config.NP_DTYPE)

    return cos, sin

def apply_rotary_emb(q, k, cos, sin, unsqueeze_dim=1):
    """
    Applies rotary positional embeddings to the query and key tensors.

    Args:
        q: Query tensor of shape (..., head_dim)
        k: Key tensor of shape (..., head_dim) 
        cos: Cosine part of the rotary embedding
        sin: Sine part of the rotary embedding
        unsqueeze_dim: Dimension along which to unsqueeze cos and sin

    Returns:
        q_embed, k_embed: Rotated query and key tensors with same shape as input
    """
    # Extract even and odd dimensions
    q_even = q[..., ::2]
    q_odd = q[..., 1::2]
    k_even = k[..., ::2]
    k_odd = k[..., 1::2]

    # Handle the specific case for our broadcasting pattern
    if unsqueeze_dim == 0 and len(cos.shape) == 3:  # (batch_size, seq_len, head_dim/2)
        # Reshape for (batch_size, seq_len, n_heads, head_dim)
        cos = cos[:, :, None, :]
        sin = sin[:, :, None, :]
    else:
        # Default unsqueeze for other cases
        cos = np.expand_dims(cos, axis=unsqueeze_dim)
        sin = np.expand_dims(sin, axis=unsqueeze_dim)

    # Apply rotation
    q_rot_even = q_even * cos - q_odd * sin
    q_rot_odd = q_even * sin + q_odd * cos
    k_rot_even = k_even * cos - k_odd * sin
    k_rot_odd = k_even * sin + k_odd * cos

    # Recombine rotated values
    q_embed = np.zeros_like(q)
    k_embed = np.zeros_like(k)
    q_embed[..., ::2] = q_rot_even
    q_embed[..., 1::2] = q_rot_odd
    k_embed[..., ::2] = k_rot_even
    k_embed[..., 1::2] = k_rot_odd

    return q_embed, k_embed

def rms_norm(x, weight, eps):
    """
    Root Mean Square (RMS) Layer Normalization.

    Formula: norm(x) = x / sqrt(mean(x²) + eps) * weight

    Shape:
        x: (batch_size, seq_len, dim) or (batch_size, dim)
        weight: (dim,)
        output: same shape as x
    """
    return x / np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps) * weight

def attention(x, q_weight, k_weight, v_weight, o_weight, mask, cos, sin,
              n_heads, n_kv_heads, head_dim, scale, layer_idx):
    """
    Multi-head attention with grouped-query attention (GQA).

    Formula:
        Q = X·W_q, K = X·W_k, V = X·W_v
        Attention(Q, K, V) = softmax(Q·K^T/√d_k)·V
        Output = Attention(Q, K, V)·W_o

    Shape:
        x: (batch_size, seq_len, dim)
        q_weight: (dim, n_heads * head_dim)
        k_weight: (dim, n_kv_heads * head_dim)
        v_weight: (dim, n_kv_heads * head_dim)
        o_weight: (n_heads * head_dim, dim)
        cos, sin: (seq_len, head_dim/2)
        output: (batch_size, seq_len, dim)
    """
    b, t = x.shape[:2]
    xq, xk, xv = x @ q_weight.T, x @ k_weight.T, x @ v_weight.T

    xq = xq.reshape(b, t, n_heads, head_dim)
    xk = xk.reshape(b, t, n_kv_heads, head_dim)
    xv = xv.reshape(b, t, n_kv_heads, head_dim)

    # Apply rotary embeddings to queries and keys
    xq, xk = apply_rotary_emb(xq, xk, cos, sin, unsqueeze_dim=0)

    if n_heads > n_kv_heads:
        rep = n_heads // n_kv_heads
        xk = np.repeat(xk, rep, axis=2)
        xv = np.repeat(xv, rep, axis=2)

    xq, xk, xv = xq.transpose(0, 2, 1, 3), xk.transpose(0, 2, 1, 3), xv.transpose(0, 2, 1, 3)
    scores = np.einsum('bhqd,bhkd->bhqk', xq, xk) * scale

    if mask is not None:
        scores += mask[None, None, :, :]

    attn = softmax(scores)
    out = np.einsum('bhqk,bhkd->bhqd', attn, xv)
    final_out = (out.transpose(0, 2, 1, 3).reshape(b, t, -1)) @ o_weight.T

    return final_out

def feed_forward(x, up_weight, gate_weight, down_weight):
    """
    SwiGLU feed-forward network.

    Formula:
        FFN(x) = (SiLU(x·W_gate) ⊙ (x·W_up))·W_down
        where ⊙ is element-wise multiplication

    Shape:
        x: (batch_size, seq_len, dim)
        up_weight: (dim, 4*dim)
        gate_weight: (dim, 4*dim)
        down_weight: (4*dim, dim)
        output: (batch_size, seq_len, dim)
    """
    up, gate, down = up_weight.T, gate_weight.T, down_weight.T
    gate_proj = x @ gate
    swish = silu(gate_proj)
    up_proj = x @ up
    ffn_out = (swish * up_proj) @ down
    return ffn_out

def transformer_block(x, layer_weights, mask, cos, sin, n_heads, n_kv_heads, head_dim, norm_eps, layer_idx):
    """
    Single transformer layer with attention and feed-forward network.

    Formula:
        h = x + Attention(RMSNorm(x))
        out = h + FFN(RMSNorm(h))

    Shape:
        x: (batch_size, seq_len, dim)
        layer_weights: dict containing all layer parameters
        cos, sin: (seq_len, head_dim/2)
        output: (batch_size, seq_len, dim)
    """
    prefix = f"model.layers.{layer_idx}."

    q_weight = layer_weights[prefix + "self_attn.q_proj.weight"]
    k_weight = layer_weights[prefix + "self_attn.k_proj.weight"]
    v_weight = layer_weights[prefix + "self_attn.v_proj.weight"]
    o_weight = layer_weights[prefix + "self_attn.o_proj.weight"]

    up_weight = layer_weights[prefix + "mlp.up_proj.weight"]
    gate_weight = layer_weights[prefix + "mlp.gate_proj.weight"]
    down_weight = layer_weights[prefix + "mlp.down_proj.weight"]

    ln1_weight = layer_weights[prefix + "input_layernorm.weight"]
    ln2_weight = layer_weights[prefix + "post_attention_layernorm.weight"]

    norm_x = rms_norm(x, ln1_weight, norm_eps)
    scale = 1.0 / math.sqrt(head_dim)

    attn_out = attention(norm_x, q_weight, k_weight, v_weight, o_weight,
                         mask, cos, sin, n_heads, n_kv_heads, head_dim, scale, layer_idx)

    h = x + attn_out
    norm_h = rms_norm(h, ln2_weight, norm_eps)
    ffn_out = feed_forward(norm_h, up_weight, gate_weight, down_weight)
    block_output = h + ffn_out

    return block_output

def forward(input_ids, weights, args, cos, sin, position_ids=None):
    """
    Forward pass through the entire model.

    Shape:
        input_ids: (batch_size, seq_len)
        weights: dict containing all model parameters
        cos, sin: Precomputed cosine and sine for position embeddings
        position_ids: Optional position IDs, defaults to incremental positions
        output: logits with shape (batch_size, 1, vocab_size)
    """
    b, t = input_ids.shape
    embed = weights["model.embed_tokens.weight"]
    lm_head = weights["lm_head.weight"].T if "lm_head.weight" in weights else weights["model.embed_tokens.weight"].T
    norm_weight = weights["model.norm.weight"]

    hidden = embed[input_ids]

    # Set up causal mask
    mask = None
    if t > 1:
        mask = np.triu(np.full((t, t), float("-inf"), dtype=model_config.NP_DTYPE), 1)

    # Create position IDs if not provided
    if position_ids is None:
        position_ids = np.arange(t, dtype=np.int64)[np.newaxis, :]
    
    # Get position embeddings for the current sequence length
    # Ensure position_ids are within bounds of cos/sin
    max_pos = cos.shape[0]
    safe_pos_ids = np.minimum(position_ids, max_pos - 1)
    
    # Get the cos/sin values for the current positions
    cos_t = cos[safe_pos_ids]
    sin_t = sin[safe_pos_ids]

    for i in range(args.n_layers):
        hidden = transformer_block(
            hidden, weights, mask, cos_t, sin_t,
            args.n_heads, args.n_kv_heads or args.n_heads, args.dim // args.n_heads,
            args.norm_eps, i
        )

    final_norm = rms_norm(hidden, norm_weight, args.norm_eps)
    logits = final_norm[:, [-1], :] @ lm_head

    return logits

def generate(input_ids, max_new_tokens, weights, args, cos, sin):
    """
    Autoregressive token generation.

    Shape:
        input_ids: (batch_size, seq_len)
        output: yields next token IDs one at a time
    """
    b, t = input_ids.shape
    np.random.seed(args.seed)
    generated_ids = []
    current_input_ids = input_ids.copy()
    seq_length = t

    for i in range(max_new_tokens):
        # Create position IDs for the current sequence
        position_ids = np.arange(seq_length, dtype=np.int64)[np.newaxis, :]

        # Get logits from forward pass
        logits = forward(current_input_ids, weights, args, cos, sin, position_ids)

        if args.do_sample:
            logits_last = logits[:, -1, :] / args.temperature
            probs = softmax(logits_last)
            next_token = np.random.choice(probs.shape[-1], p=probs[0])
            next_id = np.array([[next_token]], dtype=current_input_ids.dtype)
        else:
            logits_last = logits[:, -1, :]
            next_id = np.argmax(logits_last, axis=-1).reshape(b, 1)

        current_input_ids = np.concatenate([current_input_ids, next_id], axis=1)
        generated_ids.append(next_id[0, 0].item())
        seq_length += 1

        yield next_id

def initialize_model(path, args):
    """
    Initializes the model by loading weights and precomputing position embeddings.

    Args:
        path: Path to model weights file
        args: Model configuration arguments

    Returns:
        weights: dict of model parameters
        cos, sin: arrays for rotary embeddings
    """
    weights = model_utils.load_parameters(path)

    # Set up rotary embeddings using theta from config
    head_dim = args.dim // args.n_heads
    base = getattr(args, "rope_theta", 10000.0)

    cos, sin = compute_cos_sin_cache(
        head_dim=head_dim, 
        max_seq_len=args.max_seq_len,
        base=base
    )

    return weights, cos, sin

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default="Once upon a time")
    parser.add_argument('--max-new-tokens', type=int, help="Override max new tokens from config")
    args_cli = parser.parse_args()

    args = model_config.ModelArgs()
    # Override max_new_tokens if provided via CLI
    if args_cli.max_new_tokens is not None:
        args.max_new_tokens = args_cli.max_new_tokens

    tokenizer = model_tokenizer.Tokenizer(model_config.NP_TOKENIZER_PATH)

    weights, cos, sin = initialize_model(model_config.NP_MODEL_PATH, args)
    input_ids = np.array([tokenizer.encode(args_cli.prompt)])

    print(f"\n{args_cli.prompt}", end="")
    start = time.time()
    token_count = input_ids.shape[1]

    for token in generate(input_ids, args.max_new_tokens, weights, args, cos, sin):
        token_count += 1
        tok_id = token[0, 0].item()
        if tok_id in [tokenizer.eos_id, tokenizer.bos_id]:
            break
        print(tokenizer.decode([tok_id]), end="")
        sys.stdout.flush()

    elapsed = time.time() - start
    print(f"\n\nToken count: {token_count}, elapsed: {elapsed:.2f}s, {round(token_count / elapsed)} tokens/s")
