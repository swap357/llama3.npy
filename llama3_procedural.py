import numpy as np
import math
import time
import sys
import logging
import numba
import config as model_config
import tokenizer as model_tokenizer
import utils as model_utils
import os

numba.config.NUMBA_DEBUG_PRINT_AFTER_COMPILE = False
logger = logging.getLogger(__name__)

USE_FLOAT32 = True

# Layer-specific caches for keys and values
k_caches = None
v_caches = None

def softmax(x):
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / (np.sum(exp_x, axis=-1, keepdims=True) + 1e-10)

@numba.njit
def softmax_jit(x):
    x_max = np.max(x, axis=1).reshape(-1, 1)
    e_x = np.exp(x - x_max)
    return e_x / (np.sum(e_x, axis=1, keepdims=True) + 1e-10)

def silu(x):
    x = np.clip(x, -88.0, 88.0)
    return x * (1.0 / (1.0 + np.exp(-x)))

@numba.njit
def silu_jit(x):
    result = np.zeros_like(x)
    for i in range(x.size):
        flat_idx = i
        val = x.flat[flat_idx]
        val_float32 = float(val)
        val_float32 = max(-88.0, min(88.0, val_float32))
        result.flat[flat_idx] = val_float32 * (1.0 / (1.0 + np.exp(-val_float32)))
    return result

def compute_cos_sin_cache(head_dim, max_seq_len, base=10000):
    inv_freq = 1.0 / (base ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    t = np.arange(max_seq_len, dtype=np.float32)
    freqs = np.outer(t, inv_freq)
    return np.cos(freqs).astype(model_config.NP_DTYPE), np.sin(freqs).astype(model_config.NP_DTYPE)

@numba.njit
def layer_norm_jit(x, weight, eps):
    var = np.mean(x ** 2, axis=1, keepdims=True)
    norm = x / np.sqrt(var + eps)
    return norm * weight

def apply_rotary_emb(xq, xk, cos, sin):
    def rotate(x):
        x_r, x_i = x[..., ::2], x[..., 1::2]
        return np.stack([x_r * cos - x_i * sin, x_r * sin + x_i * cos], axis=-1).reshape(x.shape)
    cos, sin = np.expand_dims(cos, (0, 2)), np.expand_dims(sin, (0, 2))
    return rotate(xq), rotate(xk)

def rms_norm(x, weight, eps, use_jit):
    try:
        if use_jit and x.ndim == 2 and x.dtype == model_config.NP_DTYPE:
            return layer_norm_jit(x, weight, eps)
    except:
        pass
    return x / np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps) * weight

def initialize_kv_caches(max_batch_size, max_seq_len, n_kv_heads, head_dim, n_layers):
    global k_caches, v_caches
    k_caches = [np.zeros((max_batch_size, max_seq_len, n_kv_heads, head_dim), dtype=model_config.NP_DTYPE) for _ in range(n_layers)]
    v_caches = [np.zeros((max_batch_size, max_seq_len, n_kv_heads, head_dim), dtype=model_config.NP_DTYPE) for _ in range(n_layers)]

def attention(x, q_weight, k_weight, v_weight, o_weight, start_pos, mask, cos, sin, 
              n_heads, n_kv_heads, head_dim, scale, layer_idx):
    global k_caches, v_caches
    
    b, t = x.shape[:2]
    xq, xk, xv = x @ q_weight.T, x @ k_weight.T, x @ v_weight.T
    
    xq = xq.reshape(b, t, n_heads, head_dim)
    xk = xk.reshape(b, t, n_kv_heads, head_dim)
    xv = xv.reshape(b, t, n_kv_heads, head_dim)

    xq, xk = apply_rotary_emb(xq, xk, cos, sin)

    # Use the appropriate layer's KV cache
    k_caches[layer_idx][:, start_pos:start_pos + t] = xk
    v_caches[layer_idx][:, start_pos:start_pos + t] = xv
    k_seq = k_caches[layer_idx][:, :start_pos + t]
    v_seq = v_caches[layer_idx][:, :start_pos + t]
    
    if n_heads > n_kv_heads:
        rep = n_heads // n_kv_heads
        k_seq = np.repeat(k_seq, rep, axis=2)
        v_seq = np.repeat(v_seq, rep, axis=2)
    
    xq, k_seq, v_seq = xq.transpose(0, 2, 1, 3), k_seq.transpose(0, 2, 1, 3), v_seq.transpose(0, 2, 1, 3)
    scores = np.einsum('bhqd,bhkd->bhqk', xq, k_seq) * scale

    if mask is not None:
        scores += mask[None, None, :, :]

    attn = softmax(scores)
    out = np.einsum('bhqk,bhkd->bhqd', attn, v_seq)
    final_out = (out.transpose(0, 2, 1, 3).reshape(b, t, -1)) @ o_weight.T

    return final_out

def feed_forward(x, up_weight, gate_weight, down_weight, use_jit):
    up, gate, down = up_weight.T, gate_weight.T, down_weight.T
    gate_proj = x @ gate
    swish = silu_jit(gate_proj) if use_jit and gate_proj.size > 32768 else silu(gate_proj)
    up_proj = x @ up
    ffn_out = (swish * up_proj) @ down
    return ffn_out

def transformer_block(x, layer_weights, pos, mask, cos, sin, use_jit, n_heads, n_kv_heads, head_dim, norm_eps, layer_idx):
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
    
    norm_x = rms_norm(x, ln1_weight, norm_eps, use_jit)
    scale = 1.0 / math.sqrt(head_dim)
    
    attn_out = attention(norm_x, q_weight, k_weight, v_weight, o_weight, 
                         pos, mask, cos, sin, n_heads, n_kv_heads, head_dim, scale, layer_idx)
    
    h = x + attn_out
    norm_h = rms_norm(h, ln2_weight, norm_eps, use_jit)
    ffn_out = feed_forward(norm_h, up_weight, gate_weight, down_weight, use_jit)
    block_output = h + ffn_out

    return block_output

def forward(input_ids, pos, weights, args, cos, sin, use_jit):
    b, t = input_ids.shape
    embed = weights["model.embed_tokens.weight"]
    lm_head = weights["lm_head.weight"].T if "lm_head.weight" in weights else weights["model.embed_tokens.weight"].T
    norm_weight = weights["model.norm.weight"]
    
    hidden = embed[input_ids]
    
    mask = None
    if t > 1:
        mask = np.triu(np.full((t, t), float("-inf"), dtype=model_config.NP_DTYPE), 1)

    cos_t, sin_t = cos[pos:pos + t], sin[pos:pos + t]

    for i in range(args.n_layers):
        hidden = transformer_block(
            hidden, weights, pos, mask, cos_t, sin_t, use_jit, 
            args.n_heads, args.n_kv_heads or args.n_heads, args.dim // args.n_heads,
            args.norm_eps, i
        )

    final_norm = rms_norm(hidden, norm_weight, args.norm_eps, use_jit)
    logits = final_norm[:, [-1], :] @ lm_head
    
    return logits

def generate(input_ids, max_new_tokens, weights, args, cos, sin, use_jit):
    b, t = input_ids.shape
    np.random.seed(args.seed)
    generated_ids = []
    current_input_ids = input_ids.copy()
    
    for i in range(max_new_tokens):
        pos = 0 if i == 0 else t + i - 1
        inputs_this_step = current_input_ids if i == 0 else current_input_ids[:, [-1]]
        
        logits = forward(inputs_this_step, pos, weights, args, cos, sin, use_jit)
        
        if args.do_sample:
            logits_last = logits[:, -1, :] / args.temperature
            probs = softmax_jit(logits_last) if use_jit else softmax(logits_last)
            next_token = np.random.choice(probs.shape[-1], p=probs[0])
            next_id = np.array([[next_token]], dtype=current_input_ids.dtype)
        else:
            logits_last = logits[:, -1, :]
            next_id = np.argmax(logits_last, axis=-1).reshape(b, 1)
            
        current_input_ids = np.concatenate([current_input_ids, next_id], axis=1)
        generated_ids.append(next_id[0, 0].item())
        
        yield next_id

def initialize_model(path, args, use_jit=False):
    weights = model_utils.load_parameters(path)
    
    if "lm_head.weight" in weights:
        logger.info("Using separate lm_head weights.")
    else:
        logger.info("Using shared embedding weights for lm_head.")
    
    cos, sin = compute_cos_sin_cache(args.dim // args.n_heads, args.max_seq_len)
    
    # Initialize KV caches for each layer
    head_dim = args.dim // args.n_heads
    n_kv_heads = args.n_kv_heads or args.n_heads
    initialize_kv_caches(args.max_batch_size, args.max_seq_len, n_kv_heads, head_dim, args.n_layers)
    
    return weights, cos, sin

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default="Once upon a time")
    parser.add_argument('--use-jit', action='store_true')
    args_cli = parser.parse_args()

    args = model_config.ModelArgs()
    tokenizer = model_tokenizer.Tokenizer(model_config.NP_TOKENIZER_PATH)
    
    weights, cos, sin = initialize_model(model_config.NP_MODEL_PATH, args, use_jit=args_cli.use_jit)
    input_ids = np.array([tokenizer.encode(args_cli.prompt)])

    print(f"\n{args_cli.prompt}", end="")
    start = time.time()
    token_count = input_ids.shape[1]
    
    for token in generate(input_ids, args.max_new_tokens, weights, args, cos, sin, args_cli.use_jit):
        token_count += 1
        tok_id = token[0, 0].item()
        if tok_id in [tokenizer.eos_id, tokenizer.bos_id]:
            break
        print(tokenizer.decode([tok_id]), end="")
        sys.stdout.flush()
        
    elapsed = time.time() - start
    print(f"\n\nToken count: {token_count}, elapsed: {elapsed:.2f}s, {round(token_count / elapsed)} tokens/s") 