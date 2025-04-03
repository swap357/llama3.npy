import numpy as np
import math
import time
import sys
import logging
import numba
import config as model_config
import tokenizer as model_tokenizer
import utils as model_utils

numba.config.NUMBA_DEBUG_PRINT_AFTER_COMPILE = False
logger = logging.getLogger(__name__)

USE_FLOAT32 = True

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
    """SiLU activation function optimized for JIT, compatible with float16"""
    result = np.zeros_like(x)
    for i in range(x.size):
        flat_idx = i
        val = x.flat[flat_idx]
        # Use float32 for intermediate calculations to avoid precision issues
        val_float32 = float(val)
        # Clip values to avoid overflow
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

class RMSNorm:
    def __init__(self, weight, eps): self.weight, self.eps = weight, eps
    def __call__(self, x, use_jit):
        try:
            if use_jit and x.ndim == 2 and x.dtype == model_config.NP_DTYPE:
                return layer_norm_jit(x, self.weight, self.eps)
        except: pass
        return x / np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps) * self.weight

class FeedForward:
    def __init__(self, up, gate, down):
        self.up, self.gate, self.down = up.T, gate.T, down.T
    def __call__(self, x, use_jit):
        gate_proj = x @ self.gate
        swish = silu_jit(gate_proj) if use_jit and gate_proj.size > 32768 else silu(gate_proj)
        return (swish * (x @ self.up)) @ self.down

class Attention:
    def __init__(self, q, k, v, o, args):
        self.n_heads, self.n_kv_heads, self.head_dim = args.n_heads, args.n_kv_heads or args.n_heads, args.dim // args.n_heads
        self.q, self.k, self.v, self.o = q.T, k.T, v.T, o.T
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.k_cache = np.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim), dtype=model_config.NP_DTYPE)
        self.v_cache = np.zeros_like(self.k_cache)

    def __call__(self, x, start_pos, mask, cos, sin):
        b, t = x.shape[:2]
        xq, xk, xv = x @ self.q, x @ self.k, x @ self.v
        for tensor in (xq, xk, xv):
            if np.any(np.isnan(tensor)) or np.any(np.isinf(tensor)):
                logger.warning("QKV contains NaN/Inf")
                tensor = np.nan_to_num(tensor)
        xq = xq.reshape(b, t, self.n_heads, self.head_dim)
        xk = xk.reshape(b, t, self.n_kv_heads, self.head_dim)
        xv = xv.reshape(b, t, self.n_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, cos, sin)
        self.k_cache[:, start_pos:start_pos + t] = xk
        self.v_cache[:, start_pos:start_pos + t] = xv
        k_seq = self.k_cache[:, :start_pos + t]
        v_seq = self.v_cache[:, :start_pos + t]
        if self.n_heads > self.n_kv_heads:
            rep = self.n_heads // self.n_kv_heads
            k_seq = np.repeat(k_seq, rep, axis=2)
            v_seq = np.repeat(v_seq, rep, axis=2)
        xq, k_seq, v_seq = xq.transpose(0, 2, 1, 3), k_seq.transpose(0, 2, 1, 3), v_seq.transpose(0, 2, 1, 3)
        scores = np.einsum('bhqd,bhkd->bhqk', xq, k_seq) * self.scale
        if mask is not None: scores += mask[None, None, :, :]
        attn = softmax(scores)
        if np.any(np.isnan(attn)) or np.any(np.isinf(attn)):
            logger.warning("Attention contains NaN/Inf")
            attn = np.nan_to_num(attn)
        out = np.einsum('bhqk,bhkd->bhqd', attn, v_seq)
        return (out.transpose(0, 2, 1, 3).reshape(b, t, -1)) @ self.o

class TransformerBlock:
    """
    Represents a single block of the Transformer architecture.

    Each transformer block consists of a self-attention mechanism followed by a
    feed-forward network (FFN). Layer normalization is applied before each
    sub-layer, and residual connections are used around each sub-layer.
    """
    def __init__(self, weights, i, args):
        """
        Initializes the TransformerBlock.

        Args:
            weights (dict): Dictionary containing the model weights.
            i (int): The index of the current layer.
            args (ModelArgs): Model configuration parameters.
        """
        prefix = f"model.layers.{i}."
        # Initialize the self-attention mechanism
        self.attn = Attention(weights[prefix + "self_attn.q_proj.weight"],
                              weights[prefix + "self_attn.k_proj.weight"],
                              weights[prefix + "self_attn.v_proj.weight"],
                              weights[prefix + "self_attn.o_proj.weight"], args)
        # Initialize the feed-forward network
        self.ffn = FeedForward(weights[prefix + "mlp.up_proj.weight"],
                               weights[prefix + "mlp.gate_proj.weight"],
                               weights[prefix + "mlp.down_proj.weight"])
        # Initialize the layer normalization before the attention block
        self.ln1 = RMSNorm(weights[prefix + "input_layernorm.weight"], args.norm_eps)
        # Initialize the layer normalization before the feed-forward network
        self.ln2 = RMSNorm(weights[prefix + "post_attention_layernorm.weight"], args.norm_eps)

    def __call__(self, x, pos, mask, cos, sin, use_jit):
        """
        Performs the forward pass through the TransformerBlock.

        Args:
            x (np.ndarray): Input tensor.
            pos (int): Current position in the sequence.
            mask (np.ndarray | None): Attention mask.
            cos (np.ndarray): Cosine values for rotary positional embeddings.
            sin (np.ndarray): Sine values for rotary positional embeddings.
            use_jit (bool): Flag to enable JIT optimization.

        Returns:
            np.ndarray: Output tensor after processing through the block.
        """
        # --- Attention Block ---
        # 1. Apply layer normalization (ln1)
        norm_x = self.ln1(x, use_jit)
        # 2. Compute self-attention (attn)
        attn_out = self.attn(norm_x, pos, mask, cos, sin)
        # 3. Add residual connection (x + ...)
        h = x + attn_out

        # --- Feed-Forward Block ---
        # 1. Apply layer normalization (ln2)
        norm_h = self.ln2(h, use_jit)
        # 2. Compute feed-forward (ffn)
        ffn_out = self.ffn(norm_h, use_jit)
        # 3. Add residual connection (h + ...)
        out = h + ffn_out
        return out

class NpLlama:
    def __init__(self, path, args, use_jit=False):
        self.args = args
        self.use_jit = use_jit
        weights = model_utils.load_parameters(path)
        self.embed = weights["model.embed_tokens.weight"]
        self.lm_head = self.embed.T
        self.cos, self.sin = compute_cos_sin_cache(args.dim // args.n_heads, args.max_seq_len)
        self.layers = [TransformerBlock(weights, i, args) for i in range(args.n_layers)]
        self.norm = RMSNorm(weights["model.norm.weight"], args.norm_eps)

    def forward(self, input_ids, pos):
        if input_ids.ndim == 1: input_ids = input_ids[None, :]
        b, t = input_ids.shape
        hidden = self.embed[input_ids]
        mask = None
        if t > 1:
            mask = np.triu(np.full((t, t), float("-inf"), dtype=model_config.NP_DTYPE), 1)
            if pos > 0:
                mask = np.concatenate([np.zeros((t, pos), dtype=model_config.NP_DTYPE), mask], axis=1)
        cos, sin = self.cos[pos:pos + t], self.sin[pos:pos + t]
        for layer in self.layers:
            hidden = layer(hidden, pos, mask, cos, sin, self.use_jit)
        return (self.norm(hidden, self.use_jit)[:, [-1], :] @ self.lm_head)

    def generate(self, input_ids, max_new_tokens):
        b, t = input_ids.shape
        np.random.seed(self.args.seed)
        last_tokens, generated = [], []
        for i in range(max_new_tokens):
            inputs = input_ids if i == 0 else next_id
            pos = 0 if i == 0 else t + i - 1
            logits = self.forward(inputs, pos)
            if self.args.do_sample:
                logits /= self.args.temperature
                if self.args.repetition_penalty != 1.0:
                    for tid in last_tokens[-5:]: logits[0, 0, tid] /= self.args.repetition_penalty
                probs = np.exp(logits - np.max(logits))
                probs /= np.sum(probs)
                next_token = np.random.choice(probs.shape[-1], p=probs[0, 0])
                next_id = np.array([[next_token]], dtype=input_ids.dtype)
            else:
                next_id = np.argmax(logits, axis=-1).reshape(b, 1)
            if next_id[0, 0] in [self.args.eos_token_id, self.args.bos_token_id]: break
            last_tokens.append(next_id[0, 0])
            generated.append(next_id[0, 0])
            yield next_id
            if len(generated) >= max_new_tokens: break

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default="Once upon a time")
    parser.add_argument('--use-jit', action='store_true', help='Enable JIT-optimized functions')
    args_cli = parser.parse_args()

    args = model_config.ModelArgs()
    tokenizer = model_tokenizer.Tokenizer(model_config.NP_TOKENIZER_PATH)
    model = NpLlama(model_config.NP_MODEL_PATH, args, use_jit=args_cli.use_jit)
    input_ids = np.array([tokenizer.encode(args_cli.prompt)])

    print(f"\n{args_cli.prompt}", end="")
    start = time.time()
    token_count = input_ids.shape[1]
    for token in model.generate(input_ids, args.max_new_tokens):
        token_count += 1
        tok_id = token[0, 0].item()
        if tok_id in [tokenizer.eos_id, tokenizer.bos_id]: break
        print(tokenizer.decode([tok_id]), end="")
        sys.stdout.flush()
    elapsed = time.time() - start
    print(f"\n\nToken count: {token_count}, elapsed: {elapsed:.2f}s, {round(token_count / elapsed)} tokens/s")
