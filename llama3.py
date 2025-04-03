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

    def __call__(self, x, use_jit, capture=False):
        gate_proj = x @ self.gate
        swish = silu_jit(gate_proj) if use_jit and gate_proj.size > 32768 else silu(gate_proj)
        up_proj = x @ self.up
        ffn_out = (swish * up_proj) @ self.down

        if capture:
            return ffn_out, gate_proj, swish, up_proj, ffn_out
        return ffn_out


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
        out = np.einsum('bhqk,bhkd->bhqd', attn, v_seq)
        return (out.transpose(0, 2, 1, 3).reshape(b, t, -1)) @ self.o


class TransformerBlock:
    def __init__(self, weights, i, args):
        prefix = f"model.layers.{i}."
        self.attn = Attention(weights[prefix + "self_attn.q_proj.weight"],
                              weights[prefix + "self_attn.k_proj.weight"],
                              weights[prefix + "self_attn.v_proj.weight"],
                              weights[prefix + "self_attn.o_proj.weight"], args)
        self.ffn = FeedForward(weights[prefix + "mlp.up_proj.weight"],
                               weights[prefix + "mlp.gate_proj.weight"],
                               weights[prefix + "mlp.down_proj.weight"])
        self.ln1 = RMSNorm(weights[prefix + "input_layernorm.weight"], args.norm_eps)
        self.ln2 = RMSNorm(weights[prefix + "post_attention_layernorm.weight"], args.norm_eps)

    def __call__(self, x, pos, mask, cos, sin, use_jit, capture_ffn=False):
        norm_x = self.ln1(x, use_jit)
        attn_out = self.attn(norm_x, pos, mask, cos, sin)
        h = x + attn_out
        norm_h = self.ln2(h, use_jit)
        if capture_ffn:
            out, gate, swish, up, down = self.ffn(norm_h, use_jit, capture=True)
            return h + out, norm_x, attn_out, h, norm_h, up, gate, swish, out, down
        else:
            return h + self.ffn(norm_h, use_jit)


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

    def forward_capture(self, input_ids, pos):
        b, t = input_ids.shape
        hidden = self.embed[input_ids]
        initial_embeddings = hidden.copy()
        mask = None
        if t > 1:
            mask = np.triu(np.full((t, t), float("-inf"), dtype=model_config.NP_DTYPE), 1)
            if pos > 0:
                mask = np.concatenate([np.zeros((t, pos), dtype=model_config.NP_DTYPE), mask], axis=1)
        cos, sin = self.cos[pos:pos + t], self.sin[pos:pos + t]

        (hidden, first_norm, attn_out, residual_1, post_attn_norm,
         ffn_up, ffn_gate, ffn_swish, ffn_output, ffn_down) = self.layers[0](
            hidden, pos, mask, cos, sin, self.use_jit, capture_ffn=True)

        for layer in self.layers[1:]:
            hidden = layer(hidden, pos, mask, cos, sin, self.use_jit)

        final_norm = self.norm(hidden, self.use_jit)
        logits = final_norm[:, [-1], :] @ self.lm_head

        return logits, {
            'embeddings': initial_embeddings[0, -1, :].copy(),
            'first_norm': first_norm[0, -1, :].copy(),
            'attn_output': attn_out[0, -1, :].copy(),
            'residual_1': residual_1[0, -1, :].copy(),
            'post_attn_norm': post_attn_norm[0, -1, :].copy(),
            'ffn_up': ffn_up[0, -1, :].copy(),
            'ffn_gate': ffn_gate[0, -1, :].copy(),
            'ffn_output': ffn_output[0, -1, :].copy(),
            'ffn_down': ffn_down[0, -1, :].copy(),
            'layer_0_output': hidden[0, -1, :].copy(),
            'final_norm': final_norm[0, -1, :].copy(),
            'logits': logits[0, -1, :].copy()
        }

    def generate(self, input_ids, max_new_tokens):
        b, t = input_ids.shape
        np.random.seed(self.args.seed)
        for i in range(max_new_tokens):
            inputs = input_ids if i == 0 else next_id
            pos = 0 if i == 0 else t + i - 1
            if i == 0:
                logits, activations = self.forward_capture(inputs, pos)
                np.savez("np_tensors.npz", **activations)
            else:
                logits = self.forward_capture(inputs, pos)[0]
            if self.args.do_sample:
                logits /= self.args.temperature
                probs = np.exp(logits - np.max(logits))
                probs /= np.sum(probs)
                next_token = np.random.choice(probs.shape[-1], p=probs[0, 0])
                next_id = np.array([[next_token]], dtype=input_ids.dtype)
            else:
                next_id = np.argmax(logits, axis=-1).reshape(b, 1)
            yield next_id


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default="Once upon a time")
    parser.add_argument('--use-jit', action='store_true')
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
