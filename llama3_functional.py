""" NumPy Llama3
"""
from __future__ import annotations

import argparse
import logging
import time
from functools import lru_cache

import numpy as np

import config as cfg
import tokenizer as tkn
import utils as U

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DTYPE = cfg.NP_DTYPE  # typically float16

# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def softmax(x: np.ndarray, *, axis: int = -1) -> np.ndarray:
    x32 = x.astype(np.float32)
    x32 = x32 - np.max(x32, axis=axis, keepdims=True)
    e = np.exp(x32)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-10)


def silu(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -88.0, 88.0)
    return x / (1.0 + np.exp(-x))


def rms_norm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    return x * w / np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)


# -----------------------------------------------------------------------------
# mask & RoPE
# -----------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _mask(seq_len: int, cache_len: int):
    if seq_len == 1:
        return None
    tri = np.triu(np.full((seq_len, seq_len), -np.inf, dtype=DTYPE), 1)
    return np.pad(tri, ((0, 0), (cache_len, 0)), constant_values=0)

def causal_mask(seq_len: int, cache_len: int):
    return _mask(seq_len, cache_len)


def precompute_rope(head_dim: int, max_len: int, base: int = 10000):
    inv = 1.0 / (base ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    t = np.arange(max_len, dtype=np.float32)
    freqs = np.outer(t, inv)
    return np.cos(freqs), np.sin(freqs)  # float32


def _rotate(x, cos, sin):
    xr, xi = x[..., ::2], x[..., 1::2]
    y = np.stack([xr * cos - xi * sin, xr * sin + xi * cos], -1)
    return y.reshape(x.shape)


def apply_rope(q, k, cos, sin):
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    return _rotate(q, cos, sin), _rotate(k, cos, sin)


# -----------------------------------------------------------------------------
# layer modules (weights are *raw*, so we transpose at use‑time)
# -----------------------------------------------------------------------------

def feed_forward(x, w):
    up   = x @ w['up'].T
    gate = silu(x @ w['gate'].T)
    return (gate * up) @ w['down'].T


def attention(x, w, cache, pos, mask, cos, sin, cfg):
    b, t, _ = x.shape
    q = (x @ w['q'].T).reshape(b, t, cfg.n_heads,   cfg.head_dim)
    k = (x @ w['k'].T).reshape(b, t, cfg.n_kv_heads, cfg.head_dim)
    v = (x @ w['v'].T).reshape(b, t, cfg.n_kv_heads, cfg.head_dim)

    q, k = apply_rope(q, k, cos, sin)

    cache['k'][:, pos:pos + t] = k
    cache['v'][:, pos:pos + t] = v
    k_seq = cache['k'][:, : pos + t]
    v_seq = cache['v'][:, : pos + t]

    if cfg.n_heads > cfg.n_kv_heads:
        rep = cfg.n_heads // cfg.n_kv_heads
        k_seq = np.repeat(k_seq, rep, 2)
        v_seq = np.repeat(v_seq, rep, 2)

    q = q.transpose(0, 2, 1, 3)
    k_seq = k_seq.transpose(0, 2, 1, 3)
    v_seq = v_seq.transpose(0, 2, 1, 3)

    scores = np.einsum('bhqd,bhkd->bhqk', q, k_seq) * (1.0 / np.sqrt(cfg.head_dim))
    if mask is not None:
        scores += mask[None, None]
    probs = softmax(scores)
    out = np.einsum('bhqk,bhkd->bhqd', probs, v_seq)
    out = out.transpose(0, 2, 1, 3).reshape(b, t, -1)
    return out @ w['o'].T


def transformer_block(x, w, cache, pos, mask, cos, sin, cfg):
    h = x + attention(rms_norm(x, w['ln1'], cfg.norm_eps), w['attn'], cache, pos, mask, cos, sin, cfg)
    return h + feed_forward(rms_norm(h, w['ln2'], cfg.norm_eps), w['ffn'])


# -----------------------------------------------------------------------------
# weights
# -----------------------------------------------------------------------------

def build_layer(i: int, raw: dict):
    base = f'model.layers.{i}'
    return {
        'ln1': raw[f'{base}.input_layernorm.weight'],
        'ln2': raw[f'{base}.post_attention_layernorm.weight'],
        'attn': {k: raw[f'{base}.self_attn.{k}_proj.weight'] for k in ('q', 'k', 'v', 'o')},
        'ffn':  {k: raw[f'{base}.mlp.{k}_proj.weight']      for k in ('gate', 'up', 'down')},
    }


# -----------------------------------------------------------------------------
# forward + generation
# -----------------------------------------------------------------------------

def forward(ids, pos, raw, caches, cfg, cos32, sin32):
    h = raw['model.embed_tokens.weight'][ids]
    L = h.shape[1]
    cos = cos32[pos:pos + L].astype(DTYPE)
    sin = sin32[pos:pos + L].astype(DTYPE)
    mask = causal_mask(L, pos)

    for i in range(cfg.n_layers):
        h = transformer_block(h, cfg.layers[i], caches[i], pos, mask, cos, sin, cfg)

    h = rms_norm(h, raw['model.norm.weight'], cfg.norm_eps)
    return h[:, [-1]] @ raw['model.embed_tokens.weight'].T


def pick(logits, cfg):
    if cfg.do_sample:
        p = softmax(logits.squeeze() / cfg.temperature)
        return int(np.random.choice(len(p), p=p))
    return int(np.argmax(logits))


def generate(prompt_ids, raw, caches, cfg, tok, cos32, sin32):
    pos = 0
    logits = forward(prompt_ids, pos, raw, caches, cfg, cos32, sin32)
    tok_id = pick(logits, cfg)
    yield tok_id

    for _ in range(cfg.max_new_tokens - 1):
        pos += 1
        logits = forward(np.array([[tok_id]]), pos, raw, caches, cfg, cos32, sin32)
        tok_id = pick(logits, cfg)
        yield tok_id
        if tok_id == tok.eos_id:
            break


# -----------------------------------------------------------------------------
# init + CLI
# -----------------------------------------------------------------------------

def init(model_path: str, cfg):
    raw = U.load_parameters(model_path)
    cfg.layers = [build_layer(i, raw) for i in range(cfg.n_layers)]
    hd = cfg.dim // cfg.n_heads
    cos32, sin32 = precompute_rope(hd, cfg.max_seq_len)

    n_kv = cfg.n_kv_heads or cfg.n_heads
    cache_shape = (cfg.max_batch_size, cfg.max_seq_len, n_kv, hd)
    caches = [{'k': np.zeros(cache_shape, DTYPE), 'v': np.zeros(cache_shape, DTYPE)} for _ in range(cfg.n_layers)]
    return raw, caches, cos32, sin32


def main():
    ap = argparse.ArgumentParser(description="NumPy Llama3 — compact & reference‑exact")
    ap.add_argument('--prompt', default="Once upon a time")
    ap.add_argument('--model', default=cfg.NP_MODEL_PATH)
    ap.add_argument('--tokenizer', default=cfg.NP_TOKENIZER_PATH)
    ap.add_argument('--max_new_tokens', type=int)
    ap.add_argument('--seed', type=int)
    cli = ap.parse_args()

    args = cfg.ModelArgs()
    if cli.max_new_tokens:
        args.max_new_tokens = cli.max_new_tokens
    if cli.seed is not None:
        np.random.seed(cli.seed)
        args.seed = cli.seed
        args.do_sample = True

    tokenizer = tkn.Tokenizer(cli.tokenizer)
    raw, caches, cos32, sin32 = init(cli.model, args)
    prompt_ids = np.array([tokenizer.encode(cli.prompt)], dtype=np.int64)

    print("\nPrompt:", cli.prompt, end="", flush=True)
    start = time.time()
    for tid in generate(prompt_ids, raw, caches, args, tokenizer, cos32, sin32):
        print(tokenizer.decode([tid]), end="", flush=True)
    print(f"\n\nDone in {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
