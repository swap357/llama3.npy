import numpy as np
import math
import time
import sys
import os
import logging
import numba

# Add repo root to path to import from root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
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
        self.capture_activations = {}

    def __call__(self, x, use_jit, capture=False):
        if capture: self.capture_activations['ffn_input'] = x[0, -1, :].copy()

        gate_proj = x @ self.gate
        if capture: self.capture_activations['ffn_gate_proj'] = gate_proj[0, -1, :].copy()

        swish = silu_jit(gate_proj) if use_jit and gate_proj.size > 32768 else silu(gate_proj)
        if capture: self.capture_activations['ffn_silu'] = swish[0, -1, :].copy()

        up_proj = x @ self.up
        if capture: self.capture_activations['ffn_up_proj'] = up_proj[0, -1, :].copy()

        ffn_out = (swish * up_proj) @ self.down
        if capture: self.capture_activations['ffn_output'] = ffn_out[0, -1, :].copy()

        return ffn_out


class Attention:
    def __init__(self, q, k, v, o, args):
        self.n_heads, self.n_kv_heads, self.head_dim = args.n_heads, args.n_kv_heads or args.n_heads, args.dim // args.n_heads
        self.q, self.k, self.v, self.o = q.T, k.T, v.T, o.T
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.k_cache = np.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim), dtype=model_config.NP_DTYPE)
        self.v_cache = np.zeros_like(self.k_cache)
        self.capture_activations = {}

    def __call__(self, x, start_pos, mask, cos, sin, capture=False):
        b, t = x.shape[:2]
        if capture: self.capture_activations['attn_input'] = x[0, -1, :].copy()

        xq, xk, xv = x @ self.q, x @ self.k, x @ self.v
        if capture:
            self.capture_activations['q_proj'] = xq[0, -1, :].copy()
            self.capture_activations['k_proj'] = xk[0, -1, :].copy()
            self.capture_activations['v_proj'] = xv[0, -1, :].copy()

        xq = xq.reshape(b, t, self.n_heads, self.head_dim)
        xk = xk.reshape(b, t, self.n_kv_heads, self.head_dim)
        xv = xv.reshape(b, t, self.n_kv_heads, self.head_dim)

        if capture:
            self.capture_activations['xq_before_rope'] = xq[0, -1, :, :].copy()
            self.capture_activations['xk_before_rope'] = xk[0, -1, :, :].copy()

        xq, xk = apply_rotary_emb(xq, xk, cos, sin)

        if capture:
            self.capture_activations['xq_after_rope'] = xq[0, -1, :, :].copy()
            self.capture_activations['xk_after_rope'] = xk[0, -1, :, :].copy()

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

        if capture: self.capture_activations['scores_before_mask'] = scores[0, :, -1, :].copy()

        if mask is not None: scores += mask[None, None, :, :]

        if capture: self.capture_activations['scores_after_mask'] = scores[0, :, -1, :].copy()

        attn = softmax(scores)

        if capture: self.capture_activations['attn_weights'] = attn[0, :, -1, :].copy()

        out = np.einsum('bhqk,bhkd->bhqd', attn, v_seq)
        final_out = (out.transpose(0, 2, 1, 3).reshape(b, t, -1)) @ self.o

        if capture: self.capture_activations['attn_output'] = final_out[0, -1, :].copy()

        return final_out


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
        self.layer_idx = i

    def __call__(self, x, pos, mask, cos, sin, use_jit, capture_activations=False):
        if capture_activations: self.attn.capture_activations['block_input'] = x[0, -1, :].copy()

        norm_x = self.ln1(x, use_jit)
        if capture_activations: self.attn.capture_activations['norm1_output'] = norm_x[0, -1, :].copy()

        attn_out = self.attn(norm_x, pos, mask, cos, sin, capture=capture_activations)
        h = x + attn_out
        if capture_activations: self.ffn.capture_activations['post_attn_residual'] = h[0, -1, :].copy()

        norm_h = self.ln2(h, use_jit)
        if capture_activations: self.ffn.capture_activations['norm2_output'] = norm_h[0, -1, :].copy()

        ffn_out = self.ffn(norm_h, use_jit, capture=capture_activations)
        block_output = h + ffn_out

        if capture_activations: self.ffn.capture_activations['block_output'] = block_output[0, -1, :].copy()

        return block_output


class NpLlama:
    def __init__(self, path, args, use_jit=False):
        self.args = args
        self.use_jit = use_jit
        weights = model_utils.load_parameters(path)
        self.embed = weights["model.embed_tokens.weight"]
        if "lm_head.weight" in weights:
             self.lm_head = weights["lm_head.weight"].T
             logger.info("Using separate lm_head weights.")
        else:
             self.lm_head = weights["model.embed_tokens.weight"].T
             logger.info("Using shared embedding weights for lm_head.")

        self.cos, self.sin = compute_cos_sin_cache(args.dim // args.n_heads, args.max_seq_len)
        self.layers = [TransformerBlock(weights, i, args) for i in range(args.n_layers)]
        self.norm = RMSNorm(weights["model.norm.weight"], args.norm_eps)

    def forward_capture(self, input_ids, pos, step_idx):
        b, t = input_ids.shape
        hidden = self.embed[input_ids]
        captured_activations = {'initial_embeddings': hidden[0, -1, :].copy()}

        mask = None
        if t > 1:
            mask = np.triu(np.full((t, t), float("-inf"), dtype=model_config.NP_DTYPE), 1)

        cos, sin = self.cos[pos:pos + t], self.sin[pos:pos + t]
        captured_activations['rope_cos'] = cos.copy()
        captured_activations['rope_sin'] = sin.copy()

        target_layer_idx = 0
        for i, layer in enumerate(self.layers):
             capture_this_layer = (i == target_layer_idx)
             hidden = layer(hidden, pos, mask, cos, sin, self.use_jit, capture_activations=capture_this_layer)
             if capture_this_layer:
                 captured_activations.update({f'layer_{i}_attn_{k}': v for k, v in layer.attn.capture_activations.items()})
                 captured_activations.update({f'layer_{i}_ffn_{k}': v for k, v in layer.ffn.capture_activations.items()})
                 layer.attn.capture_activations.clear()
                 layer.ffn.capture_activations.clear()

        captured_activations['final_hidden_state'] = hidden[0, -1, :].copy()

        final_norm = self.norm(hidden, self.use_jit)
        captured_activations['final_norm_output'] = final_norm[0, -1, :].copy()

        logits = final_norm[:, [-1], :] @ self.lm_head
        captured_activations['logits'] = logits[0, -1, :].copy()

        save_dir = "np_activations"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"step_{step_idx}.npz")
        saveable_activations = {k: v for k, v in captured_activations.items() if isinstance(v, np.ndarray)}
        np.savez(save_path, **saveable_activations)
        logger.info(f"Saved {len(saveable_activations)} NumPy activations to {save_path}")

        return logits

    def generate(self, input_ids, max_new_tokens):
        b, t = input_ids.shape
        np.random.seed(self.args.seed)
        generated_ids = []
        current_input_ids = input_ids.copy()

        capture_steps = 10

        for i in range(max_new_tokens):
            pos = 0 if i == 0 else t + i - 1
            inputs_this_step = current_input_ids if i == 0 else current_input_ids[:, [-1]]

            if i < capture_steps:
                logger.info(f"Capturing activations for step {i} (pos {pos}), input shape {inputs_this_step.shape}")
                logits = self.forward_capture(inputs_this_step, pos, i)
            else:
                 logits = self.forward(inputs_this_step, pos)

            if self.args.do_sample:
                logits_last = logits[:, -1, :] / self.args.temperature
                probs = softmax_jit(logits_last) if self.use_jit else softmax(logits_last)
                next_token = np.random.choice(probs.shape[-1], p=probs[0])
                next_id = np.array([[next_token]], dtype=current_input_ids.dtype)
            else:
                logits_last = logits[:, -1, :]
                next_id = np.argmax(logits_last, axis=-1).reshape(b, 1)

            current_input_ids = np.concatenate([current_input_ids, next_id], axis=1)
            generated_ids.append(next_id[0, 0].item())

            yield next_id

    def forward(self, input_ids, pos):
         b, t = input_ids.shape
         hidden = self.embed[input_ids]
         mask = None
         if t > 1:
              mask = np.triu(np.full((t, t), float("-inf"), dtype=model_config.NP_DTYPE), 1)

         cos, sin = self.cos[pos:pos + t], self.sin[pos:pos + t]

         for layer in self.layers:
              hidden = layer(hidden, pos, mask, cos, sin, self.use_jit, capture_activations=False)

         final_norm = self.norm(hidden, self.use_jit)
         logits = final_norm[:, [-1], :] @ self.lm_head
         return logits


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
