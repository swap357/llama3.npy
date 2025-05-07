import pytest
import numpy as np
import torch
import sys
import os

# Add repo root to path to import from root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from llama3 import compute_cos_sin_cache as llama3_compute_cos_sin_cache
from llama3 import apply_rotary_emb as llama3_apply_rotary_emb
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from config import ModelArgs, NP_DTYPE

def test_compute_cos_sin_cache():
    args = ModelArgs()
    head_dim = args.dim // args.n_heads
    max_seq_len = args.max_seq_len
    base = 10000 # Default in Llama

    # 1. llama3.py version
    cos_llama3, sin_llama3 = llama3_compute_cos_sin_cache(head_dim, max_seq_len, base)

    # 2. Hugging Face version (replicated logic for cache generation)
    # We need to replicate the cache generation part of LlamaRotaryEmbedding
    # LlamaRotaryEmbedding uses `dim` which is head_dim, and `max_position_embeddings` which is max_seq_len
    # The following lines for hf_rotary_emb, cos_hf_full, sin_hf_full are removed
    # as the test uses a direct replication of HF logic for comparison.

    # Re-create HF logic more directly for comparison:
    inv_freq_hf = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t_hf = torch.arange(max_seq_len, dtype=torch.float32)
    freqs_hf = torch.outer(t_hf, inv_freq_hf)
    cos_hf_direct = torch.cos(freqs_hf).cpu().numpy().astype(NP_DTYPE)
    sin_hf_direct = torch.sin(freqs_hf).cpu().numpy().astype(NP_DTYPE)

    assert cos_llama3.shape == cos_hf_direct.shape, "cos cache shapes do not match"
    assert sin_llama3.shape == sin_hf_direct.shape, "sin cache shapes do not match"

    assert np.allclose(cos_llama3, cos_hf_direct, atol=1e-6), "cos cache values do not match"
    assert np.allclose(sin_llama3, sin_hf_direct, atol=1e-6), "sin cache values do not match"

# PyTorch equivalent of llama3.py's rotate_half logic for testing
def hf_rotate_half(x):
    # x: (..., head_dim)
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# PyTorch equivalent of llama3.py's apply_rotary_emb logic for testing
# This version will take cos/sin in the shape (seq_len, head_dim/2) like llama3.py does.
def hf_apply_rotary_emb_equivalent(query_states, key_states, cos_cache_half, sin_cache_half, position_ids):
    # query_states, key_states: [batch, seq_len, num_heads, head_dim]
    # cos_cache_half, sin_cache_half: [max_seq_len, head_dim/2]
    # position_ids: [batch, seq_len]

    # Select cos and sin for the given positions, and unsqueeze for broadcasting
    # HF LlamaRotaryEmbedding.forward returns cos/sin of shape [bs, seq_len, 1, dim] or [bs, 1, seq_len, dim] depending on usage
    # Here, we directly use the (max_seq_len, head_dim/2) cache and select based on position_ids
    # Then unsqueeze to (bs, seq_len, 1, head_dim/2) to broadcast over heads
    cos = cos_cache_half[position_ids].unsqueeze(2) # [bs, seq_len, 1, head_dim/2]
    sin = sin_cache_half[position_ids].unsqueeze(2) # [bs, seq_len, 1, head_dim/2]

    # Reshape query and key to separate real and imaginary parts for rotation
    # q_reshaped: [bs, seq_len, num_heads, head_dim/2, 2]
    # k_reshaped: [bs, seq_len, num_heads, head_dim/2, 2]
    def reshape_for_rotation(x):
        return x.float().reshape(*x.shape[:-1], -1, 2)

    query_reshaped = reshape_for_rotation(query_states)
    key_reshaped = reshape_for_rotation(key_states)

    # Split into real and imaginary parts
    q_r, q_i = query_reshaped[..., 0], query_reshaped[..., 1]
    k_r, k_i = key_reshaped[..., 0], key_reshaped[..., 1]

    # Apply rotation: (x_r + i*x_i) * (cos + i*sin) = (x_r*cos - x_i*sin) + i*(x_r*sin + x_i*cos)
    # Or using the formula from llama3.py comment: x_rot_i = x_i * cos - x_i+1 * sin; x_rot_i+1 = x_i * sin + x_i+1 * cos
    # This means q_rotated_real = q_r * cos - q_i * sin
    #             q_rotated_imag = q_r * sin + q_i * cos

    q_rotated_r = q_r * cos - q_i * sin
    q_rotated_i = q_r * sin + q_i * cos
    k_rotated_r = k_r * cos - k_i * sin
    k_rotated_i = k_r * sin + k_i * cos

    # Combine back: stack and reshape to original head_dim
    q_embed = torch.stack((q_rotated_r, q_rotated_i), dim=-1).reshape_as(query_states)
    k_embed = torch.stack((k_rotated_r, k_rotated_i), dim=-1).reshape_as(key_states)

    return q_embed, k_embed

def test_apply_rotary_emb():
    args = ModelArgs()
    head_dim = args.dim // args.n_heads
    n_heads = args.n_heads
    n_kv_heads = args.n_kv_heads or n_heads # Handle GQA
    seq_len = 10 # Test sequence length
    batch_size = 2
    base = 10000

    # 1. Generate cos/sin cache (using llama3.py version, already tested to match HF)
    # Use a max_seq_len that covers the test seq_len
    cos_llama3_cache, sin_llama3_cache = llama3_compute_cos_sin_cache(head_dim, args.max_seq_len, base)

    # Select the portion of cache relevant for this test's seq_len
    # llama3_apply_rotary_emb expects cos/sin for the current sequence length
    cos_active = cos_llama3_cache[:seq_len]
    sin_active = sin_llama3_cache[:seq_len]

    # 2. Generate dummy query (xq) and key (xk) states
    xq_np = np.random.rand(batch_size, seq_len, n_heads, head_dim).astype(NP_DTYPE)
    xk_np = np.random.rand(batch_size, seq_len, n_kv_heads, head_dim).astype(NP_DTYPE)

    # 3. Apply rotary embeddings using llama3.py
    xq_rotated_llama3, xk_rotated_llama3 = llama3_apply_rotary_emb(xq_np.copy(), xk_np.copy(), cos_active, sin_active)

    # 4. Apply rotary embeddings using PyTorch equivalent
    xq_torch = torch.tensor(xq_np)
    xk_torch = torch.tensor(xk_np)

    # The hf_apply_rotary_emb_equivalent expects the full cache and position_ids
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, seq_len)
    cos_torch_cache_half = torch.tensor(cos_llama3_cache) # full cache (max_seq_len, head_dim/2)
    sin_torch_cache_half = torch.tensor(sin_llama3_cache) # full cache (max_seq_len, head_dim/2)

    xq_rotated_hf, xk_rotated_hf = hf_apply_rotary_emb_equivalent(
        xq_torch.clone(), xk_torch.clone(),
        cos_torch_cache_half, sin_torch_cache_half,
        position_ids
    )
    xq_rotated_hf_np = xq_rotated_hf.cpu().numpy()
    xk_rotated_hf_np = xk_rotated_hf.cpu().numpy()

    # 5. Compare results
    assert xq_rotated_llama3.shape == xq_rotated_hf_np.shape, "xq shapes do not match"
    assert xk_rotated_llama3.shape == xk_rotated_hf_np.shape, "xk shapes do not match"

    assert np.allclose(xq_rotated_llama3, xq_rotated_hf_np, atol=1e-5), "xq rotated values do not match"
    assert np.allclose(xk_rotated_llama3, xk_rotated_hf_np, atol=1e-5), "xk rotated values do not match"

if __name__ == "__main__":
    pytest.main([__file__])