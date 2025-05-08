import pytest
import numpy as np
import torch
import sys
import os

# Add repo root to path to import from root directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from llama3 import compute_cos_sin_cache as llama3_compute_cos_sin_cache
from llama3 import apply_rotary_emb as llama3_apply_rotary_emb
# Import the actual HF RoPE application function from the transformers library
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as hf_apply_rotary_pos_emb
from config import ModelArgs, NP_DTYPE

def test_compute_cos_sin_cache():
    """Verify that the generated cos/sin cache matches reference logic."""
    args = ModelArgs()
    head_dim = args.dim // args.n_heads
    max_seq_len = args.max_seq_len
    base = getattr(args, 'rope_theta', 10000.0) # Use rope_theta from config or default

    # 1. llama3.py version
    cos_llama3, sin_llama3 = llama3_compute_cos_sin_cache(head_dim, max_seq_len, base)

    # 2. Replicate HF Logic using PyTorch for comparison
    inv_freq_hf = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t_hf = torch.arange(max_seq_len, dtype=torch.float32)
    freqs_hf = torch.outer(t_hf, inv_freq_hf)
    cos_hf_direct = torch.cos(freqs_hf).cpu().numpy().astype(NP_DTYPE)
    sin_hf_direct = torch.sin(freqs_hf).cpu().numpy().astype(NP_DTYPE)

    # 3. Compare results
    assert cos_llama3.shape == cos_hf_direct.shape, "cos cache shapes mismatch"
    assert sin_llama3.shape == sin_hf_direct.shape, "sin cache shapes mismatch"

    atol, rtol = (1e-5, 1e-4) if NP_DTYPE == np.float32 else (1e-7, 1e-5)
    assert np.allclose(cos_llama3, cos_hf_direct, atol=atol, rtol=rtol), "cos cache values mismatch"
    assert np.allclose(sin_llama3, sin_hf_direct, atol=atol, rtol=rtol), "sin cache values mismatch"


def test_apply_rotary_emb():
    """Compare the final output of llama3 RoPE application vs HF reference."""
    args = ModelArgs()
    head_dim = args.dim // args.n_heads
    n_heads = args.n_heads
    n_kv_heads = args.n_kv_heads or n_heads
    seq_len = 10
    batch_size = 2
    base = getattr(args, 'rope_theta', 10000.0)
    # Use float64 for higher precision during comparison in this test
    test_np_dtype = np.float64
    test_torch_dtype = torch.float64

    # 1. Generate cos/sin cache for head_dim/2 (as needed by llama3 impl)
    cos_cache_half, sin_cache_half = llama3_compute_cos_sin_cache(head_dim, args.max_seq_len, base)
    cos_active_half = cos_cache_half[:seq_len].astype(test_np_dtype)
    sin_active_half = sin_cache_half[:seq_len].astype(test_np_dtype)

    # 2. Generate dummy query (xq) and key (xk) states
    # Shape: (batch_size, seq_len, num_heads, head_dim)
    xq_np = np.random.rand(batch_size, seq_len, n_heads, head_dim).astype(test_np_dtype)
    # Shape: (batch_size, seq_len, num_kv_heads, head_dim)
    xk_np = np.random.rand(batch_size, seq_len, n_kv_heads, head_dim).astype(test_np_dtype)

    # --- Apply llama3 RoPE ---
    # Input shapes: xq/xk=(b, t, h, d), cos/sin=(t, d/2)
    xq_rotated_llama3, xk_rotated_llama3 = llama3_apply_rotary_emb(
        xq_np.copy(),
        xk_np.copy(),
        cos_active_half,
        sin_active_half
    )

    # --- Apply Hugging Face RoPE ---
    # a) Prepare HF inputs
    # HF expects Q/K shape: (batch_size, num_heads, seq_len, head_dim)
    xq_torch_hf_in = torch.from_numpy(xq_np).to(test_torch_dtype).permute(0, 2, 1, 3)
    xk_torch_hf_in = torch.from_numpy(xk_np).to(test_torch_dtype).permute(0, 2, 1, 3)

    # HF expects cos/sin for the full head_dim, batched
    # Shape: (batch_size, seq_len, head_dim)
    cos_hf_full_dim = np.concatenate((cos_active_half, cos_active_half), axis=-1)
    sin_hf_full_dim = np.concatenate((sin_active_half, sin_active_half), axis=-1)
    cos_torch_input = torch.from_numpy(
        np.ascontiguousarray(np.expand_dims(cos_hf_full_dim, 0).repeat(batch_size, axis=0))
    ).to(test_torch_dtype)
    sin_torch_input = torch.from_numpy(
        np.ascontiguousarray(np.expand_dims(sin_hf_full_dim, 0).repeat(batch_size, axis=0))
    ).to(test_torch_dtype)

    # b) Call HF function
    xq_rotated_hf_permuted, xk_rotated_hf_permuted = hf_apply_rotary_pos_emb(
        xq_torch_hf_in,
        xk_torch_hf_in,
        cos_torch_input,
        sin_torch_input
    )

    # c) Permute HF output back to NumPy layout: (b, t, h, d)
    xq_rotated_hf_np = xq_rotated_hf_permuted.permute(0, 2, 1, 3).detach().cpu().numpy()
    xk_rotated_hf_np = xk_rotated_hf_permuted.permute(0, 2, 1, 3).detach().cpu().numpy()

    # --- Compare Results ---
    assert xq_rotated_llama3.shape == xq_rotated_hf_np.shape, f"xq shape mismatch: llama3={xq_rotated_llama3.shape}, hf={xq_rotated_hf_np.shape}"
    assert xk_rotated_llama3.shape == xk_rotated_hf_np.shape, f"xk shape mismatch: llama3={xk_rotated_llama3.shape}, hf={xk_rotated_hf_np.shape}"

    # Using stricter tolerance because we forced float64 for the test
    atol, rtol = (1e-7, 1e-6)

    assert np.allclose(xq_rotated_llama3, xq_rotated_hf_np, atol=atol, rtol=rtol), "xq rotated values do not match"
    assert np.allclose(xk_rotated_llama3, xk_rotated_hf_np, atol=atol, rtol=rtol), "xk rotated values do not match"

if __name__ == "__main__":
    pytest.main([__file__])
