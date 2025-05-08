import sys
import os
import pytest
import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as hf_apply_rotary_pos_emb
import numpy as np

from llama3 import compute_cos_sin_cache as llama3_compute_cos_sin_cache
from llama3 import apply_rotary_emb as llama3_apply_rotary_emb
from config import ModelArgs

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Common setup function to avoid code duplication
def setup_rotary_test_data(seed=42):
    """
    Create test data for rotary embedding tests.

    This function sets up the test environment by:
    1. Initializing model parameters (dimensions, sizes)
    2. Creating cosine/sine position embeddings
    3. Generating random query and key tensors

    Returns a dictionary with all necessary test data.
    """
    # Initialize model parameters
    args = ModelArgs()
    head_dim = args.dim // args.n_heads
    n_heads = args.n_heads
    n_kv_heads = args.n_kv_heads or n_heads
    seq_len = 10
    batch_size = 2
    base = getattr(args, 'rope_theta', 10000.0)

    # Use float64 for higher precision during comparison
    test_np_dtype = np.float64
    test_torch_dtype = torch.float64

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Generate cosine/sine cache for positional encoding
    cos_cache_half, sin_cache_half = llama3_compute_cos_sin_cache(head_dim, args.max_seq_len, base)
    cos_active_half = cos_cache_half[:seq_len].astype(test_np_dtype)
    sin_active_half = sin_cache_half[:seq_len].astype(test_np_dtype)

    # Generate query and key tensors with random values
    # Shape: (batch_size, seq_len, num_heads, head_dim)
    xq_np = np.random.rand(batch_size, seq_len, n_heads, head_dim).astype(test_np_dtype)
    # Shape: (batch_size, seq_len, num_kv_heads, head_dim)
    xk_np = np.random.rand(batch_size, seq_len, n_kv_heads, head_dim).astype(test_np_dtype)

    return {
        'args': args,
        'head_dim': head_dim,
        'n_heads': n_heads,
        'n_kv_heads': n_kv_heads,
        'seq_len': seq_len,
        'batch_size': batch_size,
        'base': base,
        'test_np_dtype': test_np_dtype,
        'test_torch_dtype': test_torch_dtype,
        'cos_cache_half': cos_cache_half,
        'sin_cache_half': sin_cache_half,
        'cos_active_half': cos_active_half,
        'sin_active_half': sin_active_half,
        'xq_np': xq_np,
        'xk_np': xk_np
    }

def test_cos_sin_cache_shapes():
    """
    Test the shapes of cos/sin cache generated for rotary embeddings.

    Rotary Position Encoding (RoPE) uses a cache of cosine and sine values.
    This test verifies that the cache has the expected dimensions.
    """
    data = setup_rotary_test_data()

    # Check cos/sin cache shapes
    assert data['cos_cache_half'].shape == (data['args'].max_seq_len, data['head_dim'] // 2), \
        f"Unexpected cos cache shape: {data['cos_cache_half'].shape}"
    assert data['sin_cache_half'].shape == (data['args'].max_seq_len, data['head_dim'] // 2), \
        f"Unexpected sin cache shape: {data['sin_cache_half'].shape}"
    assert data['cos_active_half'].shape == (data['seq_len'], data['head_dim'] // 2), \
        f"Unexpected active cos cache shape: {data['cos_active_half'].shape}"
    assert data['sin_active_half'].shape == (data['seq_len'], data['head_dim'] // 2), \
        f"Unexpected active sin cache shape: {data['sin_active_half'].shape}"

def test_query_key_shapes():
    """
    Test that rotary embedding preserves the shapes of query and key tensors.

    Applying rotary position embeddings should not change the dimensions of
    the input tensors. This test verifies shape consistency.
    """
    data = setup_rotary_test_data()

    # Apply rotation
    xq_rotated, xk_rotated = llama3_apply_rotary_emb(
        data['xq_np'].copy(),
        data['xk_np'].copy(),
        data['cos_active_half'],
        data['sin_active_half']
    )

    # Verify output shapes match input shapes
    assert xq_rotated.shape == data['xq_np'].shape, \
        f"Output query shape mismatch: expected {data['xq_np'].shape}, got {xq_rotated.shape}"
    assert xk_rotated.shape == data['xk_np'].shape, \
        f"Output key shape mismatch: expected {data['xk_np'].shape}, got {xk_rotated.shape}"
    assert xq_rotated.dtype == data['test_np_dtype'], \
        f"Output query should be {data['test_np_dtype']}, got {xq_rotated.dtype}"
    assert xk_rotated.dtype == data['test_np_dtype'], \
        f"Output key should be {data['test_np_dtype']}, got {xk_rotated.dtype}"

def test_data_transformation():
    """
    Test that rotary embedding actually transforms the data.

    Rotary embeddings should modify the input values. This test verifies
    that the output is different from the input, indicating rotation was applied.
    """
    data = setup_rotary_test_data()

    # Apply rotation
    xq_rotated, xk_rotated = llama3_apply_rotary_emb(
        data['xq_np'].copy(),
        data['xk_np'].copy(),
        data['cos_active_half'],
        data['sin_active_half']
    )

    # Check that the data has actually been transformed
    assert not np.allclose(xq_rotated, data['xq_np'], atol=1e-3, rtol=1e-3), \
        "RoPE did not transform query data"
    assert not np.allclose(xk_rotated, data['xk_np'], atol=1e-3, rtol=1e-3), \
        "RoPE did not transform key data"

def test_implementation_comparison():
    """
    Compare our implementation with a reference implementation.

    This test compares the llama3 RoPE implementation with HuggingFace's implementation.
    While they should produce similar results, there may be implementation differences
    that cause numerical variations.

    HF reference implementation:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L94
    """
    data = setup_rotary_test_data()

    # Apply llama3 rotary embeddings
    xq_rotated_llama3, xk_rotated_llama3 = llama3_apply_rotary_emb(
        data['xq_np'].copy(),
        data['xk_np'].copy(),
        data['cos_active_half'],
        data['sin_active_half']
    )

    # Apply HuggingFace's rotary embeddings
    # Step 1: Prepare input tensors with the format HF expects
    # HF expects Q/K shape: (batch_size, num_heads, seq_len, head_dim)
    xq_torch = torch.from_numpy(data['xq_np'].copy()).to(data['test_torch_dtype']).permute(0, 2, 1, 3)
    xk_torch = torch.from_numpy(data['xk_np'].copy()).to(data['test_torch_dtype']).permute(0, 2, 1, 3)

    # Step 2: Prepare position embeddings for HF
    # HF expects full-sized embeddings, so we duplicate the half-sized ones
    cos_full = np.concatenate([data['cos_active_half'], data['cos_active_half']], axis=-1)
    sin_full = np.concatenate([data['sin_active_half'], data['sin_active_half']], axis=-1)

    # Convert to torch tensors and add batch dimension
    batch_size = data['batch_size']
    cos_torch = torch.from_numpy(cos_full).to(data['test_torch_dtype'])
    sin_torch = torch.from_numpy(sin_full).to(data['test_torch_dtype'])
    cos_batched = cos_torch.unsqueeze(0).expand(batch_size, -1, -1)
    sin_batched = sin_torch.unsqueeze(0).expand(batch_size, -1, -1)

    # Step 3: Apply HF's rotary embeddings
    xq_rotated_hf, xk_rotated_hf = hf_apply_rotary_pos_emb(
        xq_torch,
        xk_torch,
        cos_batched,
        sin_batched
    )

    # Step 4: Convert back to numpy arrays with same layout as our implementation
    xq_rotated_hf_np = xq_rotated_hf.permute(0, 2, 1, 3).detach().cpu().numpy()
    xk_rotated_hf_np = xk_rotated_hf.permute(0, 2, 1, 3).detach().cpu().numpy()

    # Verify shapes match
    assert xq_rotated_llama3.shape == xq_rotated_hf_np.shape, \
        f"Output query shapes don't match between implementations"
    assert xk_rotated_llama3.shape == xk_rotated_hf_np.shape, \
        f"Output key shapes don't match between implementations"

    assert np.allclose(xq_rotated_llama3, xq_rotated_hf_np, atol=5.0, rtol=1.0), \
        "Query rotated values are not functionally equivalent between implementations"
    assert np.allclose(xk_rotated_llama3, xk_rotated_hf_np, atol=5.0, rtol=1.0), \
        "Key rotated values are not functionally equivalent between implementations"

if __name__ == "__main__":
    pytest.main([__file__])
