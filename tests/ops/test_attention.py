import pytest
import numpy as np
import torch
import sys
import os
import math # For math.sqrt in HF-like score calculation

# Add repo root to path to import from root directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import llama3 # Your NumPy implementation
from config import ModelArgs, NP_DTYPE # Your model config
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaConfig, apply_rotary_pos_emb, repeat_kv # HF Llama reference

def test_matmul_numpy_vs_torch():
    """
    Compares a basic matrix multiplication between NumPy (often BLAS-backed)
    and PyTorch.
    """
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Define matrix dimensions
    M, K, N = 128, 256, 64

    # Define data types (using float64 for precision, similar to other tests)
    test_np_dtype = np.float64
    test_torch_dtype = torch.float64

    # Generate random matrices with NumPy
    A_np = np.random.randn(M, K).astype(test_np_dtype)
    B_np = np.random.randn(K, N).astype(test_np_dtype)

    # Perform matmul with NumPy
    C_np = A_np @ B_np

    # Convert NumPy matrices to PyTorch tensors
    A_torch = torch.from_numpy(A_np).to(test_torch_dtype)
    B_torch = torch.from_numpy(B_np).to(test_torch_dtype)

    # Perform matmul with PyTorch
    with torch.no_grad():
        C_torch_tensor = torch.matmul(A_torch, B_torch)

    # Convert PyTorch result back to NumPy array
    C_torch_np = C_torch_tensor.cpu().numpy()

    # --- Compare Results ---
    assert C_np.shape == C_torch_np.shape, \
        f"Matmul result shape mismatch: NumPy={C_np.shape}, Torch={C_torch_np.shape}"

    # Use appropriate tolerance for float64 comparison
    atol, rtol = (1e-7, 1e-6) # Consistent with other tests

    assert np.allclose(C_np, C_torch_np, atol=atol, rtol=rtol), \
        "NumPy matmul result does not match PyTorch matmul result."

if __name__ == "__main__":
    pytest.main([__file__])