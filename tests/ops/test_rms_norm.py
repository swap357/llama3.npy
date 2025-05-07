import pytest
import numpy as np
import torch
import sys
import os

# Add repo root to path to import from root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from llama3 import rms_norm as llama3_rms_norm
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from config import ModelArgs # For dim and norm_eps, if needed directly

def test_rms_norm():
    # Setup parameters
    args = ModelArgs() # Using default values from config
    dim = args.dim
    eps = args.norm_eps
    batch_size = 2
    seq_len = 5

    # Generate random input data (e.g., hidden states)
    # Shape: (batch_size, seq_len, dim)
    data_np = np.random.rand(batch_size, seq_len, dim).astype(np.float32)
    data_torch = torch.tensor(data_np)

    # Generate random weights for the normalization layer
    # Shape: (dim,)
    weights_np = np.random.rand(dim).astype(np.float32)
    weights_torch = torch.tensor(weights_np)

    # 1. Compute RMSNorm using llama3.py implementation
    # The `use_jit` argument is not actually used in the llama3_rms_norm calculation logic itself.
    # So, we can pass False or True, it won't affect the numerical output for this test.
    rms_norm_llama3 = llama3_rms_norm(data_np, weights_np, eps, use_jit=False)

    # 2. Compute RMSNorm using Hugging Face LlamaRMSNorm
    # Instantiate HF LlamaRMSNorm with the specific dimension and epsilon
    hf_rms_norm_layer = LlamaRMSNorm(hidden_size=dim, eps=eps)

    # Set the weights of the HF layer to match our generated weights
    # LlamaRMSNorm has `weight` as a Parameter. We need to assign to its .data
    with torch.no_grad():
        hf_rms_norm_layer.weight.copy_(weights_torch)

    hf_rms_norm_layer.eval() # Set to evaluation mode
    with torch.no_grad():
        rms_norm_hf_torch = hf_rms_norm_layer(data_torch)
    rms_norm_hf_np = rms_norm_hf_torch.cpu().numpy()

    # 3. Compare the results
    print("Llama3 RMSNorm shape:", rms_norm_llama3.shape)
    print("HF RMSNorm shape:", rms_norm_hf_np.shape)
    assert rms_norm_llama3.shape == rms_norm_hf_np.shape, "RMSNorm output shapes do not match"

    # Using a suitable tolerance for floating point comparisons
    assert np.allclose(rms_norm_llama3, rms_norm_hf_np, atol=1e-5), \
        "RMSNorm implementations do not match"

if __name__ == "__main__":
    pytest.main([__file__])