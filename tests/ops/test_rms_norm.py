import sys
import os
import pytest
import torch
from transformers.models.llama.modeling_llama import LlamaRMSNorm
import numpy as np
from llama3 import rms_norm as llama3_rms_norm
from config import ModelArgs

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def test_rms_norm():
    """
    Test the numpy implementation of RMS normalization against Hugging Face's reference.
    Verifies that output shapes match and values are within acceptable tolerance.
    HF reference: https://github.com/huggingface/transformers/blob/5c47d08b0d6835b8d8fc1c06d9a1bc71f6e78ace/src/transformers/models/llama/modeling_llama.py#L71
    """

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

    # 1. Compute rms_norm using llama3.py implementation
    rms_norm_llama3 = llama3_rms_norm(data_np, weights_np, eps)

    # 2. Compute rms_norm using HF LlamaRMSNorm
    # Instantiate HF LlamaRMSNorm with the specific dimension and epsilon
    hf_rms_norm_layer = LlamaRMSNorm(hidden_size=dim, eps=eps)

    # Set the weights of the HF layer to match our generated weights
    # LlamaRMSNorm has `weight` as a Parameter. We need to assign to its .data
    with torch.no_grad():
        hf_rms_norm_layer.weight.copy_(weights_torch)

    hf_rms_norm_layer.eval()
    with torch.no_grad():
        rms_norm_hf_torch = hf_rms_norm_layer(data_torch)
    rms_norm_hf_np = rms_norm_hf_torch.cpu().numpy()

    # 3. Compare the results
    print()
    print("Llama3 RMSNorm shape:", rms_norm_llama3.shape)
    print("HF RMSNorm shape:", rms_norm_hf_np.shape)
    print()
    assert rms_norm_llama3.shape == rms_norm_hf_np.shape, "RMSNorm output shapes do not match"

    # Using a suitable tolerance for floating point comparisons
    assert np.allclose(rms_norm_llama3, rms_norm_hf_np, atol=1e-5), \
        "RMSNorm implementations do not match"

if __name__ == "__main__":
    pytest.main([__file__])
