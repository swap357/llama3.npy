import pytest
import numpy as np
import torch
import sys
import os

# Add repo root to path to import from root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Assuming llama3.silu can be imported directly
from llama3 import silu as llama3_silu

def test_silu():
    # Generate some random data
    data_np = (np.random.rand(2, 10, 50).astype(np.float32) - 0.5) * 20 # Data with negative values
    data_torch = torch.tensor(data_np)

    # llama3.py silu
    silu_llama3_val = llama3_silu(data_np)

    # PyTorch silu
    silu_torch_val = torch.nn.functional.silu(data_torch).cpu().numpy()

    print("Llama3 SiLU shape:", silu_llama3_val.shape)
    print("PyTorch SiLU shape:", silu_torch_val.shape)
    assert silu_llama3_val.shape == silu_torch_val.shape, "SiLU output shapes do not match"

    # Compare results
    assert np.allclose(silu_llama3_val, silu_torch_val, atol=1e-6), \
        "SiLU implementations do not match"

if __name__ == "__main__":
    pytest.main([__file__])