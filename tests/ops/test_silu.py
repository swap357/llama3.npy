import sys
import os
import pytest
import numpy as np
import torch
from llama3 import silu as llama3_silu

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def test_silu():
    """
    Test the numpy implementation of SiLU activation against PyTorch reference.
    Verifies that output shapes match and values are within acceptable tolerance.
    """
    # Generate some random data
    data_np = (np.random.rand(2, 10, 50).astype(np.float32) - 0.5) * 20
    data_torch = torch.tensor(data_np)

    # llama3.py silu
    silu_llama3_val = llama3_silu(data_np)

    # PyTorch silu
    silu_torch_val = torch.nn.functional.silu(data_torch).cpu().numpy()

    # Compare shapes
    assert silu_llama3_val.shape == silu_torch_val.shape, \
        "silu output shapes do not match"

    # Compare results
    assert np.allclose(silu_llama3_val, silu_torch_val, atol=1e-6), \
        "silu implementations do not match"

if __name__ == "__main__":
    pytest.main([__file__])
