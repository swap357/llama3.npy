import sys
import os
import torch
import pytest
import numpy as np
from llama3 import softmax as llama3_softmax

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def test_softmax():
    """
    Test the numpy implementation of softmax against PyTorch reference.
    Verifies that output shapes match and values are within acceptable tolerance.
    """
    # Generate some random data
    # shape: (batch_size, sequence_length, vocab_size-like_dim)
    data_np = np.random.rand(2, 10, 50).astype(np.float32)
    data_torch = torch.tensor(data_np)

    # llama3.py softmax
    softmax_llama3_val = llama3_softmax(data_np)

    # PyTorch softmax
    softmax_torch_val = torch.nn.functional.softmax(data_torch, dim=-1).numpy()

    # Compare shapes
    assert softmax_llama3_val.shape == softmax_torch_val.shape, \
        "softmax output shapes do not match"

    # Compare results
    assert np.allclose(softmax_llama3_val, softmax_torch_val, atol=1e-6), \
        "softmax implementations do not match"

if __name__ == "__main__":
    pytest.main([__file__])
